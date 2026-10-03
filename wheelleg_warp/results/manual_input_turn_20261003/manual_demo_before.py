"""当前115～380mm VMC/LQR GPU控制演示：键盘手控与原行驶停车任务。"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native.environment import NativeEnv,command_step,after,begin,reduce_contacts,collect_physical
from native.shared_reference import update as update_reference
from native.controller import D
from native.terrain import HeightTerrainScenario,HEIGHT_115_GEOMETRIC_MIN
from training_contract import source_hashes,digest
import wheelleg_sim as sim
from fast_physics import PackedPhysics

HEIGHTS=(.115,.16,.25,.30,.38)
SCENES=(
    ('flat','Flat',{}),
    ('asymmetric','Asymmetric bumps',dict(height_l=.015,height_r=.008,offset=.06)),
    ('ramp','Ramp2deg',dict(terrain='ramp',grade_deg=2.,relative_attitude=True)),
    ('cross_slope','Cross slope2deg',dict(terrain='cross_slope',grade_deg=2.,relative_attitude=True,transition_run_m=.4,lateral_margin_m=.35)),
    ('rough','Rough4mm',dict(terrain='rough',roughness_m=.004)),
    ('step','Step15mm',dict(terrain='step',step_height_m=.015)),
    ('mixed','Bumps and rough',dict(terrain='mixed',height_l=.015,height_r=.008,offset=.06,roughness_m=.004)),
    ('rolling_slope','Rolling slope2deg',dict(terrain='rolling_slope',grade_deg=2.,relative_attitude=True)),
    ('multi_step','Multi steps6/12/18mm',dict(terrain='multi_step',step_height_m=.006)),
    ('split_level','Split level10.5mm',dict(terrain='split_level',grade_deg=2.,step_height_m=.3*np.tan(np.deg2rad(2.)),relative_attitude=True,transition_run_m=.4,lateral_margin_m=.35)),
    ('single_side_ramp','Single side ramp2deg',dict(terrain='single_side_ramp',grade_deg=2.,relative_attitude=True)),
    ('asymmetric_rough','Asymmetric rough4mm',dict(terrain='asymmetric_rough',roughness_m=.004)),
)

def scenario(height,speed,scene):
    index=next(i for i,s in enumerate(SCENES) if s[0]==scene)
    return HeightTerrainScenario(speed=speed,stand_height_m=float(height),terrain_seed=313200+index,**SCENES[index][2])

@wp.kernel
def manual_command(inputs:wp.array2d[D],memory:wp.array2d[D],task:wp.array2d[D],param:wp.array2d[D],
                   command:wp.array[D],q:wp.array2d[float]):
    w=wp.tid();target=inputs[w,0]
    if wp.abs(target)>D(.01):
        inputs[w,3]=D(1)
        inputs[w,2]+=wp.clamp(target-inputs[w,2],D(-.0005),D(.0005))
        command[w]=inputs[w,2]*wp.clamp(memory[w,0]-D(1),D(0),D(1))
        param[w,0]=target;task[w,1]=D(-1)
    else:
        inputs[w,2]=D(0);command[w]=D(0)
        # The shared braking reference reads the same arrival flag and last
        # cruise speed. Keep manual parking open; it has no scored episode.
        if inputs[w,3]>D(0) and task[w,0]>D(2000):
            if task[w,1]<D(0):task[w,2]=D(q[w,0]);task[w,3]=D(q[w,1])
            task[w,1]=task[w,0]*D(.0005)
    if wp.abs(inputs[w,1])>D(0):
        memory[w,9]+=inputs[w,1]*D(.0005)
        memory[w,10]=D(.1)

class Demo:
    def __init__(self,heights=(.3,),speed=1.,task=False,scene='flat'):
        if not np.isfinite(speed) or not .5<=abs(speed)<=1.:raise ValueError('speed must be0.5～1m/s')
        self.scene=scene
        self.env=NativeEnv.height115_candidate(n=len(heights),scenario=[scenario(h,speed,scene) for h in heights],shared_reference=True,residual_scale=0.)
        self.env.reset();self.task=task;self.finished=False;self.result=None;self.terminal=None
        self.inputs=wp.zeros((len(heights),4),dtype=D)
        self.height=np.asarray(heights,dtype=float)
        if not task:
            e=self.env;p=e.param.numpy();p[:,2:4]=1.e9;e.param.assign(p)
            with wp.ScopedCapture() as capture:
                wp.launch(begin,e.num_envs,[e.reward])
                for slot in range(40):
                    wp.launch(command_step,e.num_envs,[e.state,e.param,e.command,e.active,e.data.qpos,e.data.qvel,e.data.qacc_warmstart,e.stopped_q,e.stopped_v,e.stopped_w,e.contact_flags])
                    wp.launch(manual_command,e.num_envs,[self.inputs,e.k['state'],e.state,e.param,e.command,e.data.qpos])
                    if slot%10==0:wp.launch(update_reference,e.num_envs,e.shared_reference_args)
                    wp.launch(e.control_kernel,e.num_envs,[e.data.qpos,e.data.qvel,e.data.sensordata,e.targets,e.command,e.active,e.k['state'],e.ids,e.k['heights'],e.k['gains'],e.k['feed'],e.k['angles'],e.k['reference'],e.k['yaw'],e.data.ctrl,e.diag,int(e.project_clipped_base),int(e.grouped_residual)]+e.control_extra,block_dim=32)
                    mjw.step(e.model,e.data)
                    wp.launch(reduce_contacts,e.data.naconmax,[e.data.nacon,e.data.contact.worldid,e.data.contact.geom,e.ids,e.contact_flags])
                    wp.launch(collect_physical,e.num_envs,e.physical_args)
                    wp.launch(after,e.num_envs,[e.data.qpos,e.data.qvel,e.data.sensordata,e.data.qacc_warmstart,e.data.time,e.contact_flags,e.ids,e.param,e.command,e.state,e.k['state'],e.diag,e.residual,e.active,e.done,e.reward,e.obs,e.history,e.stopped_q,e.stopped_v,e.stopped_w,e.wheel_offsets],block_dim=32)
            self.graph=capture.graph

    def set_height(self,heights):
        values=np.broadcast_to(heights,self.height.shape).astype(float)
        if not np.isfinite(values).all() or np.any((values<.115)|(values>.38)):raise ValueError('height must be0.115～0.38m')
        if self.task:raise ValueError('Scored task height remains fixed')
        refs=self.env.k['reference'].numpy();p=self.env.param.numpy()
        for i,h in enumerate(values):refs[i,:3]=[*sim.ik(float(h)),h]
        p[:,13]=values;self.env.k['reference'].assign(refs);self.env.param.assign(p);self.height=values.copy()

    def step(self,speed=0.,turn=0.):
        if self.finished:return
        if self.task:
            _,_,done,infos=self.env.step(np.zeros((self.env.num_envs,3),np.float32))
            if done[0]:
                self.result=infos[0];self.terminal=(self.env.stopped_q.numpy()[0].copy(),self.env.stopped_v.numpy()[0].copy());self.finished=True
        else:
            if not np.isfinite([speed,turn]).all() or abs(speed)>1. or abs(turn)>.3:raise ValueError('Invalid manual command')
            values=self.inputs.numpy();values[:,0]=speed;values[:,1]=turn;self.inputs.assign(values)
            wp.capture_launch(self.graph)
            self.finished=bool(np.any(self.env.done.numpy()))

    def close(self):self.env.close()

    def pose(self):return self.terminal if self.terminal else (self.env.data.qpos.numpy()[0],self.env.data.qvel.numpy()[0])

    @classmethod
    def after_jump(cls,jump):
        """Return to the full ground range without resetting position or velocity."""
        from state_estimation import leg_kinematics
        q,v=jump.pose();height=float(jump.height[0])
        ground=cls((height,),jump.env.scenarios[0].speed,scene=jump.scene);e=ground.env
        e.data.qpos.assign(q[None,:].astype(np.float32));e.data.qvel.assign(v[None,:].astype(np.float32))
        e.data.qacc_warmstart.assign(jump.physics.data.qacc_warmstart.numpy())
        age=max(2.,float(jump.data.time));mjw.forward(e.model,e.data)
        ids=e.ids.numpy();legs=[leg_kinematics(q[ids[2*s:2*s+2]],v[ids[4+2*s:6+2*s]]) for s in range(2)]
        roll,pitch,yaw=sim.euler(jump.data);gyro=e.data.sensordata.numpy()[0,ids[10]:ids[10]+3]
        memory=e.k['state'].numpy();memory[0,0]=age
        memory[0,1]=np.mean([np.linalg.norm(leg[0]) for leg in legs])
        memory[0,2]=np.mean([np.arctan2(leg[0][1],leg[0][0]) for leg in legs])+np.pi/2-pitch
        memory[0,3]=np.mean([leg[3][:,0]@v[ids[4+2*s:6+2*s]] for s,leg in enumerate(legs)])
        memory[0,4]=np.mean([leg[3][:,1]@v[ids[4+2*s:6+2*s]] for s,leg in enumerate(legs)])-gyro[1]
        memory[0,5:9]=[gyro[0],gyro[1],sim.forward_component(v,yaw),gyro[2]]
        memory[0,9]=yaw;e.k['state'].assign(memory)
        # A new manual segment starts its evidence counter at zero. Only the
        # controller is warm; never invent evidence for preceding jump steps.
        values=ground.inputs.numpy();values[0,2]=np.clip(memory[0,7],-1.,1.);values[0,3]=int(jump.control.ever_driven);ground.inputs.assign(values)
        jump.close();return ground

class JumpDemo:
    """Original host jump state machine; actual integration on MuJoCo Warp CUDA."""
    def __init__(self,ground):
        self.env=ground.env;self.scene=ground.scene;self.task=False;self.finished=False;self.result=None
        self.height=np.maximum(ground.height,.16);self.data=mujoco.MjData(self.env.cpu)
        q,v=ground.pose();self.data.qpos[:]=q;self.data.qvel[:]=v;mujoco.mj_forward(self.env.cpu,self.data)
        self.control=sim.make_state(self.env.cpu,hardware=True,six_state=True)
        ids=self.env.ids.numpy();length=float(np.mean([sim.fk_joints(q[ids[2*s]],q[ids[2*s+1]])['leg_len'] for s in range(2)]))
        pitch=sim.euler(self.data)[1]
        angle=float(np.mean([sim.fk_joints(q[ids[2*s]],q[ids[2*s+1]])['phi5'] for s in range(2)]))+np.pi/2-pitch
        st=self.control;st.boot_t=2.;st.L_cur=st.L_prev=length;st.th_prev=angle;st.xy_stop=q[:2].copy()
        st.yaw_target=sim.euler(self.data)[2];st.vf_f=sim.forward_component(v,st.yaw_target)
        st.ever_driven=abs(st.vf_f)>.03;self.set_height(self.height)
        self.pending=True;self.phases=set();self.peak_z=float(q[2]);self.start_z=float(q[2])
        self.physics=PackedPhysics(self.env.cpu,self.data)

    def pose(self):return self.data.qpos.copy(),self.data.qvel.copy()
    def set_height(self,heights):
        h=float(np.clip(np.asarray(heights).reshape(-1)[0],.16,.38))
        self.height=np.array([h]);self.control.leg_ref=h-sim.L_STAND
    def request_jump(self):self.pending=True
    def step(self,speed=0.,turn=0.):
        if self.finished:return
        self.control.cmd_vel=speed;self.control.cmd_turn=turn
        for _ in range(40):
            self.control.cmd_jump=self.pending;self.pending=False
            sim.control(self.env.cpu,self.data,self.control);self.physics.step(self.env.cpu,self.data)
            self.phases.add(self.control.jp);self.peak_z=max(self.peak_z,float(self.data.qpos[2]))
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or max(abs(a) for a in sim.euler(self.data)[:2])>np.deg2rad(40) or self.data.qpos[2]<.02:
                self.finished=True;break
    def close(self):self.env.close()

def show_terrain(model):
    """Display existing collision boxes; change rendering fields only."""
    for i in range(model.ngeom):
        if model.geom(i).name.startswith(('bump_','terrain_')) and model.geom_pos[i,2]>-1.:
            model.geom_group[i]=0;model.geom_matid[i]=-1;model.geom_rgba[i]=[.95,.5,.12,1.]

def run_window(height,speed,scene='flat'):
    import glfw
    demo=Demo((height,),speed,scene=scene);m=demo.env.cpu;show_terrain(m);d=mujoco.MjData(m)
    if not glfw.init():raise RuntimeError('无法初始化图形窗口，请在桌面终端运行')
    window=glfw.create_window(1280,800,'Wheelleg115-380mm | GPU VMC +6-state LQR',None,None)
    if not window:glfw.terminate();demo.close();raise RuntimeError('无法创建图形窗口')
    glfw.make_context_current(window);glfw.swap_interval(1)
    ctx=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150.value)
    render_scene=mujoco.MjvScene(m,2000);opt=mujoco.MjvOption();cam=mujoco.MjvCamera()
    cam.distance=1.8 if scene=='flat' else 3.2;cam.azimuth=125.;cam.elevation=-25.
    pending=[];target=height;paused=False;direction=1.
    def key_cb(window,key,scancode,action,mods):
        if action!=glfw.PRESS:return
        if glfw.KEY_1<=key<=glfw.KEY_5:pending.append(('reset',HEIGHTS[key-glfw.KEY_1]))
        elif glfw.KEY_KP_1<=key<=glfw.KEY_KP_5:pending.append(('reset',HEIGHTS[key-glfw.KEY_KP_1]))
        elif key==glfw.KEY_R:pending.append(('reset',target))
        elif key==glfw.KEY_T:pending.append(('task',target))
        elif key==glfw.KEY_B:pending.append(('direction',None))
        elif key in (glfw.KEY_7,glfw.KEY_KP_7):pending.append(('jump',None))
        elif glfw.KEY_F1<=key<=glfw.KEY_F12:pending.append(('scene',SCENES[key-glfw.KEY_F1][0]))
        elif key==glfw.KEY_SPACE:pending.append(('pause',None))
        elif key==glfw.KEY_ESCAPE:glfw.set_window_should_close(window,True)
    glfw.set_key_callback(window,key_cb)
    glfw.set_scroll_callback(window,lambda window,x,y:setattr(cam,'distance',float(np.clip(cam.distance*.9**y,.5,8.))))
    print('W/S前后，A/D转向，↑/↓调高；1～5高度；7跳跃(GPU物理/原主机状态机)；F1～F12场景；T原任务；B任务方向；R回地面手控；滚轮缩放；空格暂停；Esc退出。',flush=True)
    try:
        while not glfw.window_should_close(window):
            start=time.monotonic();glfw.poll_events()
            for action,value in pending:
                if action=='pause':paused=not paused
                elif action=='direction':direction=-direction
                elif action=='jump':
                    if isinstance(demo,JumpDemo):demo.request_jump()
                    else:demo=JumpDemo(demo);target=float(demo.height[0])
                    paused=False
                else:
                    if action=='scene':scene=value
                    else:target=value
                    demo.close();demo=Demo((target,),direction*abs(speed),task=action=='task',scene=scene);paused=False
                    m=demo.env.cpu;show_terrain(m);d=mujoco.MjData(m)
                    if action=='scene':cam.distance=1.8 if scene=='flat' else 3.2
            pending.clear()
            focused=bool(glfw.get_window_attrib(window,glfw.FOCUSED))
            def held(key):return int(focused and glfw.get_key(window,key)==glfw.PRESS)
            if not paused and not demo.finished:
                if not demo.task:
                    low=.16 if isinstance(demo,JumpDemo) else .115
                    target=float(np.clip(target+(held(glfw.KEY_UP)-held(glfw.KEY_DOWN))*.02*.02,low,.38))
                    demo.set_height(target)
                demo.step(abs(speed)*(held(glfw.KEY_W)-held(glfw.KEY_S)),.3*(held(glfw.KEY_A)-held(glfw.KEY_D)))
                if isinstance(demo,JumpDemo) and not demo.finished and 'LAND' in demo.phases and demo.control.jp=='DRIVE':
                    demo=Demo.after_jump(demo);target=float(demo.height[0]);m=demo.env.cpu;show_terrain(m);d=mujoco.MjData(m)
            q,v=demo.pose()
            d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);cam.lookat[:]=d.xpos[m.body('base').id] if 'base' in [m.body(i).name for i in range(m.nbody)] else q[:3]
            if scene!='flat':cam.lookat[0]+=np.sign(demo.env.scenarios[0].speed)
            ids=demo.env.ids.numpy();actual=float(np.mean([sim.fk_joints(q[ids[2*s]],q[ids[2*s+1]])['leg_len'] for s in range(2)]))
            state=demo.env.state.numpy()[0];mode='TASK' if demo.task else 'MANUAL'
            if isinstance(demo,JumpDemo):mode='GPU JUMP '+demo.control.jp
            result='PASS' if demo.result and demo.result['success'] else 'FAIL' if demo.finished else 'PAUSED' if paused else 'RUNNING'
            title=next(s[1] for s in SCENES if s[0]==scene)
            command=demo.control.cmd_vel if isinstance(demo,JumpDemo) else demo.env.command.numpy()[0]
            lines=f'{mode}  {result}\nScene: {title}\nTarget: {target*1000:.1f} mm | Mean FK leg: {actual*1000:.1f} mm\nCommand: {command:+.2f} m/s | World vx: {v[0]:+.2f} m/s\nTask direction: {direction:+.0f} | Speed: {abs(speed):.2f} m/s'
            if demo.result:
                r=demo.result;lines+=f'\nStop: {r["stop_distance_m"]:.3f}m | Tail: {r["tail_speed_m_s"]:.3f}m/s\nHeight RMSE: {r["height_rmse_m"]*1000:.2f}mm | Reason: {r["reason"]}'
            elif isinstance(demo,JumpDemo):lines+=f'\nGPU physics / host jump control | Target160-380mm\nBody rise from entry: {(demo.peak_z-demo.start_z)*1000:.1f}mm | No ground-task score'
            elif not demo.task:lines+=f'\nActive design margin: {state[38]:+.4f}rad | Manual mode has no task score'
            width,depth=glfw.get_framebuffer_size(window)
            if width and depth:
                viewport=mujoco.MjrRect(0,0,width,depth);mujoco.mjv_updateScene(m,d,opt,None,cam,mujoco.mjtCatBit.mjCAT_ALL.value,render_scene)
                mujoco.mjr_render(viewport,render_scene,ctx)
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,lines,'',ctx)
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,viewport,'W/S move | A/D turn | Up/Down height |7 jump(GPU physics)\n1..5 height | F1..F12 scene | T task | B direction | R ground/reset\nScroll zoom | Space pause | Esc exit','',ctx)
                glfw.swap_buffers(window)
            time.sleep(max(0.,.02-(time.monotonic()-start)))
    finally:ctx.free();glfw.destroy_window(window);glfw.terminate();demo.close()

def check(output):
    output.mkdir(parents=True,exist_ok=False)
    frozen=source_hashes(__file__)
    scenes=[HeightTerrainScenario(speed=v,stand_height_m=h) for h in HEIGHTS for v in (1.,-1.)]
    env=NativeEnv.height115_candidate(n=10,scenario=scenes,shared_reference=True,residual_scale=0.)
    rows=[None]*10
    try:
        env.reset()
        for _ in range(800):
            _,_,done,infos=env.step(np.zeros((10,3),np.float32))
            for i in np.flatnonzero(done):
                if rows[i] is None:rows[i]={k:v for k,v in infos[i].items() if k!='terminal_observation'}
            if all(r is not None for r in rows):break
        assert all(r is not None and r['success'] and r['physical_safety_passed'] and r['design_joint_passed'] for r in rows)
    finally:env.close()
    demo=Demo(HEIGHTS);manual=[]
    try:
        for step in range(500):
            demo.step(speed=1. if 100<=step<200 else 0.)
            assert not demo.finished
        state=demo.env.state.numpy();assert np.all(state[:,31:33]>=HEIGHT_115_GEOMETRIC_MIN) and np.all(state[:,33]>=0) and np.all(state[:,38]>=0)
        assert np.all(state[:,35:37]<=1e-6)
        manual=state.tolist()
    finally:demo.close()
    demo=Demo((.3,));transition=[]
    try:
        for target in (.115,.38,.3):
            for _ in range(int(np.ceil(abs(float(demo.height[0])-target)/.0004))+100):
                h=float(demo.height[0]);demo.set_height(h+np.clip(target-h,-.0004,.0004));demo.step()
                assert not demo.finished
            q=demo.env.data.qpos.numpy()[0];ids=demo.env.ids.numpy()
            actual=float(np.mean([sim.fk_joints(q[ids[2*s]],q[ids[2*s+1]])['leg_len'] for s in range(2)]))
            assert abs(actual-target)<=.02
            transition.append(dict(target_m=target,actual_m=actual))
        for _ in range(100):demo.step(turn=.3);assert not demo.finished
        for _ in range(100):demo.step(turn=-.3);assert not demo.finished
        state=demo.env.state.numpy()[0]
        assert min(state[31:33])>=HEIGHT_115_GEOMETRIC_MIN and state[33]>=0 and state[38]>=0 and max(state[35:37])<=1e-6
    finally:demo.close()
    obstacle_scenes=[scenario(h,v,scene[0]) for scene in SCENES for h in HEIGHTS for v in (1.,-1.)]
    env=NativeEnv.height115_candidate(n=len(obstacle_scenes),scenario=obstacle_scenes,shared_reference=True,residual_scale=0.)
    obstacle_rows=[None]*len(obstacle_scenes)
    try:
        env.reset()
        for _ in range(800):
            _,_,done,infos=env.step(np.zeros((len(obstacle_scenes),3),np.float32))
            for i in np.flatnonzero(done):
                if obstacle_rows[i] is None:obstacle_rows[i]=dict(scene=SCENES[i//10][0],height_m=obstacle_scenes[i].stand_height_m,**{k:v for k,v in infos[i].items() if k!='terminal_observation'})
            if all(r is not None for r in obstacle_rows):break
        assert all(r is not None and r['physical_steps']==r['physical_evidence_steps'] for r in obstacle_rows)
    finally:env.close()
    jumps=[]
    for h in (.115,.16,.3,.38):
        ground=Demo((h,))
        for _ in range(100):ground.step()
        jump=JumpDemo(ground)
        try:
            for _ in range(350):
                jump.step();assert not jump.finished
            assert {'SQUAT','JUMP','FLY','LAND','DRIVE'}.issubset(jump.phases)
            assert jump.control.jp=='DRIVE' and sim.wheel_contact(jump.env.cpu,jump.data)
            assert str(jump.physics.data.qpos.device).startswith('cuda')
            jumps.append(dict(entry_height_m=h,target_height_m=float(jump.height[0]),phases=sorted(jump.phases),
                body_rise_from_entry_m=jump.peak_z-jump.start_z,final_vertical_velocity=float(jump.data.qvel[2]),
                physical_device=str(jump.physics.data.qpos.device),host_control=True,returned_to_ground=True))
        finally:jump.close()
    assert source_hashes(__file__)==frozen,'Source changed during check'
    result=dict(passed=True,baseline=env.baseline_version,heights_m=HEIGHTS,task_runs=rows,manual_state=manual,
        height_transitions=transition,manual_turns_checked=True,obstacle_runs=obstacle_rows,jumps=jumps,
        source_sha256={**frozen,'wheelleg_warp/fast_physics.py':digest(Path(__file__).resolve().parent/'fast_physics.py')},learning=False,frozen_training_sources_changed=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS five heights xforward/back tasks; five-height manual forward/release-stop; shared current controller',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--height',type=float,default=.3)
    parser.add_argument('--speed',type=float,default=1.);parser.add_argument('--scene',choices=[s[0] for s in SCENES],default='flat');parser.add_argument('--test',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.test:
        if args.output is None:parser.error('--test需要--output新目录')
        check(args.output.resolve())
    else:run_window(args.height,args.speed,args.scene)
