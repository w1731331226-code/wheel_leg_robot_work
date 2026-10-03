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
from native.terrain import HeightTerrainScenario
from training_contract import source_hashes,digest
import wheelleg_sim as sim

HEIGHTS=(.115,.16,.25,.30,.38)

@wp.kernel
def manual_command(inputs:wp.array2d[D],memory:wp.array2d[D],task:wp.array2d[D],param:wp.array2d[D],
                   command:wp.array[D],q:wp.array2d[float]):
    w=wp.tid();target=inputs[w,0]
    if wp.abs(target)>D(.01):
        inputs[w,2]+=wp.clamp(target-inputs[w,2],D(-.0005),D(.0005))
        command[w]=inputs[w,2]*wp.clamp(task[w,0]*D(.0005)-D(1),D(0),D(1))
        param[w,0]=target;task[w,1]=D(-1)
    else:
        inputs[w,2]=D(0);command[w]=D(0)
        # The shared braking reference reads the same arrival flag and last
        # cruise speed. Keep manual parking open; it has no scored episode.
        if task[w,0]>D(2000):
            if task[w,1]<D(0):task[w,2]=D(q[w,0]);task[w,3]=D(q[w,1])
            task[w,1]=task[w,0]*D(.0005)
    if wp.abs(inputs[w,1])>D(0):
        memory[w,9]+=inputs[w,1]*D(.0005)
        memory[w,10]=D(.1)

class Demo:
    def __init__(self,heights=(.3,),speed=1.,task=False):
        if not np.isfinite(speed) or not .1<=abs(speed)<=1.:raise ValueError('speed must be0.1～1m/s')
        self.env=NativeEnv.height115_candidate(n=len(heights),scenario=[HeightTerrainScenario(speed=speed,stand_height_m=float(h)) for h in heights],shared_reference=True,residual_scale=0.)
        self.env.reset();self.task=task;self.finished=False;self.result=None;self.terminal=None
        self.inputs=wp.zeros((len(heights),3),dtype=D)
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

def run_window(height,speed):
    import glfw
    demo=Demo((height,),speed);m=demo.env.cpu;d=mujoco.MjData(m)
    if not glfw.init():raise RuntimeError('无法初始化图形窗口，请在桌面终端运行')
    window=glfw.create_window(1280,800,'Wheelleg115-380mm | GPU VMC +6-state LQR',None,None)
    if not window:glfw.terminate();demo.close();raise RuntimeError('无法创建图形窗口')
    glfw.make_context_current(window);glfw.swap_interval(1)
    ctx=mujoco.MjrContext(m,mujoco.mjtFontScale.mjFONTSCALE_150.value)
    scene=mujoco.MjvScene(m,2000);opt=mujoco.MjvOption();cam=mujoco.MjvCamera()
    cam.distance=1.5;cam.azimuth=125.;cam.elevation=-18.
    pending=[];target=height;paused=False;direction=1.
    def key_cb(window,key,scancode,action,mods):
        if action!=glfw.PRESS:return
        if glfw.KEY_1<=key<=glfw.KEY_5:pending.append(('reset',HEIGHTS[key-glfw.KEY_1]))
        elif key==glfw.KEY_R:pending.append(('reset',target))
        elif key==glfw.KEY_T:pending.append(('task',target))
        elif key==glfw.KEY_B:pending.append(('direction',None))
        elif key==glfw.KEY_SPACE:pending.append(('pause',None))
        elif key==glfw.KEY_ESCAPE:glfw.set_window_should_close(window,True)
    glfw.set_key_callback(window,key_cb)
    print('W/S前后，A/D转向，↑/↓调高；1～5选择115/160/250/300/380mm并重置；T原行驶停车任务；B任务方向；R手控重置；空格暂停；Esc退出。',flush=True)
    try:
        while not glfw.window_should_close(window):
            start=time.monotonic();glfw.poll_events()
            for action,value in pending:
                if action=='pause':paused=not paused
                elif action=='direction':direction=-direction
                else:
                    demo.close();target=value;demo=Demo((target,),direction*abs(speed),task=action=='task');paused=False
            pending.clear()
            focused=bool(glfw.get_window_attrib(window,glfw.FOCUSED))
            def held(key):return int(focused and glfw.get_key(window,key)==glfw.PRESS)
            if not paused and not demo.finished:
                if not demo.task:
                    target=float(np.clip(target+(held(glfw.KEY_UP)-held(glfw.KEY_DOWN))*.02*.02,.115,.38))
                    demo.set_height(target)
                demo.step(abs(speed)*(held(glfw.KEY_W)-held(glfw.KEY_S)),.3*(held(glfw.KEY_A)-held(glfw.KEY_D)))
            q,v=demo.terminal if demo.terminal else (demo.env.data.qpos.numpy()[0],demo.env.data.qvel.numpy()[0])
            d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);cam.lookat[:]=d.xpos[m.body('base').id] if 'base' in [m.body(i).name for i in range(m.nbody)] else q[:3]
            ids=demo.env.ids.numpy();actual=float(np.mean([sim.fk_joints(q[ids[2*s]],q[ids[2*s+1]])['leg_len'] for s in range(2)]))
            state=demo.env.state.numpy()[0];mode='TASK' if demo.task else 'MANUAL'
            result='PASS' if demo.result and demo.result['success'] else 'FAIL' if demo.finished else 'PAUSED' if paused else 'RUNNING'
            lines=f'{mode}  {result}\nTarget: {target*1000:.1f} mm | Mean FK leg: {actual*1000:.1f} mm\nCommand: {demo.env.command.numpy()[0]:+.2f} m/s | World vx: {v[0]:+.2f} m/s\nTask direction: {direction:+.0f} | Speed: {abs(speed):.2f} m/s'
            if demo.result:
                r=demo.result;lines+=f'\nStop: {r["stop_distance_m"]:.3f}m | Tail: {r["tail_speed_m_s"]:.3f}m/s\nHeight RMSE: {r["height_rmse_m"]*1000:.2f}mm | Reason: {r["reason"]}'
            elif not demo.task:lines+=f'\nActive design margin: {state[38]:+.4f}rad | Manual mode has no task score'
            width,depth=glfw.get_framebuffer_size(window)
            if width and depth:
                viewport=mujoco.MjrRect(0,0,width,depth);mujoco.mjv_updateScene(m,d,opt,None,cam,mujoco.mjtCatBit.mjCAT_ALL.value,scene)
                mujoco.mjr_render(viewport,scene,ctx)
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,lines,'',ctx)
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,viewport,'W/S move | A/D turn | Up/Down height\n1..5 height/reset | T task | B direction | R manual/reset | Space pause | Esc exit','',ctx)
                glfw.swap_buffers(window)
            time.sleep(max(0.,.02-(time.monotonic()-start)))
    finally:ctx.free();glfw.destroy_window(window);glfw.terminate();demo.close()

def check(output):
    output.mkdir(parents=True,exist_ok=False)
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
        state=demo.env.state.numpy();assert np.all(state[:,31:34]>=0) and np.all(state[:,38]>=0)
        assert np.all(state[:,35:37]<=1e-6)
        manual=state.tolist()
    finally:demo.close()
    result=dict(passed=True,baseline=env.baseline_version,heights_m=HEIGHTS,task_runs=rows,manual_state=manual,
        source_sha256=source_hashes(__file__),learning=False,frozen_training_sources_changed=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS five heights xforward/back tasks; five-height manual forward/release-stop; shared current controller',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--height',type=float,default=.3)
    parser.add_argument('--speed',type=float,default=1.);parser.add_argument('--test',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.test:
        if args.output is None:parser.error('--test需要--output新目录')
        check(args.output.resolve())
    else:run_window(args.height,args.speed)
