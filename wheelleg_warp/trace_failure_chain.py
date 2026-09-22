"""Read-only 2kHz GPU telemetry. No policy updates, alternative controls, or reward changes."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from native.controller import D,V2,allowed,control,polar_jac
from native.environment import begin,command_step,reduce_contacts,after,terrain_attitude
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from dashboard.live_env import atomic_json as write

# Columns are fixed for auditable offline reward and event reconstruction.
COL=['valid','start_s','end_s','reason','return_before','return_after','reward_actual',
     'speed_reward','attitude_cost','residual_cost','smooth_cost','terminal_reward',
     'roll','pitch','yaw','roll_error','pitch_error','command','velocity','lambda','base_infeasible',
     'velocity_limited_mask','nominal_limited_mask']
for stem,count in [('target',6),('filtered',6),('base_raw',6),('base',6),('requested',6),('executed',6),('motor_speed',6),('bound',6),('actual_torque',6)]:
    COL += [f'{stem}_{i}' for i in range(count)]
# Per wheel: actual contacts, target-terrain contacts, normal force, force-weighted
# tangential slip, normal-force sum for averaging. Contact mechanics use pre-integration kinematics.
COL += ['left_contacts','left_target_contacts','left_normal_N','left_slip_weighted','left_load_weight',
        'right_contacts','right_target_contacts','right_normal_N','right_slip_weighted','right_load_weight']
WIDTH=len(COL)

@wp.kernel
def before_physics(slot:int,active:wp.array[int],state:wp.array2d[D],q:wp.array2d[float],v:wp.array2d[float],
                   ids:wp.array[int],targets:wp.array2d[float],cs:wp.array2d[D],diag:wp.array2d[D],yaw:wp.array[D],out:wp.array3d[D]):
    w=wp.tid()
    for k in range(WIDTH):out[slot,w,k]=D(0)
    if active[w]==0:return
    out[slot,w,0]=D(1);out[slot,w,1]=state[w,0]*D(.0005);out[slot,w,4]=state[w,20]
    out[slot,w,19]=diag[w,12];out[slot,w,20]=diag[w,13]
    for k in range(targets.shape[1]):out[slot,w,23+k]=D(targets[w,k]);out[slot,w,29+k]=cs[w,16+k]
    jl=polar_jac(D(q[w,ids[0]]),D(q[w,ids[1]]));jr=polar_jac(D(q[w,ids[2]]),D(q[w,ids[3]]))
    left=jl*V2(cs[w,16]*D(3.4335),cs[w,17]);right=jr*V2(-cs[w,16]*D(3.4335),-cs[w,17])
    wl=cs[w,18]*yaw[3];wr=-wl
    if targets.shape[1]==6:
        left=jl*V2(cs[w,16]*D(3.4335),cs[w,18]);right=jr*V2(cs[w,17]*D(3.4335),cs[w,19])
        wl=cs[w,20]*yaw[3];wr=cs[w,21]*yaw[3]
    out[slot,w,47]=left[0];out[slot,w,48]=left[1];out[slot,w,49]=right[0];out[slot,w,50]=right[1];out[slot,w,51]=wl;out[slot,w,52]=wr
    velocity_mask=int(0);nominal_mask=int(0)
    for k in range(6):
        speed=D(v[w,ids[4+k]]);bound=allowed(speed,k<4);limit=D(4.5)
        if k<4:limit=D(40)
        raw=diag[w,15+k];nominal=wp.clamp(raw,-limit,limit)
        if wp.abs(raw)>limit+D(1.e-9):nominal_mask=nominal_mask|(1<<k)
        if wp.abs(nominal)>bound+D(1.e-9):velocity_mask=velocity_mask|(1<<k)
        out[slot,w,35+k]=raw;out[slot,w,41+k]=diag[w,k];out[slot,w,53+k]=diag[w,6+k]
        out[slot,w,59+k]=speed;out[slot,w,65+k]=bound
    out[slot,w,21]=D(velocity_mask);out[slot,w,22]=D(nominal_mask)

@wp.func
def point_velocity(body:int,w:int,pos:wp.vec3,cvel:wp.array2d[wp.spatial_vector],com:wp.array2d[wp.vec3],root:wp.array[int]):
    return wp.spatial_bottom(cvel[w,body])+wp.cross(wp.spatial_top(cvel[w,body]),pos-com[w,root[body]])

@wp.kernel
def contacts(slot:int,ncon:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],dist:wp.array[float],pos:wp.array[wp.vec3],
             frame:wp.array[wp.mat33],friction:wp.array[vec5],dim:wp.array[int],address:wp.array2d[int],force:wp.array2d[float],
             njmax:int,cone:int,ids:wp.array[int],body:wp.array[int],root:wp.array[int],cvel:wp.array2d[wp.spatial_vector],com:wp.array2d[wp.vec3],out:wp.array3d[D]):
    j=wp.tid()
    if j>=ncon[0]:return
    w=world[j]
    if w<0 or w>=out.shape[1]:return
    if out[slot,w,0]==D(0):return
    a=geom[j][0];b=geom[j][1]
    f=contact_force_fn(cone,frame,friction,dim,address,force,njmax,ncon,w,j,False)
    if dist[j]>0. and f[0]<=0.:return
    relative=point_velocity(body[b],w,pos[j],cvel,com,root)-point_velocity(body[a],w,pos[j],cvel,com,root)
    normal=wp.vec3(frame[j][0,0],frame[j][0,1],frame[j][0,2])
    slip=wp.length(relative-wp.dot(relative,normal)*normal);load=wp.max(f[0],0.)
    for side in range(2):
        if a==ids[11+side] or b==ids[11+side]:
            col=77+5*side
            wp.atomic_add(out,slot,w,col,D(1))
            if (a>=ids[15] and a<=ids[16]) or (b>=ids[15] and b<=ids[16]):wp.atomic_add(out,slot,w,col+1,D(1))
            wp.atomic_add(out,slot,w,col+2,D(load));wp.atomic_add(out,slot,w,col+3,D(load*slip));wp.atomic_add(out,slot,w,col+4,D(load))

@wp.kernel
def reward_parts(slot:int,q:wp.array2d[float],v:wp.array2d[float],param:wp.array2d[D],cmd:wp.array[D],diag:wp.array2d[D],
                 previous:wp.array2d[D],torque:wp.array2d[float],out:wp.array3d[D]):
    w=wp.tid()
    if out[slot,w,0]==D(0):return
    qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
    roll=wp.atan2(D(2)*(qw*qx+qy*qz),D(1)-D(2)*(qx*qx+qy*qy));pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz));ref=terrain_attitude(param,q,w)
    er=roll-ref[0];ep=pitch-ref[1];vx=wp.cos(yaw)*D(v[w,0])+wp.sin(yaw)*D(v[w,1]);err=vx-cmd[w]
    out[slot,w,7]=D(.0005)*wp.exp(-(err/D(.25))*(err/D(.25)))
    out[slot,w,8]=D(.0005)*(er*er+ep*ep+yaw*yaw)/(D(.08726646)*D(.08726646))
    for k in range(6):
        scale=D(4.5)
        if k<4:scale=D(40)
        r=diag[w,6+k]/scale;smooth=(r-previous[w,k])/D(.0005)
        out[slot,w,9]=out[slot,w,9]+D(.0005)*D(.05)*r*r
        out[slot,w,10]=out[slot,w,10]+D(.0005)*D(.0001)*smooth*smooth
        out[slot,w,71+k]=D(torque[w,k])
    out[slot,w,12]=roll;out[slot,w,13]=pitch;out[slot,w,14]=yaw;out[slot,w,15]=er;out[slot,w,16]=ep
    out[slot,w,17]=cmd[w];out[slot,w,18]=vx

@wp.kernel
def after_physics(slot:int,state:wp.array2d[D],done:wp.array[int],out:wp.array3d[D]):
    w=wp.tid()
    if out[slot,w,0]==D(0):return
    out[slot,w,2]=state[w,0]*D(.0005);out[slot,w,3]=D(done[w]);out[slot,w,5]=state[w,20]
    value=state[w,20]-out[slot,w,4];out[slot,w,6]=value
    out[slot,w,11]=value-out[slot,w,7]+out[slot,w,8]+out[slot,w,9]+out[slot,w,10]

class RecordedEnv(TerrainEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);n=self.num_envs
        self.diag=wp.zeros((n,21),dtype=D);self.trace=wp.zeros((40,n,WIDTH),dtype=D)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,n,[self.reward])
            for slot in range(40):
                wp.launch(command_step,n,[self.state,self.param,self.command,self.active,self.data.qpos,self.data.qvel,self.data.qacc_warmstart,self.stopped_q,self.stopped_v,self.stopped_w,self.contact_flags])
                wp.launch(control,n,[self.data.qpos,self.data.qvel,self.data.sensordata,self.targets,self.command,self.active,self.k['state'],self.ids,self.k['heights'],self.k['gains'],self.k['feed'],self.k['angles'],self.k['reference'],self.k['yaw'],self.data.ctrl,self.diag,int(self.project_clipped_base)],block_dim=32)
                wp.launch(before_physics,n,[slot,self.active,self.state,self.data.qpos,self.data.qvel,self.ids,self.targets,self.k['state'],self.diag,self.k['yaw'],self.trace])
                mjw.step(self.model,self.data)
                c=self.data.contact
                wp.launch(contacts,self.data.naconmax,[slot,self.data.nacon,c.worldid,c.geom,c.dist,c.pos,c.frame,c.friction,c.dim,c.efc_address,self.data.efc.force,self.data.njmax,self.model.opt.cone,self.ids,self.model.geom_bodyid,self.model.body_rootid,self.data.cvel,self.data.subtree_com,self.trace])
                wp.launch(reduce_contacts,self.data.naconmax,[self.data.nacon,c.worldid,c.geom,self.ids,self.contact_flags])
                wp.launch(reward_parts,n,[slot,self.data.qpos,self.data.qvel,self.param,self.command,self.diag,self.residual,self.data.actuator_force,self.trace])
                wp.launch(after,n,[self.data.qpos,self.data.qvel,self.data.sensordata,self.data.qacc_warmstart,self.data.time,self.contact_flags,self.ids,self.param,self.command,self.state,self.k['state'],self.diag,self.residual,self.active,self.done,self.reward,self.obs,self.history,self.stopped_q,self.stopped_v,self.stopped_w,self.wheel_offsets],block_dim=32)
                wp.launch(after_physics,n,[slot,self.state,self.done,self.trace])
        self.graph=capture.graph

def analyze(trace):
    get=lambda name:trace[:,COL.index(name)]
    time=get('end_s');first=lambda mask:float(time[np.flatnonzero(mask)[0]]) if np.any(mask) else None
    contact=(get('left_target_contacts')+get('right_target_contacts'))>0
    breach=np.maximum.reduce([abs(get('roll_error')),abs(get('pitch_error')),abs(get('yaw'))])>np.deg2rad(5)
    i=np.flatnonzero(breach);start=int(i[0]) if len(i) else len(trace)
    terminal=get('terminal_reward');expected=np.where(get('reason')!=0,np.where(terminal>0,10.,-10.),0.)
    error=float(np.max(abs(terminal-expected)))
    assert error<1e-7,('reward decomposition mismatch',error)
    assert np.allclose(np.diff(time),.0005,atol=1e-9)
    discount=.99**np.floor(get('start_s')/.02+1e-8)
    limited=get('base_infeasible')>0;contact_time=first(contact)
    window=(time>=contact_time)&(time<contact_time+.3) if contact_time is not None else np.zeros(len(time),bool)
    requested=trace[:,47:53];executed=trace[:,53:59];base=trace[:,41:47];bound=trace[:,65:71]
    np.testing.assert_allclose(executed,requested*get('lambda')[:,None],atol=1e-8,rtol=0.)
    assert np.all(limited==(get('velocity_limited_mask')!=0))
    limits=np.ones_like(requested)
    np.divide(bound-base,requested,out=limits,where=requested>0)
    np.divide(-bound-base,requested,out=limits,where=requested<0)
    feasible_lambda=np.clip(limits.min(axis=1),0,1)
    changed=np.r_[False,np.any(abs(np.diff(trace[:,23:29],axis=0))>1e-6,axis=1)]
    if contact_time is not None:changed &= get('start_s')>=contact_time
    else:changed[:]=False
    command_indices=np.flatnonzero(changed)
    immediate_raw=None;immediate_discounted=None
    if start<len(trace):
        # Re-score only the identical observed prefix. This is not a newly
        # simulated trajectory and does not demonstrate a learned improvement.
        prefix=get('speed_reward')[:start+1]-get('attitude_cost')[:start+1]-get('residual_cost')[:start+1]-get('smooth_cost')[:start+1]
        immediate_raw=float(prefix.sum()-10.)
        immediate_discounted=float(np.sum(discount[:start+1]*prefix)-10.*discount[start])
    return dict(first_target_contact_s=first(contact),first_attitude_failure_s=first(breach),termination_s=float(time[-1]),
        failure_feedback_delay_s=float(time[-1]-time[start]) if start<len(trace) else None,reward=float(get('reward_actual').sum()),
        reward_components={name:float(get(name).sum()) for name in ('speed_reward','attitude_cost','residual_cost','smooth_cost','terminal_reward')},
        discounted_reward=float(np.sum(discount*get('reward_actual'))),discounted_reward_after_breach=float(np.sum(discount[start:]*get('reward_actual')[start:])),
        reward_after_breach=float(get('reward_actual')[start:].sum()),decomposition_max_error=error,
        same_prefix_immediate_failure_reward=immediate_raw,same_prefix_immediate_failure_discounted_reward=immediate_discounted,
        entry_window_seconds=.3,entry_limited_fraction=float(limited[window].mean()) if window.any() else None,
        first_policy_target_change_after_contact_s=float(get('start_s')[command_indices[0]]) if len(command_indices) else None,
        policy_target_change_is_not_proof_of_correct_reaction=True,
        limited_but_nonzero_residual_feasible_steps=int(np.sum(window&limited&(feasible_lambda>1e-6))),
        limited_but_full_residual_feasible_steps=int(np.sum(window&limited&(feasible_lambda>=1.-1e-6))),
        entry_slip_m_s={side:float(get(side+'_slip_weighted')[window].sum()/max(get(side+'_load_weight')[window].sum(),1e-12)) for side in ('left','right')},
        velocity_limit_hits={str(k):int(np.sum((get('velocity_limited_mask')[window].astype(int)&(1<<k))!=0)) for k in range(6)},
        mean_lambda_near_entry=float(get('lambda')[window].mean()) if window.any() else None)

def run(output,check=False):
    source=ROOT/'wheelleg_warp/results/virtual6_capability_20260922'
    baseline=json.loads((source/'baseline.json').read_text())['runs'];seeds=[700000,700102,700202,700302]
    cases=[TerrainScenario(**next(r['scenario'] for r in baseline if r['seed']==s)) for s in seeds]
    if check:
        a=TerrainEnv(1,scenario=cases[0]);b=RecordedEnv(1,scenario=cases[0])
        try:
            a.reset();b.reset();rng=np.random.default_rng(814)
            for _ in range(50):
                action=rng.uniform(-.2,.2,(1,3)).astype(np.float32);x=a.step(action);y=b.step(action)
                np.testing.assert_allclose(x[0],y[0],atol=1e-6,rtol=0.);np.testing.assert_allclose(x[1],y[1],atol=1e-6,rtol=0.)
                t=b.trace.numpy()[:,0];np.testing.assert_allclose(t[:,6].sum(),y[1][0],atol=1e-8)
            print('PASS: telemetry does not change short paired states/rewards',flush=True)
        finally:a.close();b.close()
        return
    output.mkdir(parents=True,exist_ok=False)
    ck=json.loads((source/'protocol.json').read_text())['checkpoint'];selected=[('source',ck,'diff3'),('trained',str(source/'round_001/step_2048000'),'virtual6')]
    summary=[]
    for label,prefix,mode in selected:
        raw=RecordedEnv(len(cases),scenario=cases,residual_mode=mode);env=VecNormalize.load(prefix+'.pkl',raw);env.training=False;env.norm_reward=False
        model=PPO.load(prefix+'.zip',device='cpu');obs=env.reset();pending=set(range(len(cases)));chunks=[[] for _ in cases]
        try:
            while pending:
                obs,_,done,infos=env.step(model.predict(obs,deterministic=True)[0]);chunk=raw.trace.numpy()
                for i in list(pending):
                    chunks[i].append(chunk[chunk[:,i,0]>0,i,:])
                    if done[i]:
                        trace=np.concatenate(chunks[i]);info={k:v for k,v in infos[i].items() if k!='terminal_observation'}
                        row=dict(policy=label,checkpoint=prefix,seed=seeds[i],info=info,**analyze(trace))
                        np.testing.assert_allclose(row['reward'],info['episode']['r'],atol=1e-7)
                        with (output/f'{label}_{seeds[i]}.npz').open('xb') as f:np.savez_compressed(f,trace=trace,columns=np.asarray(COL))
                        summary.append(row);pending.remove(i);print(label,seeds[i],{k:row[k] for k in ('first_target_contact_s','first_attitude_failure_s','termination_s','failure_feedback_delay_s','reward')},info['reason'],flush=True)
        finally:env.close()
    files=[Path(__file__),ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py']
    write(output/'summary.json',dict(runs=summary,recording_hz=2000,gamma=.99,trained=False,changed_policy_or_reward=False,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))

def reanalyze(output):
    summary=json.loads((output/'summary.json').read_text())
    for row in summary['runs']:
        with np.load(output/f"{row['policy']}_{row['seed']}.npz",allow_pickle=False) as data:
            assert data['columns'].tolist()==COL
            trace=data['trace'];assert np.isfinite(trace).all()
            row.update(analyze(trace))
            np.testing.assert_allclose(row['reward'],row['info']['episode']['r'],atol=1e-7)
            assert len(trace)==row['info']['physical_steps']
    summary['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['note']='Collection hashes preserved; offline analysis checks rewards, timing and action mapping without running physics or updating weights.'
    write(output/'summary.json',summary)
    print('PASS: all eight archived traces match reward, timing and executed residual identities')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--analyze-only',action='store_true');a=p.parse_args()
    if not a.check and a.output is None:p.error('--output is required unless --check is used')
    if a.analyze_only:reanalyze(a.output)
    else:run(a.output,a.check)
