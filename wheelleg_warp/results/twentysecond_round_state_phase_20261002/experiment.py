"""One causal state-phase trial; unchanged motor domain and physical task gates."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from native.environment import NativeEnv,begin,command_step,reduce_contacts,collect_physical,after
from native.controller import D,fk,polar_jac
from native.terrain import model,HeightTerrainScenario
from probe_height_115_margin import cases
from select_braking_common_action import lqr_cost
from model_lqr import vmc_coordinates
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent
PLANS=OUT.parent/'eighteenth_round_robust_reference_20261002'


@wp.kernel
def features(q:wp.array2d[float],v:wp.array2d[float],sensor:wp.array2d[float],ids:wp.array[int],
             heights:wp.array[D],angles:wp.array[D],memory:wp.array2d[D],out:wp.array2d[D]):
    w=wp.tid();qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
    pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    al=D(q[w,ids[0]]);bl=D(q[w,ids[1]]);ar=D(q[w,ids[2]]);br=D(q[w,ids[3]])
    left=fk(al,bl);right=fk(ar,br);jl=polar_jac(al,bl);jr=polar_jac(ar,br)
    length=(left[3]+right[3])/D(2);index=int(0)
    for knot in range(1,heights.shape[0]-1):
        if length>heights[knot]:index=knot
    ratio=wp.clamp((length-heights[index])/(heights[index+1]-heights[index]),D(0),D(1))
    eq=(D(1)-ratio)*angles[index]+ratio*angles[index+1]
    rate=(jl[0,1]*D(v[w,ids[4]])+jl[1,1]*D(v[w,ids[5]])+jr[0,1]*D(v[w,ids[6]])+jr[1,1]*D(v[w,ids[7]]))/D(2)
    gyro=D(sensor[w,ids[10]+1]);position=D(0)
    if memory[w,13]>D(0):position=wp.cos(yaw)*(D(q[w,0])-memory[w,14])+wp.sin(yaw)*(D(q[w,1])-memory[w,15])
    out[w,0]=(left[2]+right[2])/D(2)+D(1.5707963267948966)-pitch-eq
    out[w,1]=rate-gyro;out[w,2]=position
    out[w,3]=wp.cos(yaw)*D(v[w,0])+wp.sin(yaw)*D(v[w,1])
    out[w,4]=pitch;out[w,5]=gyro;out[w,6]=memory[w,13]


def graph(env):
    n=env.num_envs;d=env.data
    with wp.ScopedCapture() as capture:
        wp.launch(begin,n,[env.reward])
        for _ in range(10):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(env.control_kernel,n,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
            mjw.step(env.model,d)
            wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags])
            wp.launch(collect_physical,n,env.physical_args)
            wp.launch(after,n,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    return capture.graph


def run(mode):
    output=OUT/mode;output.mkdir()
    plans=[np.load(PLANS/f'solve_world{d}/best.npz',allow_pickle=False)['schedule'] for d in range(2)]
    if mode=='template':
        scenes=cases()[4:6];templates=None;metric=None
    else:
        source=OUT.parent/'twentyfirst_round_native_reference_20261002/cell_v2/registered_cases.json'
        scenes=[HeightTerrainScenario(**s) for s in json.loads(source.read_text())['scenarios']]
        saved=np.load(OUT/'template/trajectory.npz',allow_pickle=False)
        templates=[saved[f'features{d}'] for d in range(2)]
        ref,Q,R,P=lqr_cost(model(HeightTerrainScenario(stand_height_m=.115)))
        c,_,_=vmc_coordinates(ref);lift=np.linalg.pinv(c[:6]);metric=lift.T@P@lift
        assert np.linalg.eigvalsh(metric).min()>0
        np.savez_compressed(output/'metric.npz',P=P,C=c[:6],lift=lift,metric=metric)
    registration=dict(mode=mode,scenarios=[asdict(s) for s in scenes],
        phase_rule='Original clock plus same existing position-hold exit.' if mode=='clock_full' else 'Nearest causal six-state template under lifted original P, velocity normalized by observed entry speed; nondecreasing phase, max2 slots/5ms (ceil(1/0.75)); latch zero target on existing position hold.',
        weight_rule='Original full amplitude, no interpolation.' if mode in ('state_phase_full','clock_full') else 'Keep prior v2 height/speed weights unchanged, no third weight search.',
        limits=dict(motor_Nm=1.,request_L1_per5ms=.1),learning=False,full_admission=False)
    (output/'registered_cases.json').write_text(json.dumps(registration,indent=2)+'\n')
    env=NativeEnv.height115_candidate(n=len(scenes),scenario=scenes,residual_scale=0,nominal_correction=True)
    n=len(scenes)
    try:
        env.reset();execution=graph(env);feature=wp.zeros((n,7),dtype=D)
        args=[env.data.qpos,env.data.qvel,env.data.sensordata,env.ids,env.k['heights'],env.k['angles'],env.k['state'],feature]
        cursor=np.full(n,-1,int);phase=np.zeros(n,int);entry_speed=np.zeros(n);finished=np.zeros(n,bool)
        current=np.zeros((n,6));recorded_features=[[] for _ in scenes];records=[];all_requests=[];first_bad=[None]*n
        for iteration in range(2800):
            state=env.state.numpy();active=env.active.numpy()!=0
            wp.launch(features,n,args);x=feature.numpy();target=current.copy()
            for w in np.flatnonzero(active & (state[:,1]>=0)):
                d=int(scenes[w].speed<0);cursor[w]+=1
                if cursor[w]==0:entry_speed[w]=x[w,3]
                assert abs(entry_speed[w])>1e-6
                if mode=='template':
                    target[w]=plans[d][min(cursor[w],399)];recorded_features[w].append(x[w].copy())
                else:
                    weight=np.clip((abs(scenes[w].speed)-.75)/.25,0,1)*np.clip((.12-scenes[w].stand_height_m)/(.12-.115),0,1)
                    if mode in ('state_phase_full','clock_full'):weight=1.
                    finished[w] |= x[w,6]>0
                    if weight==0 or finished[w]:target[w]=0
                    else:
                        if mode=='clock_full':phase[w]=min(cursor[w],399)
                        else:
                            reference=templates[d];end=np.flatnonzero(reference[:,6]>0)
                            last=int(end[0])-1 if len(end) else len(reference)-1
                            candidates=np.arange(phase[w],min(phase[w]+2,last)+1)
                            if not len(candidates):candidates=np.array([phase[w]])
                            query=x[w,:6].copy();query[3]*=reference[0,3]/entry_speed[w]
                            delta=reference[candidates,:6]-query
                            costs=np.einsum('ni,ij,nj->n',delta,metric,delta)
                            phase[w]=int(candidates[np.argmin(costs)])
                        target[w]=weight*plans[d][phase[w]]
            delta=target-current
            current=np.clip(current+delta*np.minimum(1.,.1/np.maximum(abs(delta).sum(axis=1),1e-30))[:,None],-1,1)
            env.set_nominal_correction(current);wp.capture_launch(execution)
            now=env.state.numpy()
            for w in range(n):
                if first_bad[w] is None and now[w,38]<0:first_bad[w]=dict(end_s=float(now[w,0]*.0005),arrival_s=float(now[w,1]),margin=float(now[w,38]))
            records.append(np.c_[state[:,0],state[:,1],cursor,phase,finished,entry_speed,x]);all_requests.append(current.copy())
            if env.done.numpy().all():break
            if iteration%500==0:print('PROGRESS',mode,iteration,flush=True)
        assert env.done.numpy().all()
        rows=[{k:v for k,v in r.items() if k!='terminal_observation'} for r in env.step_wait()[3]]
        assert all(e['physical_steps']==e['physical_evidence_steps'] for e in rows)
        arrays=dict(records=np.array(records),requests=np.array(all_requests))
        if mode=='template':
            assert all(e['success'] for e in rows)
            for w,values in enumerate(recorded_features):arrays[f'features{w}']=np.array(values)
        np.savez_compressed(output/'trajectory.npz',**arrays)
        result=dict(episodes=rows,first_design_failure=first_bad,total=n,physical=sum(e['physical_safety_passed'] for e in rows),
            design=sum(e['design_joint_passed'] for e in rows),success=sum(e['success'] for e in rows),learning=False,full_admission=False,
            trajectory_sha256=sha(output/'trajectory.npz'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                (Path(__file__),ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('RESULT',mode,result['physical'],result['design'],result['success'],'/',n,flush=True)
        print('FAILED',[(w,scenes[w].stand_height_m,scenes[w].speed,e['stop_distance_m'],e['tail_speed_m_s'],e['min_active_design_margin_rad']) for w,e in enumerate(rows) if not e['success']],flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('template','state_phase','state_phase_full','clock_full'));run(parser.parse_args().mode)
