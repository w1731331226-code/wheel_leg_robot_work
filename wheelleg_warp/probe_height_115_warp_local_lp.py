"""CPU局部LP的三个恒定小增量在Warp同图公开平地轨迹上复核。"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco_warp as mjw
import numpy as np
import warp as wp
from native.controller import D,allowed,control
from native.environment import NativeEnv,begin,command_step,reduce_contacts,after,wheel_center
from native.terrain import HEIGHT_115_GEOMETRIC_MIN,HeightTerrainScenario,bank_height_115
from trace_first_divergence_v2 import ordered_contacts


@wp.kernel
def pulse(qvel:wp.array2d[float],ctrl:wp.array2d[float],state:wp.array2d[D],active:wp.array[int],
          ids:wp.array[int],mode:wp.array[int],start:wp.array[int],delta:wp.array2d[D],
          archived:wp.array3d[D],stats:wp.array2d[D]):
    w=wp.tid();step=int(state[w,0])
    if active[w]==0 or step<start[w] or step>=start[w]+40:return
    slot=step-start[w]
    for j in range(6):
        stats[w,2]=wp.max(stats[w,2],wp.abs(D(ctrl[w,j])-archived[w,slot,j]))
    if mode[w]==0:return
    stats[w,0]=stats[w,0]+D(1)
    for j in range(6):
        speed=D(qvel[w,ids[4+j]]);bound=allowed(speed,j<4)
        baseline=D(ctrl[w,j])
        if mode[w]==1:baseline=archived[w,slot,j]
        request=baseline+delta[w,j]
        applied=wp.clamp(request,-bound,bound)
        if wp.abs(request-applied)>D(1.e-8):stats[w,1]=stats[w,1]+D(1)
        ctrl[w,j]=float(applied)


@wp.kernel
def capture_start(clock:wp.array[float],q:wp.array2d[float],v:wp.array2d[float],
                  warm:wp.array2d[float],ctrl:wp.array2d[float],state:wp.array2d[D],
                  active:wp.array[int],start:wp.array[int],out:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0 or int(state[w,0])!=start[w]:return
    nq=q.shape[1];nv=v.shape[1];nu=ctrl.shape[1]
    out[w,0]=D(clock[w])
    for j in range(nq):out[w,1+j]=D(q[w,j])
    for j in range(nv):
        out[w,1+nq+j]=D(v[w,j])
        out[w,1+nq+nv+j]=D(warm[w,j])
    for j in range(nu):out[w,1+nq+2*nv+j]=D(ctrl[w,j])


@wp.kernel
def sample(slot:int,q:wp.array2d[float],state:wp.array2d[D],active:wp.array[int],ids:wp.array[int],
           passive:wp.array[int],offsets:wp.array3d[wp.vec3d],out:wp.array3d[D]):
    w=wp.tid()
    out[slot,w,0]=state[w,0]*D(.0005)+D(.0005)
    for j in range(1,6):out[slot,w,j]=D(0)
    out[slot,w,6]=D(active[w])
    if active[w]==0:return
    rotation=wp.quatd(D(q[w,4]),D(q[w,5]),D(q[w,6]),D(q[w,3]))
    root=wp.vec3d(D(q[w,0]),D(q[w,1]),D(q[w,2]))
    shortest=D(1);margin=D(1)
    for side in range(2):
        hip=root+wp.quat_rotate(rotation,wp.vec3d(D(.075),offsets[w,side,0][1],D(0)))
        wheel=wheel_center(q,ids,offsets,w,side)
        shortest=wp.min(shortest,wp.length(wheel-hip))
    for j in range(4):
        margin=wp.min(margin,D(1.5)-wp.abs(D(q[w,ids[j]])))
        margin=wp.min(margin,D(2.5)-wp.abs(D(q[w,passive[j]])))
    qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
    roll=wp.atan2(D(2)*(qw*qx+qy*qz),D(1)-D(2)*(qx*qx+qy*qy))
    pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    out[slot,w,1]=shortest;out[slot,w,2]=margin
    out[slot,w,3]=wp.abs(roll);out[slot,w,4]=wp.abs(pitch);out[slot,w,5]=wp.abs(yaw)


def run(output):
    assert not output.exists()
    source=ROOT/'wheelleg_warp/results/height_115_local_lp_20260928/verification.json'
    lp=json.loads(source.read_text())
    assert lp['summary']['nonlinear_verified_same_contact']==3
    ref=ROOT/'wheelleg_warp/results/height_115_local_states_20260928/verification.json'
    archive=json.loads(ref.read_text())
    for record in (lp,archive):
        assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
                   for name,digest in record['source_sha256'].items())
    assert hashlib.sha256((source.parent/'linearization.npz').read_bytes()).hexdigest()==lp['linearization_sha256']
    assert hashlib.sha256((ref.parent/'windows.npz').read_bytes()).hexdigest()==archive['windows_sha256']
    selected=[next((i,e) for i,e in enumerate(archive['events'])
                   if e['world']==row['world'] and e['kind']==row['event']) for row in lp['rows']]
    events=[item[1] for item in selected]
    base=[HeightTerrainScenario(**event['scenario']) for event in events]
    scenarios=base*3;n=len(scenarios)
    starts=np.asarray([round(row['start_s']/.0005) for row in lp['rows']]*3,np.int32)
    windows=np.load(ref.parent/'windows.npz');columns=archive['columns']
    sequences=[];reference_pre=[]
    for row,(event_id,_) in zip(lp['rows'],selected):
        pre=windows[f'event_{event_id}_pre']
        k=int(np.argmin(abs(pre[:,0]-row['start_s'])))
        assert k+40<=len(pre) and abs(pre[k,0]-row['start_s'])<1.e-6
        reference_pre.append(pre[k])
        sequences.append(pre[k:k+40,columns['ctrl_start']:columns['ctrl_start']+6])
    archived_ctrl=np.tile(np.asarray(sequences),(3,1,1))
    delta=np.zeros((n,6));delta[len(base):]=np.tile([row['candidate']['delta_torque_Nm'] for row in lp['rows']],(2,1))
    env=NativeEnv(n=n,scenario=scenarios,bank_factory=bank_height_115,
                  height_conditioned=True,height_design='range115',residual_scale=0)
    env.reset();mode=wp.array(np.repeat(np.arange(3,dtype=np.int32),len(base)),dtype=wp.int32)
    start=wp.array(starts,dtype=wp.int32);actions=wp.array(delta,dtype=D)
    archived_actions=wp.array(archived_ctrl,dtype=D)
    stats=wp.zeros((n,3),dtype=D)
    start_state=wp.zeros((n,1+env.cpu.nq+2*env.cpu.nv+env.cpu.nu),dtype=D)
    passive=wp.array([env.cpu.joint(name).qposadr[0] for name in
                      ('passA_L','passC_L','passA_R','passC_R')],dtype=wp.int32)
    data=env.data;trace=wp.zeros((40,n,7),dtype=D)
    pairs=wp.zeros((40,data.naconmax,3),dtype=wp.int32)
    with wp.ScopedCapture() as captured:
        wp.launch(begin,n,[env.reward])
        for slot in range(40):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,
                data.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,
                env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],
                env.k['reference'],env.k['yaw'],data.ctrl,env.diag,0,0],block_dim=32)
            wp.launch(capture_start,n,[data.time,data.qpos,data.qvel,data.qacc_warmstart,data.ctrl,
                env.state,env.active,start,start_state])
            wp.launch(pulse,n,[data.qvel,data.ctrl,env.state,env.active,env.ids,mode,start,actions,
                archived_actions,stats])
            mjw.step(env.model,data)
            wp.launch(sample,n,[slot,data.qpos,env.state,env.active,env.ids,passive,env.wheel_offsets,trace])
            wp.launch(ordered_contacts,data.naconmax,[slot,data.nacon,data.contact.worldid,data.contact.geom,pairs])
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,data.contact.worldid,data.contact.geom,
                env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,
                env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,
                env.residual,env.active,env.done,env.reward,env.obs,env.history,
                env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    env.targets.assign(np.zeros((n,3),np.float32))
    trace_rows=[[] for _ in range(n)];contact_rows=[[] for _ in range(n)];contact_raw=[[] for _ in range(n)]
    for period in range(700):
        wp.capture_launch(captured.graph)
        chunk_begin=period*40
        if any(chunk_begin<step+40 and chunk_begin+40>step for step in starts):
            values=trace.numpy();contact=pairs.numpy()
            for w in range(n):
                selected=np.flatnonzero((np.arange(40)+chunk_begin>=starts[w])&
                                        (np.arange(40)+chunk_begin<starts[w]+40))
                if len(selected):
                    trace_rows[w].append(values[selected,w].copy())
                    for slot in selected:
                        contact_raw[w].append(contact[slot].copy())
                        contact_rows[w].append({tuple(sorted((int(a),int(b)))) for world,a,b in contact[slot]
                                                if int(world)==w})
        if np.all(env.done.numpy()!=0):break
    assert np.all(env.done.numpy()!=0)
    rows=[];terminal=env.state.numpy();terminal_reason=env.done.numpy();pulse_stats=stats.numpy()
    recorded_start=start_state.numpy();mode_names=('live_baseline','archived_ctrl_plus_delta','live_ctrl_plus_delta')
    for w,scenario in enumerate(scenarios):
        data=np.concatenate(trace_rows[w]);assert len(data)==40 and len(contact_rows[w])==40
        reference=reference_pre[w%len(base)];pre=recorded_start[w]
        assert abs(pre[0]-starts[w]*.0005)<1.e-6
        start_diffs=dict(qpos=float(np.max(abs(pre[1:1+env.cpu.nq]-reference[1:1+env.cpu.nq]))),
            qvel=float(np.max(abs(pre[1+env.cpu.nq:1+env.cpu.nq+env.cpu.nv]-
                                  reference[1+env.cpu.nq:1+env.cpu.nq+env.cpu.nv]))),
            warmstart=float(np.max(abs(pre[1+env.cpu.nq+env.cpu.nv:1+env.cpu.nq+2*env.cpu.nv]-
                                       reference[1+env.cpu.nq+env.cpu.nv:1+env.cpu.nq+2*env.cpu.nv]))),
            base_ctrl=float(np.max(abs(pre[1+env.cpu.nq+2*env.cpu.nv:]-reference[1+env.cpu.nq+2*env.cpu.nv:]))))
        active_steps=int(np.count_nonzero(data[:,6]>.5))
        window_safe=bool(active_steps==40 and np.min(data[:,1])>=HEIGHT_115_GEOMETRIC_MIN and np.min(data[:,2])>=0 and
                         np.max(data[:,3:6])<=np.deg2rad(5))
        rows.append(dict(scenario=asdict(scenario),mode=mode_names[w//len(base)],
            start_s=float(starts[w]*.0005),delta_torque_Nm=delta[w].tolist(),
            min_actual_leg_m=float(np.min(data[:,1])),min_joint_margin_rad=float(np.min(data[:,2])),
            max_abs_attitude_rad=np.max(data[:,3:6],axis=0).tolist(),
            active_physical_steps=active_steps,window_safe=window_safe,start_vs_archived_pre_max_abs=start_diffs,
            max_live_vs_archived_base_ctrl_Nm=float(pulse_stats[w,2]),
            first_actual_below_proxy_s=next((float(r[0]) for r in data if r[1]<HEIGHT_115_GEOMETRIC_MIN),None),
            first_joint_limit_s=next((float(r[0]) for r in data if r[2]<0),None),
            applied_physical_steps=int(pulse_stats[w,0]),clipped_motor_physical_commands=int(pulse_stats[w,1]),
            terminal_reason_code=int(terminal_reason[w]),full_episode_task_success=bool(terminal[w,19]),
            full_episode_min_fk_leg_m=float(terminal[w,30])))
    for arm in (1,2):
        for i in range(len(base)):
            w=arm*len(base)+i;before=recorded_start[i];current=recorded_start[w]
            rows[w]['start_vs_live_baseline_pre_max_abs']=dict(
                qpos=float(np.max(abs(current[1:1+env.cpu.nq]-before[1:1+env.cpu.nq]))),
                qvel=float(np.max(abs(current[1+env.cpu.nq:1+env.cpu.nq+env.cpu.nv]-
                                      before[1+env.cpu.nq:1+env.cpu.nq+env.cpu.nv]))),
                warmstart=float(np.max(abs(current[1+env.cpu.nq+env.cpu.nv:1+env.cpu.nq+2*env.cpu.nv]-
                                           before[1+env.cpu.nq+env.cpu.nv:1+env.cpu.nq+2*env.cpu.nv]))),
                base_ctrl=float(np.max(abs(current[1+env.cpu.nq+2*env.cpu.nv:]-before[1+env.cpu.nq+2*env.cpu.nv:]))))
            rows[w]['contact_pair_steps_equal_vs_live_baseline']=sum(
                a==b for a,b in zip(contact_rows[i],contact_rows[w]))
    summary={}
    for arm,name in enumerate(mode_names):
        group=rows[arm*len(base):(arm+1)*len(base)]
        summary[name]=dict(total=len(group),window_safe=sum(row['window_safe'] for row in group),
            pulse_complete=sum(row['applied_physical_steps']==40 for row in group) if arm else 0,
            clipped_motor_physical_commands=sum(row['clipped_motor_physical_commands'] for row in group),
            full_episode_task_success=sum(row['full_episode_task_success'] for row in group),
            max_start_vs_archived_pre_qpos=max(row['start_vs_archived_pre_max_abs']['qpos'] for row in group),
            max_live_vs_archived_base_ctrl_Nm=max(row['max_live_vs_archived_base_ctrl_Nm'] for row in group))
        if arm:
            summary[name]['same_contact_steps_vs_live_baseline']=sum(a==b for i in range(len(base))
                for a,b in zip(contact_rows[i],contact_rows[arm*len(base)+i]))
            summary[name]['max_start_vs_live_baseline_qpos']=max(
                row['start_vs_live_baseline_pre_max_abs']['qpos'] for row in group)
            summary[name]['max_start_vs_live_baseline_qvel']=max(
                row['start_vs_live_baseline_pre_max_abs']['qvel'] for row in group)
            summary[name]['max_start_vs_live_baseline_warmstart']=max(
                row['start_vs_live_baseline_pre_max_abs']['warmstart'] for row in group)
            summary[name]['max_start_vs_live_baseline_base_ctrl']=max(
                row['start_vs_live_baseline_pre_max_abs']['base_ctrl'] for row in group)
    sources=('wheelleg_warp/probe_height_115_warp_local_lp.py','wheelleg_warp/probe_height_115_local_lp.py',
             'wheelleg_warp/trace_first_divergence_v2.py',
             'wheelleg_warp/native/controller.py','wheelleg_warp/native/environment.py',
             'wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py',
             'wheelleg_ppo/tools/rm_controller.py','wheelleg_ppo/tools/model_lqr.py',
             'wheelleg_ppo/tools/wheelleg_sim.py','wheelleg_ppo/tools/hardware_profile.py',
             'wheelleg_ppo/tools/state_estimation.py',
             'wheelleg_ppo/xml/wheelleg.xml')
    output.mkdir(parents=True)
    np.savez_compressed(output/'trace.npz',window_trace=np.stack([np.concatenate(items) for items in trace_rows]),
                        contact_pairs=np.stack([np.stack(items) for items in contact_raw]),
                        start_state=recorded_start)
    (output/'verification.json').write_text(json.dumps(dict(role='public_privileged_timed_warp_check_of_cpu_local_lp',
        nominal_target_m=.115,horizon_s=.02,geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,
        arms=mode_names,archived_ctrl_is_open_loop=True,live_ctrl_recomputed_each_step=True,
        actual_chain_and_joint_gate_scope='selected_40_physics_steps_only',
        training=False,final_holdout_opened=False,summary=summary,rows=rows,
        source_lp_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256((output/'trace.npz').read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}),
        ensure_ascii=False,indent=2)+'\n')
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
