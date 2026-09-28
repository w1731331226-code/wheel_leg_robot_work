"""同图对照原M3轮差矩：准确2 kHz接触、50 Hz区间末与预瞄触发。"""
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
from native.controller import D,control
from native.environment import NativeEnv,begin,command_step,reduce_contacts,after
from native.terrain import HeightTerrainScenario,bank_height_v3
from probe_height_side_oracle import new_cases


@wp.kernel
def choose(mode:int,state:wp.array2d[D],param:wp.array2d[D],active:wp.array[int],
           target:wp.array2d[float],contact_step:wp.array[D],reference:wp.array[int]):
    w=wp.tid()
    if active[w]==0:return
    target[w,0]=0.;target[w,1]=0.;target[w,2]=0.
    contact=int(state[w,14])&3
    if contact and contact_step[w]<D(0):contact_step[w]=state[w,0]
    step=int(state[w,0]);start=-1
    if mode==1 and contact_step[w]>=D(0):
        start=int(contact_step[w])
    elif mode==2:
        start=reference[w]
    elif mode==3:
        start=reference[w]-200
    if mode and start>=0 and step>=start and step<start+320:
        direction=wp.sign(param[w,0])
        if int(param[w,5])&2:direction=-direction
        target[w,2]=float(direction)


@wp.kernel
def audit(q:wp.array2d[float],state:wp.array2d[D],active:wp.array[int],target:wp.array2d[float],
          filtered:wp.array2d[D],diag:wp.array2d[D],contact_step:wp.array[D],out:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    t=state[w,0]*D(.0005)
    if target[w,2]!=0.:
        out[w,4]=out[w,4]+D(1)
        if out[w,0]<D(0):out[w,0]=t
    if wp.abs(filtered[w,18])>D(.01) and out[w,1]<D(0):out[w,1]=t
    if wp.abs(diag[w,10])+wp.abs(diag[w,11])>D(1.e-5):
        out[w,5]=out[w,5]+D(1)
        if out[w,2]<D(0):out[w,2]=t
    qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    if wp.abs(yaw)>=D(.08726646259971647) and out[w,3]<D(0):out[w,3]=t
    if contact_step[w]>=D(0) and t-contact_step[w]*D(.0005)<D(.16):
        out[w,6]=out[w,6]+D(1)
        if diag[w,12]<D(1.e-10):out[w,7]=out[w,7]+D(1)


def rollout(cases,mode,reference=None):
    n=len(cases);env=NativeEnv(n=n,scenario=cases,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=1)
    env.reset();data=env.data;marker=wp.array(np.full(n,-1.),dtype=D)
    reference=wp.array(np.full(n,-1,np.int32) if reference is None else reference,dtype=wp.int32)
    initial=np.full((n,8),-1.);initial[:,4:]=0.
    stats=wp.array(initial,dtype=D)
    with wp.ScopedCapture() as capture:
        wp.launch(begin,n,[env.reward])
        for _ in range(40):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,data.qacc_warmstart,
                env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(choose,n,[mode,env.state,env.param,env.active,env.targets,marker,reference])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,
                env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],
                env.k['reference'],env.k['yaw'],data.ctrl,env.diag,int(env.project_clipped_base),int(env.grouped_residual)],block_dim=32)
            mjw.step(env.model,data)
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,data.contact.worldid,data.contact.geom,env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,env.contact_flags,env.ids,
                env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,
                env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
            wp.launch(audit,n,[data.qpos,env.state,env.active,env.targets,env.k['state'],env.diag,marker,stats])
    env.graph=capture.graph;zero=np.zeros((n,3),np.float32);results=[None]*n
    for _ in range(700):
        env.step_async(zero)
        done_before=env.done.numpy();marker_before=marker.numpy();stats_before=stats.numpy()
        _,_,_,infos=env.step_wait()
        for i in np.flatnonzero(done_before):
            if results[i] is None:
                info=infos[i];v=stats_before[i]
                results[i]=dict(scenario=asdict(cases[i]),success=info['success'],reason=info['reason'],
                    peak_deg=info['peak_deg'],height_rmse_m=info['height_rmse_m'],velocity_rmse=info['velocity_rmse'],
                    terrain_evidence_passed=info['terrain_evidence_passed'],
                    contact_physical_step=int(marker_before[i]) if marker_before[i]>=0 else None,
                    contact_s=float(marker_before[i]*.0005) if marker_before[i]>=0 else None,
                    first_request_s=float(v[0]) if v[0]>=0 else None,
                    first_filtered_s=float(v[1]) if v[1]>=0 else None,
                    first_executed_s=float(v[2]) if v[2]>=0 else None,
                    first_yaw5_s=float(v[3]) if v[3]>=0 else None,
                    requested_physical_steps=int(v[4]),executed_physical_steps=int(v[5]),
                    lambda_zero_fraction=float(v[7]/v[6]) if v[6]>0 else None)
        if all(item is not None for item in results):break
    assert all(item is not None and item['contact_s'] is not None for item in results)
    if mode:
        bad=[(i,row['first_request_s'],row['first_executed_s'],row['requested_physical_steps'])
             for i,row in enumerate(results) if row['first_request_s'] is None or row['first_executed_s'] is None
             or row['requested_physical_steps']!=320]
        assert not bad,(mode,bad)
        if mode==1:assert all(abs(row['first_request_s']-row['contact_s']-.0005)<1.e-8 for row in results)
    return results


def summarize(rows):
    return dict(success=sum(item['success'] for item in rows),total=len(rows),
        by_height={str(h):sum(item['success'] for item in rows if item['scenario']['stand_height_m']==h)
                   for h in (.16,.20,.25,.30,.35,.38)})


def run(output):
    old_path=ROOT/'wheelleg_warp/results/height_scope_recheck_20260928/boundary_run1/verification.json'
    panels=dict(old_public=[HeightTerrainScenario(**row['scenario']) for row in json.loads(old_path.read_text())['rows']],
                new_public=new_cases())
    results={};summaries={}
    for name,cases in panels.items():
        baseline=rollout(cases,0)
        contact_steps=np.asarray([row['contact_physical_step'] for row in baseline],dtype=np.int32)
        reference=((contact_steps+39)//40)*40
        arms=dict(baseline=baseline,exact_contact=rollout(cases,1),bin_end=rollout(cases,2,reference),
                  preview=rollout(cases,3,reference))
        results[name]=arms;summaries[name]={mode:summarize(rows) for mode,rows in arms.items()}
        print('PANEL',name,summaries[name],flush=True)
    names=('wheelleg_warp/probe_height_exact_contact.py','wheelleg_warp/probe_height_side_oracle.py',
           'wheelleg_warp/native/environment.py','wheelleg_warp/native/controller.py','wheelleg_warp/native/terrain.py',
           'wheelleg_warp/native/models.py','wheelleg_ppo/tools/ppo_env.py')
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    (output/'verification.json').write_text(json.dumps(dict(protocol='same graph M3 side-aware 160ms: zero, exact 2k contact, 50Hz bin end, bin minus 100ms',
        source_sha256=hashes,old_panel_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        summary=summaries,panels=results,training=False,holdout_opened=False),ensure_ascii=False,indent=2)+'\n')
    print('PASS',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.output)
