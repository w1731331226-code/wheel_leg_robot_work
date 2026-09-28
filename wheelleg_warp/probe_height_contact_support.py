"""原M3第1通道的左右接触触发差动支撑受限上界。"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import mujoco_warp as mjw
import numpy as np
import warp as wp
from native.controller import D,control
from native.environment import NativeEnv,begin,command_step,reduce_contacts,after
from native.terrain import HeightTerrainScenario,bank_height_v3


@wp.kernel
def request(polarity:int,state:wp.array2d[D],active:wp.array[int],target:wp.array2d[float],contact_step:wp.array[D]):
    w=wp.tid()
    if active[w]==0:return
    target[w,0]=0.;target[w,1]=0.;target[w,2]=0.
    if polarity==0:return
    side=int(state[w,14])&3
    if side and contact_step[w]<D(0):contact_step[w]=state[w,0]
    if contact_step[w]>=D(0) and (state[w,0]-contact_step[w])*D(.0005)<D(.1):
        direction=D(1)
        if side&2:direction=D(-1)
        target[w,0]=float(D(polarity)*direction)


@wp.kernel
def measure(state:wp.array2d[D],active:wp.array[int],contact_step:wp.array[D],
            filtered:wp.array2d[D],diag:wp.array2d[D],out:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0 or contact_step[w]<D(0):return
    if (state[w,0]-contact_step[w])*D(.0005)>=D(.1):return
    out[w,0]=out[w,0]+D(1)
    if diag[w,12]<D(.999999):out[w,1]=out[w,1]+D(1)
    out[w,2]=out[w,2]+wp.abs(filtered[w,16])
    for j in range(4):out[w,3]=out[w,3]+wp.abs(diag[w,6+j])


def rollout(cases,polarity):
    n=len(cases);env=NativeEnv(n=n,scenario=cases,bank_factory=bank_height_v3,height_conditioned=True,
                               residual_scale=1 if polarity else 0)
    env.reset();data=env.data;contact_step=wp.array(np.full(n,-1.),dtype=D);stats=wp.zeros((n,4),dtype=D)
    with wp.ScopedCapture() as capture:
        wp.launch(begin,n,[env.reward])
        for _ in range(40):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,data.qacc_warmstart,
                env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(request,n,[polarity,env.state,env.active,env.targets,contact_step])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,env.k['state'],
                env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],
                data.ctrl,env.diag,int(env.project_clipped_base),int(env.grouped_residual)],block_dim=32)
            wp.launch(measure,n,[env.state,env.active,contact_step,env.k['state'],env.diag,stats])
            mjw.step(env.model,data)
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,data.contact.worldid,data.contact.geom,env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,env.contact_flags,env.ids,
                env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,
                env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    env.graph=capture.graph;zero=np.zeros((n,3),np.float32);results=[None]*n
    for _ in range(700):
        env.step_async(zero)
        done_before=env.done.numpy();stats_before=stats.numpy();contact_before=contact_step.numpy()
        _,_,_,infos=env.step_wait()
        for i in np.flatnonzero(done_before):
            if results[i] is None:
                info=infos[i];s=stats_before[i]
                results[i]=dict(scenario=asdict(cases[i]),success=info['success'],reason=info['reason'],
                    peak_deg=info['peak_deg'],height_rmse_m=info['height_rmse_m'],
                    contact_s=round(float(contact_before[i]*.0005),4) if contact_before[i]>=0 else None,
                    window_steps=int(s[0]),lambda_limited_fraction=float(s[1]/s[0]) if s[0]>0 else None,
                    mean_filtered_action=float(s[2]/s[0]) if s[0]>0 else None,
                    mean_executed_leg_torque_sum_Nm=float(s[3]/s[0]) if s[0]>0 else None)
        if all(row is not None for row in results):break
    assert all(row is not None for row in results)
    if polarity:assert all(row['contact_s'] is not None and row['window_steps']>0 for row in results)
    return results


def summarize(rows):
    return dict(success=sum(row['success'] for row in rows),total=len(rows),
                by_height={str(h):sum(row['success'] for row in rows if row['scenario']['stand_height_m']==h)
                           for h in (.16,.20,.25,.30,.35,.38)})


def run(output):
    panel=Path(__file__).resolve().parents[1]/'wheelleg_warp/results/height_scope_recheck_20260928/boundary_run1/verification.json'
    cases=[HeightTerrainScenario(**row['scenario']) for row in json.loads(panel.read_text())['rows']]
    assert len(cases)==48 and all(max(s.height_l,s.height_r)==.02 for s in cases)
    groups={name:rollout(cases,mode) for name,mode in (('baseline',0),('positive',1),('negative',-1))}
    summary={name:summarize(rows) for name,rows in groups.items()}
    root=Path(__file__).resolve().parents[1]
    names=('wheelleg_warp/probe_height_contact_support.py','wheelleg_warp/native/environment.py',
           'wheelleg_warp/native/controller.py','wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    (output/'verification.json').write_text(json.dumps(dict(protocol='first contact side, 2 kHz, M3 first action ±side, 100ms',
        source_sha256=hashes,panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),summary=summary,rows=groups,
        privileged_contact=True,training=False,holdout_opened=False),ensure_ascii=False,indent=2)+'\n')
    print('PASS',summary,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.output)
