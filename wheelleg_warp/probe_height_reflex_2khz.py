"""比较2 kHz可得陀螺仪反射与接触oracle；保留原动作映射和限幅。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]

import mujoco_warp as mjw
import numpy as np
import warp as wp
from native.controller import D,allowed,control
from native.environment import NativeEnv,begin,command_step,reduce_contacts,after
from native.terrain import HeightTerrainScenario,bank_height_v3


@wp.kernel
def choose(mode:int,sensor:wp.array2d[float],ids:wp.array[int],state:wp.array2d[D],param:wp.array2d[D],active:wp.array[int],target:wp.array2d[float],contact_step:wp.array[D]):
    w=wp.tid()
    if active[w]==0:return
    target[w,0]=0.;target[w,1]=0.;target[w,2]=0.
    if mode==1:
        gyro=D(sensor[w,ids[10]+2])
        if wp.abs(gyro)>=D(.05):target[w,2]=float(-wp.sign(gyro))
    elif mode>=2 and (int(state[w,14])&1)!=0:
        if contact_step[w]<D(0):contact_step[w]=state[w,0]
        if mode<=3 and (state[w,0]-contact_step[w])*D(.0005)<D(.1):
            direction=wp.sign(param[w,0])
            if mode==3:direction=-direction
            target[w,2]=float(direction)


@wp.kernel
def right_wheel_oracle(mode:int,state:wp.array2d[D],param:wp.array2d[D],v:wp.array2d[float],ids:wp.array[int],
                       active:wp.array[int],contact_step:wp.array[D],ctrl:wp.array2d[float],diag:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    if mode==6:
        shortfall=diag[w,19]-diag[w,4]
        if wp.abs(shortfall)>D(1.e-6) and wp.abs(diag[w,20]-diag[w,5])<=D(1.e-6):
            bound=allowed(D(v[w,ids[9]]),False);base=D(ctrl[w,5])
            desired=wp.clamp(base-shortfall,-bound,bound)
            ctrl[w,5]=float(desired);diag[w,11]=desired-base
        return
    if mode<4 or contact_step[w]<D(0):return
    if (state[w,0]-contact_step[w])*D(.0005)>=D(.1):return
    bound=allowed(D(v[w,ids[9]]),False);base=D(ctrl[w,5]);direction=wp.sign(param[w,0])
    desired=wp.clamp(base-direction*D(.3),-bound,bound)
    if mode==5:desired=-direction*bound
    ctrl[w,5]=float(desired);diag[w,11]=desired-base


@wp.kernel
def snapshot(slot:int,q:wp.array2d[float],v:wp.array2d[float],sensor:wp.array2d[float],ids:wp.array[int],
             state:wp.array2d[D],target:wp.array2d[float],filtered:wp.array2d[D],diag:wp.array2d[D],out:wp.array3d[D]):
    w=wp.tid();qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
    out[slot,w,0]=state[w,0]*D(.0005)
    out[slot,w,1]=state[w,14]
    out[slot,w,2]=D(sensor[w,ids[10]+2])
    out[slot,w,3]=D(target[w,2]);out[slot,w,4]=filtered[w,18]
    out[slot,w,5]=diag[w,12];out[slot,w,6]=diag[w,10];out[slot,w,7]=diag[w,11]
    out[slot,w,8]=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    out[slot,w,9]=D(v[w,ids[8]])


def run(output,mode):
    heights=(.16,.20,.25,.30,.35,.38)
    cases=[HeightTerrainScenario(speed=v,height_l=.027,stand_height_m=h) for v in (.5,-.5) for h in heights]
    env=NativeEnv(n=len(cases),scenario=cases,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=1)
    env.reset();n=len(cases);telemetry=wp.zeros((40,n,10),dtype=D);contact_step=wp.array(np.full(n,-1.),dtype=D);data=env.data
    with wp.ScopedCapture() as captured:
        wp.launch(begin,n,[env.reward])
        for slot in range(40):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,data.qacc_warmstart,
                env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(choose,n,[mode,data.sensordata,env.ids,env.state,env.param,env.active,env.targets,contact_step])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,env.k['state'],
                env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],
                data.ctrl,env.diag,int(env.project_clipped_base),int(env.grouped_residual)],block_dim=32)
            wp.launch(right_wheel_oracle,n,[mode,env.state,env.param,data.qvel,env.ids,env.active,contact_step,data.ctrl,env.diag])
            mjw.step(env.model,data)
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,data.contact.worldid,data.contact.geom,env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,env.contact_flags,env.ids,
                env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,
                env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
            wp.launch(snapshot,n,[slot,data.qpos,data.qvel,data.sensordata,env.ids,env.state,env.targets,env.k['state'],env.diag,telemetry])
    env.graph=captured.graph;actions=np.zeros((n,3),np.float32);chunks=[];results=[None]*n
    for step in range(700):
        env.step_async(actions)
        if 4.8<=(step+1)*.02<=5.7:chunks.append(telemetry.numpy().copy())
        _,_,done,infos=env.step_wait()
        for i in np.flatnonzero(done):
            if results[i] is None:results[i]=dict(reason=infos[i]['reason'],success=infos[i]['success'],
                peak_deg=infos[i]['peak_deg'],height_rmse_m=infos[i]['height_rmse_m'])
        if all(row is not None for row in results):break
    assert all(row is not None for row in results)
    trace=np.concatenate(chunks,axis=0);rows=[]
    for w,case in enumerate(cases):
        a=trace[:,w];time=a[:,0];contact=(a[:,1].astype(int)&1)!=0
        first=lambda mask:round(float(time[np.flatnonzero(mask)[0]]),4) if np.any(mask) else None
        c=first(contact);assert c is not None
        active=time>=c;before5=active&(abs(a[:,8])<=np.deg2rad(5))
        executed=(abs(a[:,6])>1e-4)|(abs(a[:,7])>1e-4)
        rows.append(dict(scenario=asdict(case),result=results[w],contact_s=c,
            gyro_threshold_s=first(active&(abs(a[:,2])>=.05)),requested_s=first(active&(abs(a[:,3])>.5)),
            filtered_s=first(active&(abs(a[:,4])>.01)),executed_s=first(active&executed),
            lambda_zero_s=first(active&(a[:,5]<.5)),yaw5_s=first(abs(a[:,8])>=np.deg2rad(5)),
            executed_steps_before_yaw5=int((before5&executed).sum())))
    if mode:assert all(row['executed_s'] is not None for row in rows)
    if mode in (1,2,3):assert all(row['requested_s'] is not None for row in rows)
    source=Path(__file__).resolve().parent
    names=('probe_height_reflex_2khz.py','native/environment.py','native/controller.py','native/terrain.py','native/models.py')
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'trace.npz',trace=trace)
    (output/'summary.json').write_text(json.dumps(dict(mode=('baseline','gyro_2khz','contact_oracle_2khz','contact_oracle_opposite_2khz',
        'right_wheel_0_3Nm_oracle','right_wheel_full_bound_oracle','unilateral_wheel_shortfall_reallocation')[mode],
        source_sha256=hashes,sample_period_s=.0005,rows=rows),ensure_ascii=False,indent=2)+'\n')
    print('PASS',mode,sum(row['result']['success'] for row in rows),output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',type=int,choices=range(7),required=True);args=parser.parse_args()
    run(args.output,args.mode)
