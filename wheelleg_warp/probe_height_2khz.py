"""复用现有2 kHz轨迹图，核对多高度单侧接触后的轮速、限幅和偏航先后。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]

import numpy as np
import warp as wp
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario,bank_height_v3
from trace_first_divergence_v2 import graph


def run(output):
    heights=(.16,.20,.25,.30,.35,.38)
    cases=[HeightTerrainScenario(speed=v,height_l=.027,stand_height_m=h) for v in (.5,-.5) for h in heights]
    env=NativeEnv(n=len(cases),scenario=cases,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=0)
    env.reset();captured,buffer,pair_buffer=graph(env,True);env.targets.assign(np.zeros((len(cases),3),np.float32))
    chunks=[];pair_chunks=[]
    for step in range(285):
        wp.capture_launch(captured)
        if 4.8<=(step+1)*.02<=5.7:
            chunks.append(buffer.numpy().copy());pair_chunks.append(pair_buffer.numpy().copy())
    trace=np.concatenate(chunks,axis=0);pairs=np.concatenate(pair_chunks,axis=0)
    nq,nv,nu=env.cpu.nq,env.cpu.nv,env.cpu.nu
    pos=1;vel=pos+nq;cmd=vel+nv;flags=cmd+nu;lam=flags+4;basebad=flags+5
    ids=env.ids.numpy();wheel_dofs=np.asarray(ids[8:10]);floor=env.cpu.geom('floor').id;rows=[]
    for w,case in enumerate(cases):
        a=trace[:,w];time=a[:,0];touch=(a[:,flags].astype(int)&1)!=0
        quat=a[:,pos+3:pos+7];qw,qx,qy,qz=quat.T
        yaw=np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        first=lambda mask:int(np.flatnonzero(mask)[0]) if np.any(mask) else None
        contact=first(touch);breach=first(abs(yaw)>=np.deg2rad(5))
        assert contact is not None and breach is not None and contact<breach
        rpm=abs(a[:-1,vel+wheel_dofs])*60/(2*np.pi)
        bound=4.5*np.clip((710-rpm)/(710-490),0,1)
        applied=abs(a[1:,cmd+4:cmd+6]);excess=np.maximum(applied-bound,0)
        assert excess.max()<1e-5,('轮指令超过同状态名义退额',case,excess.max())
        wheel1_sat=np.r_[False,(bound[:,0]>1e-9)&(applied[:,0]>=bound[:,0]-1e-4)]
        period=np.arange(len(time));window=(period>=contact)&(period<=breach)
        own=pairs[:,:,0]==w;ga=pairs[:,:,1];gb=pairs[:,:,2]
        touches=lambda x,y:np.any(own&(((ga==x)&(gb==y))|((ga==y)&(gb==x))),axis=1)
        left_floor=touches(ids[11],floor);left_bump=touches(ids[11],ids[13]);right_floor=touches(ids[12],floor)
        left_any=np.any(own&((ga==ids[11])|(gb==ids[11])),axis=1)
        event=lambda mask:round(float(time[first(mask)]),4) if first(mask) is not None else None
        lost=first(window&(a[:,lam]<.5))
        at=lambda i:np.round(rpm[i-1],2).tolist() if i is not None else None
        rows.append(dict(scenario=asdict(case),contact_s=round(float(time[contact]),4),yaw5_s=round(float(time[breach]),4),
                         first_lambda_zero_s=event(window&(a[:,lam]<.5)),
                         first_base_infeasible_s=event(window&(a[:,basebad]>.5)),
                         first_wheel1_saturated_s=event(window&wheel1_sat),
                         first_wheel1_below_1Nm_s=event(window&np.r_[False,bound[:,0]<1]),
                         wheel_rpm_at_contact=at(contact),wheel_rpm_at_lambda_zero=at(lost),wheel_rpm_at_yaw5=at(breach),
                         max_wheel1_rpm_before_yaw5=round(float(rpm[contact:breach,0].max()),2),
                         min_wheel1_bound_nm_before_yaw5=round(float(bound[contact:breach,0].min()),3),
                         wheel1_saturated_steps_before_yaw5=int((window&wheel1_sat).sum()),
                         left_floor_contact_steps_before_yaw5=int((window&left_floor).sum()),
                         left_bump_contact_steps_before_yaw5=int((window&left_bump).sum()),
                         left_no_contact_steps_before_yaw5=int((window&~left_any).sum()),
                         last_left_bump_contact_s=round(float(time[np.flatnonzero(window&left_bump)[-1]]),4),
                         right_floor_contact_steps_before_yaw5=int((window&right_floor).sum()),
                         window_physical_steps=int(window.sum())))
    source=Path(__file__).resolve().parent
    names=('probe_height_2khz.py','trace_first_divergence_v2.py','native/environment.py','native/controller.py','native/terrain.py','native/models.py')
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'trace.npz',trace=trace,pairs=pairs)
    (output/'summary.json').write_text(json.dumps(dict(source_sha256=hashes,sample_period_s=.0005,
        recorded_interval_s=[round(float(trace[0,0,0]),4),round(float(trace[-1,0,0]),4)],columns=dict(qpos=pos,qvel=vel,ctrl=cmd,contact_mask=flags,lambda_col=lam,base_infeasible=basebad),rows=rows),ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(rows),output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.output)
