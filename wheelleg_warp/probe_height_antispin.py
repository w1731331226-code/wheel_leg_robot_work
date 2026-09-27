"""单一可得轮速/车速防空转规则，与特权载荷触发的相同零力矩上界。"""
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
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5
from native.controller import D,allowed,control
from native.environment import NativeEnv,begin,command_step,reduce_contacts,after
from native.terrain import HeightTerrainScenario,bank_height_v3


@wp.kernel
def gyro_target(mode:int,sensor:wp.array2d[float],ids:wp.array[int],active:wp.array[int],target:wp.array2d[float]):
    w=wp.tid()
    if active[w]==0:return
    target[w,0]=0.;target[w,1]=0.;target[w,2]=0.
    if mode>=3:
        z=D(sensor[w,ids[10]+2])
        if wp.abs(z)>=D(.05):target[w,2]=float(-wp.sign(z))


@wp.kernel
def coast(mode:int,slot:int,q:wp.array2d[float],v:wp.array2d[float],ids:wp.array[int],active:wp.array[int],
          state:wp.array2d[D],loaded:wp.array[int],first_coast:wp.array[D],ctrl:wp.array2d[float],diag:wp.array2d[D],out:wp.array3d[D]):
    w=wp.tid()
    if active[w]==0:return
    qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    forward=wp.cos(yaw)*D(v[w,0])+wp.sin(yaw)*D(v[w,1])
    speed=D(v[w,ids[8]]);surface=wp.abs(speed)*D(.05)
    threshold=D(2)*wp.max(wp.abs(forward),D(.2))
    trigger=((mode==1 or mode==3) and surface>threshold) or ((mode==2 or mode==4) and loaded[w]==0)
    original=D(ctrl[w,4]);applied=original
    if trigger:
        applied=D(0);ctrl[w,4]=0.;diag[w,10]=-original
        if first_coast[w]<D(0):first_coast[w]=state[w,0]*D(.0005)
    out[slot,w,0]=D(q[w,0]);out[slot,w,1]=forward;out[slot,w,2]=surface
    out[slot,w,3]=threshold;out[slot,w,4]=D(1) if trigger else D(0)
    out[slot,w,5]=original;out[slot,w,6]=applied;out[slot,w,7]=D(loaded[w])
    out[slot,w,8]=allowed(speed,False)


@wp.kernel
def clear_loaded(loaded:wp.array[int]):loaded[wp.tid()]=0


@wp.kernel
def mark_loaded(ncon:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],frame:wp.array[wp.mat33],
                friction:wp.array[vec5],dim:wp.array[int],address:wp.array2d[int],force:wp.array2d[float],
                njmax:int,cone:int,ids:wp.array[int],loaded:wp.array[int]):
    j=wp.tid()
    if j>=ncon[0]:return
    w=world[j]
    if w<0 or w>=loaded.shape[0]:return
    pair=geom[j]
    if pair[0]!=ids[11] and pair[1]!=ids[11]:return
    f=contact_force_fn(cone,frame,friction,dim,address,force,njmax,ncon,w,j,False)
    if f[0]>1.:wp.atomic_or(loaded,w,1)


@wp.kernel
def finish(slot:int,q:wp.array2d[float],v:wp.array2d[float],ids:wp.array[int],state:wp.array2d[D],
           diag:wp.array2d[D],torque:wp.array2d[float],loaded:wp.array[int],target:wp.array2d[float],
           filtered:wp.array2d[D],out:wp.array3d[D]):
    w=wp.tid();qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
    out[slot,w,9]=state[w,0]*D(.0005)
    out[slot,w,10]=state[w,14]
    out[slot,w,11]=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    out[slot,w,12]=diag[w,12];out[slot,w,13]=D(loaded[w])
    out[slot,w,14]=D(torque[w,4])
    out[slot,w,15]=D(target[w,2]);out[slot,w,16]=filtered[w,18];out[slot,w,17]=diag[w,11]


def run(output,mode,controls):
    heights=(.16,.20,.25,.30,.35,.38)
    if controls:
        cases=[HeightTerrainScenario(speed=v,stand_height_m=h) for v in (.5,-.5) for h in heights]
        cases += [HeightTerrainScenario(speed=v,terrain='step',step_height_m=.02,relative_attitude=True,stand_height_m=h)
                  for v in (.5,-.5) for h in heights]
    else:cases=[HeightTerrainScenario(speed=v,height_l=.027,stand_height_m=h) for v in (.5,-.5) for h in heights]
    env=NativeEnv(n=len(cases),scenario=cases,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=0)
    env.reset();n=len(cases);data=env.data;loaded=wp.ones(n,dtype=wp.int32)
    first_coast=wp.array(np.full(n,-1.),dtype=D)
    trace=wp.zeros((40,n,18),dtype=D)
    with wp.ScopedCapture() as capture:
        wp.launch(begin,n,[env.reward])
        for slot in range(40):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,data.qacc_warmstart,
                env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(gyro_target,n,[mode,data.sensordata,env.ids,env.active,env.targets])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,env.k['state'],
                env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],
                data.ctrl,env.diag,int(env.project_clipped_base),int(env.grouped_residual)],block_dim=32)
            wp.launch(coast,n,[mode,slot,data.qpos,data.qvel,env.ids,env.active,env.state,loaded,first_coast,data.ctrl,env.diag,trace])
            mjw.step(env.model,data)
            wp.launch(clear_loaded,n,[loaded])
            c=data.contact
            wp.launch(mark_loaded,data.naconmax,[data.nacon,c.worldid,c.geom,c.frame,c.friction,c.dim,c.efc_address,
                data.efc.force,data.njmax,env.model.opt.cone,env.ids,loaded])
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,c.worldid,c.geom,env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,env.contact_flags,env.ids,
                env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,
                env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
            wp.launch(finish,n,[slot,data.qpos,data.qvel,env.ids,env.state,env.diag,data.actuator_force,loaded,
                env.targets,env.k['state'],trace])
    env.graph=capture.graph;zero=np.zeros((n,3),np.float32);chunks=[];results=[None]*n
    for step in range(700):
        env.step_async(zero)
        if 4.8<=(step+1)*.02<=5.7:chunks.append(trace.numpy().copy())
        done_before=env.done.numpy();coast_before=first_coast.numpy()
        _,_,_,infos=env.step_wait()
        for i in np.flatnonzero(done_before):
            if results[i] is None:results[i]=dict(reason=infos[i]['reason'],success=infos[i]['success'],
                peak_deg=infos[i]['peak_deg'],height_rmse_m=infos[i]['height_rmse_m'],
                first_coast_s=float(coast_before[i]) if coast_before[i]>=0 else None)
        if all(row is not None for row in results):break
    assert all(row is not None for row in results)
    data_trace=np.concatenate(chunks,axis=0);rows=[]
    for w,case in enumerate(cases):
        a=data_trace[:,w];time=a[:,9];contact=(a[:,10].astype(int)&(12 if case.terrain=='step' else 1))!=0
        first=lambda mask:round(float(time[np.flatnonzero(mask)[0]]),4) if np.any(mask) else None
        rows.append(dict(scenario=asdict(case),result=results[w],first_contact_s=first(contact),
            first_coast_window_s=first(a[:,4]>.5),first_left_unloaded_s=first(a[:,13]<.5),
            first_gyro_request_s=first(abs(a[:,15])>.5),first_executed_right_residual_s=first(abs(a[:,17])>1.e-4),
            first_yaw5_s=first(abs(a[:,11])>=np.deg2rad(5)),
            max_left_surface_speed_m_s=round(float(a[:,2].max()),3),
            max_actual_vs_applied_torque_error_Nm=round(float(np.max(abs(a[:,14]-a[:,6]))),6)))
    source=Path(__file__).resolve().parent
    names=('probe_height_antispin.py','native/environment.py','native/controller.py','native/terrain.py','native/models.py')
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'trace.npz',trace=data_trace)
    (output/'summary.json').write_text(json.dumps(dict(mode=('baseline','speed_ratio_coast','contact_load_oracle_coast',
        'speed_ratio_coast_plus_gyro','contact_load_oracle_coast_plus_gyro')[mode],
        controls=controls,source_sha256=hashes,sample_period_s=.0005,rows=rows),ensure_ascii=False,indent=2)+'\n')
    print('PASS',mode,controls,sum(row['result']['success'] for row in rows),output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',type=int,choices=range(5),required=True);parser.add_argument('--controls',action='store_true')
    args=parser.parse_args();run(args.output,args.mode,args.controls)
