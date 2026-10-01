"""Causal 2kHz windows for the coordinated request trial; no parameter search."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from native.controller import D,fk,polar_jac,inverse2,V2
from native.environment import NativeEnv,begin,command_step,reduce_contacts,collect_physical,after
from native.terrain import bank_height_115,HEIGHT_115_GEOMETRIC_MIN as LIMIT
from probe_height_115_local_states import snapshot
from probe_height_115_margin import cases
from probe_height_115_passive import geometry
from probe_height_115_action_predict_loow import sha


@wp.kernel
def copy_diagnostic(slot:int,diag:wp.array2d[D],out:wp.array3d[D]):
    w=wp.tid()
    for j in range(diag.shape[1]):out[slot,w,j]=diag[w,j]


@wp.kernel
def radial(slot:int,q:wp.array2d[float],v:wp.array2d[float],ctrl:wp.array2d[float],ids:wp.array[int],
           active:wp.array[int],state:wp.array2d[D],command:wp.array[D],out:wp.array3d[D]):
    w=wp.tid();out[slot,w,0]=state[w,0]*D(.0005);out[slot,w,1]=D(active[w])
    if active[w]==0:return
    out[slot,w,2]=command[w];out[slot,w,3]=state[w,1]
    for s in range(2):
        a=D(q[w,ids[2*s]]);b=D(q[w,ids[2*s+1]])
        J=polar_jac(a,b);virtual=inverse2(J)*V2(D(ctrl[w,2*s]),D(ctrl[w,2*s+1]))
        out[slot,w,4+s]=fk(a,b)[3]
        out[slot,w,6+s]=J[0,0]*D(v[w,ids[4+2*s]])+J[1,0]*D(v[w,ids[5+2*s]])
        out[slot,w,8+s]=virtual[0];out[slot,w,10+s]=virtual[1]
    out[slot,w,12]=(D(ctrl[w,4])+D(ctrl[w,5]))/D(2)


def run(output,braking=False,current_vmc=False):
    assert not output.exists();scenes=cases()[:6]
    env=NativeEnv(n=6,scenario=scenes,bank_factory=bank_height_115,height_conditioned=True,height_design='range115',
                  residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=braking)
    try:
        if current_vmc:
            from native.design import current_vmc_table
            table,_=current_vmc_table();env.k['gains'].assign(np.stack([t[0] for t in table]));env.k['feed'].assign(np.stack([t[1] for t in table]));env.k['angles'].assign(np.array([t[2] for t in table]))
        env.reset();d=env.data;nq,nv=env.cpu.nq,env.cpu.nv;width=1+nq+2*nv+6
        pre=wp.zeros((40,6,width),dtype=D);post=wp.zeros_like(pre);metrics=wp.zeros((40,6,13),dtype=D)
        diagnostic=wp.zeros((40,6,env.diag.shape[1]),dtype=D)
        controller=wp.zeros((40,6,env.k['state'].shape[1]),dtype=D)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,6,[env.reward])
            for slot in range(40):
                wp.launch(command_step,6,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
                wp.launch(env.control_kernel,6,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
                    env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
                wp.launch(snapshot,6,[slot,d.time,d.qpos,d.qvel,d.qacc_warmstart,d.ctrl,pre])
                wp.launch(radial,6,[slot,d.qpos,d.qvel,d.ctrl,env.ids,env.active,env.state,env.command,metrics])
                if braking:
                    wp.launch(copy_diagnostic,6,[slot,env.diag,diagnostic])
                    wp.launch(copy_diagnostic,6,[slot,env.k['state'],controller])
                mjw.step(env.model,d)
                wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags])
                wp.launch(collect_physical,6,env.physical_args)
                wp.launch(after,6,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,
                    env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,
                    env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
                wp.launch(snapshot,6,[slot,d.time,d.qpos,d.qvel,d.qacc_warmstart,d.ctrl,post])
        chunks=[]
        for _ in range(700):
            wp.capture_launch(capture.graph);chunks.append((pre.numpy().copy(),post.numpy().copy(),metrics.numpy().copy(),diagnostic.numpy().copy(),controller.numpy().copy()))
            if np.all(env.done.numpy()!=0):break
        assert np.all(env.done.numpy()!=0)
        p=np.concatenate([c[0] for c in chunks]);z=np.concatenate([c[1] for c in chunks]);r=np.concatenate([c[2] for c in chunks])
        arrays={};events=[]
        for w in range(6):
            valid=r[:,w,1]>0;q=z[:,w,1:1+nq];g=geometry(env.cpu,q)
            mid=np.array([(env.cpu.body_pos[env.cpu.body('leg'+s).id]+env.cpu.body_pos[env.cpu.body('leg'+s+'_D').id])/2 for s in ('L','R')])
            actual=np.minimum(g[1].min(axis=1),np.linalg.norm(g[5]-mid,axis=-1).min(axis=1))
            if braking:
                arrival=float(r[valid,w,3].max());assert arrival>=0
                selected=valid&(r[:,w,0]>=arrival-.1)
                key=f'w{w}_braking';arrays[key+'_pre']=p[selected,w];arrays[key+'_post']=z[selected,w]
                arrays[key+'_radial']=r[selected,w];arrays[key+'_diagnostic']=np.concatenate([c[3] for c in chunks])[selected,w]
                arrays[key+'_controller_after_nominal']=np.concatenate([c[4] for c in chunks])[selected,w]
                assert len(arrays[key+'_pre'])>=4000 and np.all(actual[valid]>=LIMIT)
                events.append(dict(world=w,event='braking',arrival_s=arrival,physical_steps=int(valid.sum()),recorded_steps=int(selected.sum())))
                continue
            hits=np.flatnonzero(valid&(actual<LIMIT));assert len(hits)>0
            first=int(hits[0]);minimum=int(np.argmin(np.where(valid,actual,np.inf)))
            for label,index in (('first_geometry',first),('minimum_geometry',minimum)):
                lo=max(0,index-100);hi=min(len(p),index+101)
                key=f'w{w}_{label}';arrays[key+'_pre']=p[lo:hi,w];arrays[key+'_post']=z[lo:hi,w];arrays[key+'_radial']=r[lo:hi,w]
                events.append(dict(world=w,event=label,step=index,time_s=float(r[index,w,0]+.0005),
                    stopped=bool(r[index,w,3]>=0),arrival_s=float(r[index,w,3]),command_m_s=float(r[index,w,2]),
                    actual_min_leg_m=float(actual[index]),pre_fk_length_m=r[index,w,4:6].tolist(),
                    pre_fk_rate_m_s=r[index,w,6:8].tolist(),nominal_radial_force_N=r[index,w,8:10].tolist(),
                    nominal_hub_Nm=r[index,w,10:12].tolist(),mean_wheel_command_Nm=float(r[index,w,12])))
        _,_,_,infos=env.step_wait()
        output.mkdir(parents=True);np.savez_compressed(output/'windows.npz',**arrays)
        if braking:np.savez_compressed(output/'controller_inputs.npz',**{k:env.k[k].numpy() for k in ('heights','gains','feed','angles','reference','yaw')},ids=env.ids.numpy(),public_gain_upper=env.actuator_gain_upper)
        result=dict(role='radial_guard_braking_windows' if braking else 'coordinated_reference_radial_failure_windows',events=events,scenarios=[asdict(s) for s in scenes],
            current_vmc_design=current_vmc,
            complete_episodes=[{k:v for k,v in row.items() if k!='terminal_observation'} for row in infos],
            columns=dict(time=0,qpos_start=1,qvel_start=1+nq,warmstart_start=1+nq+nv,ctrl_start=1+nq+2*nv),
            radial_columns=['time','valid','command','arrival','fk_L','fk_R','rate_L','rate_R','force_L','force_R','hub_L','hub_R','wheel_mean'],
            windows_sha256=sha(output/'windows.npz'),source_sha256={str(f.relative_to(ROOT)):sha(f) for f in
                (Path(__file__),ROOT/'wheelleg_warp/probe_current_vmc_design.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
        (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('COMPLETED',events,flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--braking',action='store_true');parser.add_argument('--current-vmc',action='store_true');args=parser.parse_args()
    run(args.output,args.braking,args.current_vmc)
