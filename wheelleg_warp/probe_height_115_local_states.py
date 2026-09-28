"""为局部可行性检查截取首次腿长/关节越界前后的2 kHz物理状态。"""
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
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5
from native.controller import D,control
from native.environment import NativeEnv,begin,command_step,reduce_contacts,after
from native.terrain import bank_height_115
from probe_height_115_margin import cases


@wp.kernel
def snapshot(slot:int,clock:wp.array[float],q:wp.array2d[float],v:wp.array2d[float],
             warm:wp.array2d[float],ctrl:wp.array2d[float],out:wp.array3d[D]):
    w=wp.tid();nq=q.shape[1];nv=v.shape[1];nu=ctrl.shape[1]
    out[slot,w,0]=D(clock[w])
    for j in range(nq):out[slot,w,1+j]=D(q[w,j])
    for j in range(nv):
        out[slot,w,1+nq+j]=D(v[w,j])
        out[slot,w,1+nq+nv+j]=D(warm[w,j])
    for j in range(nu):out[slot,w,1+nq+2*nv+j]=D(ctrl[w,j])


@wp.kernel
def contact_snapshot(slot:int,ncon:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],
                     pos:wp.array[wp.vec3],dist:wp.array[float],frame:wp.array[wp.mat33],
                     friction:wp.array[vec5],dim:wp.array[int],
                     address:wp.array2d[int],force:wp.array2d[float],njmax:int,cone:int,
                     out:wp.array3d[D]):
    j=wp.tid()
    for k in range(25):out[slot,j,k]=D(0)
    out[slot,j,0]=D(-1);out[slot,j,1]=D(-1);out[slot,j,2]=D(-1)
    if j>=ncon[0]:return
    w=world[j]
    if w<0:return
    f=contact_force_fn(cone,frame,friction,dim,address,force,njmax,ncon,w,j,False)
    out[slot,j,0]=D(w);out[slot,j,1]=D(geom[j][0]);out[slot,j,2]=D(geom[j][1])
    out[slot,j,3]=D(dist[j])
    for k in range(3):
        out[slot,j,4+k]=D(pos[j][k])
        out[slot,j,16+k]=D(f[k])
    for row in range(3):
        for col in range(3):out[slot,j,7+3*row+col]=D(frame[j][row,col])
    for k in range(5):out[slot,j,19+k]=D(friction[j][k])
    out[slot,j,24]=D(dim[j])


def event_list(rows):
    events=[]
    for w,row in enumerate(rows):
        actual=[row[side]['first_actual_below_proxy_s'] for side in ('L','R')]
        actual=[value for value in actual if value is not None]
        assert actual and row['first_active_joint_limit_s'] is not None
        for kind,t in (('actual_leg',min(actual)),('active_joint',row['first_active_joint_limit_s'])):
            events.append(dict(world=w,kind=kind,reference_s=float(t),scenario=row['scenario']))
    return events


def run(output):
    assert not output.exists()
    reference=ROOT/'wheelleg_warp/results/height_115_passive_20260928/verification.json'
    previous=json.loads(reference.read_text())
    scenarios=cases();assert [asdict(s) for s in scenarios]==[r['scenario'] for r in previous['rows']]
    events=event_list(previous['rows']);assert len(events)==28
    n=len(scenarios)
    env=NativeEnv(n=n,scenario=scenarios,bank_factory=bank_height_115,
                  height_conditioned=True,height_design='range115',residual_scale=0)
    env.reset();data=env.data;c=data.contact
    nq,nv,nu=env.cpu.nq,env.cpu.nv,env.cpu.nu
    width=1+nq+2*nv+nu
    pre=wp.zeros((40,n,width),dtype=D);post=wp.zeros((40,n,width),dtype=D)
    contacts=wp.zeros((40,data.naconmax,25),dtype=D)
    with wp.ScopedCapture() as captured:
        wp.launch(begin,n,[env.reward])
        for slot in range(40):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,
                data.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,
                env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],
                env.k['reference'],env.k['yaw'],data.ctrl,env.diag,0,0],block_dim=32)
            wp.launch(snapshot,n,[slot,data.time,data.qpos,data.qvel,data.qacc_warmstart,data.ctrl,pre])
            mjw.step(env.model,data)
            wp.launch(snapshot,n,[slot,data.time,data.qpos,data.qvel,data.qacc_warmstart,data.ctrl,post])
            wp.launch(contact_snapshot,data.naconmax,[slot,data.nacon,c.worldid,c.geom,c.pos,c.dist,c.frame,
                c.friction,c.dim,c.efc_address,data.efc.force,data.njmax,env.model.opt.cone,contacts])
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,c.worldid,c.geom,env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,
                env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,
                env.residual,env.active,env.done,env.reward,env.obs,env.history,
                env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    env.targets.assign(np.zeros((n,3),np.float32))
    records=[dict(pre=[],post=[],contacts=[]) for _ in events]
    for step in range(700):
        wp.capture_launch(captured.graph)
        mid=(step+.5)*.02
        if any(abs(mid-event['reference_s'])<.07 for event in events):
            a=pre.numpy();b=post.numpy();pairs=contacts.numpy()
            for event,record in zip(events,records):
                if abs(mid-event['reference_s'])>=.07:continue
                w=event['world'];t=event['reference_s']
                selected=np.flatnonzero((a[:,w,0]>=t-.05)&(a[:,w,0]<=t+.05))
                if len(selected):
                    record['pre'].append(a[selected,w].copy())
                    record['post'].append(b[selected,w].copy())
                    record['contacts'].append(pairs[selected].copy())
        if np.all(env.done.numpy()!=0):break
    assert np.all(env.done.numpy()!=0)
    arrays={}
    for i,(event,record) in enumerate(zip(events,records)):
        assert all(record.values()),event
        for kind in ('pre','post','contacts'):
            arrays[f'event_{i}_{kind}']=np.concatenate(record[kind])
        times=arrays[f'event_{i}_pre'][:,0]
        after_times=arrays[f'event_{i}_post'][:,0]
        assert len(times)>=150 and np.all(np.diff(times)>0) and np.max(abs(after_times-times-.0005))<2e-6,event
        event['samples']=len(times)
        event['recorded_interval_s']=[float(times[0]),float(after_times[-1])]
    output.mkdir(parents=True)
    np.savez_compressed(output/'windows.npz',**arrays)
    sources=('wheelleg_warp/probe_height_115_local_states.py','wheelleg_warp/native/controller.py',
             'wheelleg_warp/native/environment.py','wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py',
             'wheelleg_warp/probe_height_115_passive.py','wheelleg_warp/probe_height_115_margin.py',
             'wheelleg_ppo/tools/rm_controller.py','wheelleg_ppo/tools/model_lqr.py',
             'wheelleg_ppo/tools/wheelleg_sim.py','wheelleg_ppo/tools/hardware_profile.py',
             'wheelleg_ppo/xml/wheelleg.xml')
    payload=dict(role='public_local_feasibility_state_windows_not_control_result',training=False,final_holdout_opened=False,
                 worlds=n,physical_step_s=.0005,event_count=len(events),nq=nq,nv=nv,nu=nu,
                 columns=dict(time=0,qpos_start=1,qvel_start=1+nq,warmstart_start=1+nq+nv,ctrl_start=1+nq+2*nv),
                 contact_columns=('world','geom1','geom2','dist_m','pos_x_m','pos_y_m','pos_z_m',
                                  'frame_00','frame_01','frame_02','frame_10','frame_11','frame_12',
                                  'frame_20','frame_21','frame_22','normal_force_N','tangent1_N','tangent2_N',
                                  'mu_0','mu_1','mu_2','mu_3','mu_4','dim'),events=events,
                 passive_verification_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                 source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                 windows_sha256=hashlib.sha256((output/'windows.npz').read_bytes()).hexdigest())
    (output/'verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print('PASS',n,len(events),[e['samples'] for e in events][:4],output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
