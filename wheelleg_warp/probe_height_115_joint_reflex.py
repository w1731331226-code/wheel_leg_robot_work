"""0.115 m单项关节角预测制动：零、径向、径向＋关节同图对照。"""
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
from native.environment import NativeEnv,begin,command_step,reduce_contacts,after
from native.terrain import HEIGHT_115_GEOMETRIC_MIN,bank_height_115
from probe_height_115_margin import cases
from probe_height_115_safety_reflex import radial_guard,audit


@wp.kernel
def joint_guard(q:wp.array2d[float],v:wp.array2d[float],ctrl:wp.array2d[float],
                ids:wp.array[int],active:wp.array[int],state:wp.array2d[D],
                mode:wp.array[int],stats:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0 or mode[w]!=2:return
    triggered=False
    for j in range(4):
        angle=D(q[w,ids[j]]);speed=D(v[w,ids[4+j]])
        sign=wp.sign(angle);outward=wp.max(D(0),sign*speed)
        predicted=wp.abs(angle)+D(.02)*outward
        if predicted>D(1.45):
            request=wp.clamp(D(100)*(predicted-D(1.45))+D(2)*outward,D(0),D(20))
            before=D(ctrl[w,j]);unclipped=before-sign*request
            bound=allowed(speed,True);applied=wp.clamp(unclipped,-bound,bound)
            ctrl[w,j]=float(applied)
            stats[w,1]=wp.max(stats[w,1],request)
            stats[w,2]=wp.max(stats[w,2],wp.abs(applied-before))
            if wp.abs(applied-unclipped)>D(1.e-8):stats[w,3]=stats[w,3]+D(1)
            triggered=True
    if triggered:
        stats[w,0]=stats[w,0]+D(1)
        if stats[w,4]<D(0):stats[w,4]=state[w,0]*D(.0005)


def run(output):
    assert not output.exists()
    base=cases();assert len(base)==14
    scenarios=base*3;n=len(scenarios)
    env=NativeEnv(n=n,scenario=scenarios,bank_factory=bank_height_115,
                  height_conditioned=True,height_design='range115',residual_scale=0)
    env.reset()
    mode=wp.array(np.repeat(np.arange(3,dtype=np.int32),len(base)),dtype=wp.int32)
    initial=np.zeros((n,11));initial[:,4]=1.;initial[:,7]=-1.;initial[:,8]=1.
    radial_stats=wp.array(initial,dtype=D)
    joint_initial=np.zeros((n,5));joint_initial[:,4]=-1.
    joint_stats=wp.array(joint_initial,dtype=D)
    passive=wp.array([env.cpu.joint(name).qposadr[0] for name in
        ('passA_L','passC_L','passA_R','passC_R')],dtype=wp.int32)
    b_offsets=wp.array(np.tile(np.asarray([[
        env.cpu.body('leg'+side+'_D').pos,
        env.cpu.body('kneeB_'+side).pos,
        env.cpu.site('couplerB_'+side+'_end').pos] for side in ('L','R')]),(n,1,1,1)),dtype=wp.vec3d)
    data=env.data
    with wp.ScopedCapture() as capture:
        wp.launch(begin,n,[env.reward])
        for _ in range(40):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,
                data.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,
                env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],
                env.k['reference'],env.k['yaw'],data.ctrl,env.diag,0,0],block_dim=32)
            wp.launch(radial_guard,n,[data.qpos,data.qvel,data.ctrl,env.ids,env.active,env.state,mode,radial_stats])
            wp.launch(joint_guard,n,[data.qpos,data.qvel,data.ctrl,env.ids,env.active,env.state,mode,joint_stats])
            mjw.step(env.model,data)
            wp.launch(audit,n,[data.qpos,data.qvel,data.ctrl,env.ids,passive,env.active,
                env.wheel_offsets,b_offsets,radial_stats])
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,data.contact.worldid,data.contact.geom,
                env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,
                env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,
                env.residual,env.active,env.done,env.reward,env.obs,env.history,
                env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    env.graph=capture.graph
    results=[None]*n;zero=np.zeros((n,3),np.float32)
    names=('baseline','radial_guard','radial_plus_joint')
    for _ in range(700):
        env.step_async(zero)
        done_before=env.done.numpy();radial_before=radial_stats.numpy();joint_before=joint_stats.numpy()
        _,_,_,infos=env.step_wait()
        for i in np.flatnonzero(done_before):
            if results[i] is not None:continue
            info=infos[i];r=radial_before[i];j=joint_before[i]
            results[i]=dict(scenario=asdict(scenarios[i]),mode=names[i//len(base)],
                success=bool(info['success']),reason=info['reason'],peak_deg=info['peak_deg'],
                height_rmse_m=info['height_rmse_m'],min_fk_leg_m=info['min_leg_m'],
                min_actual_chain_leg_m=float(r[8]),min_joint_margin_rad=float(r[4]),
                max_chain_vs_fk_m=float(r[9]),max_closed_chain_error_m=float(r[10]),
                peak_hip_rpm=float(r[5]),peak_hip_torque_Nm=float(r[6]),
                radial_trigger_physical_steps=int(r[0]),radial_peak_force_N=float(r[1]),
                radial_peak_executed_delta_Nm=float(r[2]),
                radial_clipped_hip_commands=int(r[3]),
                joint_trigger_physical_steps=int(j[0]),joint_peak_requested_Nm=float(j[1]),
                joint_peak_executed_delta_Nm=float(j[2]),joint_clipped_hip_commands=int(j[3]),
                first_joint_trigger_s=float(j[4]) if j[4]>=0 else None,
                full_gate=bool(info['success'] and r[8]>=HEIGHT_115_GEOMETRIC_MIN and r[4]>=0))
        if all(row is not None for row in results):break
    assert all(row is not None for row in results)
    assert all(row['radial_trigger_physical_steps']==0 and row['joint_trigger_physical_steps']==0
               for row in results[:len(base)])
    assert all(row['joint_trigger_physical_steps']==0 for row in results[len(base):2*len(base)])
    summary={}
    for arm,name in enumerate(names):
        rows=results[arm*len(base):(arm+1)*len(base)]
        summary[name]=dict(total=len(rows),task_success=sum(row['success'] for row in rows),
                           full_gate=sum(row['full_gate'] for row in rows),
                           normal_full_gate=sum(row['full_gate'] for row in rows[:6]),
                           worst_actual_chain_leg_m=min(row['min_actual_chain_leg_m'] for row in rows),
                           worst_joint_margin_rad=min(row['min_joint_margin_rad'] for row in rows),
                           max_closed_chain_error_m=max(row['max_closed_chain_error_m'] for row in rows))
    sources=('wheelleg_warp/probe_height_115_joint_reflex.py','wheelleg_warp/probe_height_115_safety_reflex.py',
             'wheelleg_warp/probe_height_115_margin.py','wheelleg_warp/native/controller.py',
             'wheelleg_warp/native/environment.py','wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py',
             'wheelleg_ppo/tools/rm_controller.py','wheelleg_ppo/tools/model_lqr.py',
             'wheelleg_ppo/tools/wheelleg_sim.py','wheelleg_ppo/tools/hardware_profile.py',
             'wheelleg_ppo/tools/state_estimation.py',
             'wheelleg_ppo/xml/wheelleg.xml')
    output.mkdir(parents=True)
    (output/'verification.json').write_text(json.dumps(dict(protocol='14 public cases x baseline/radial/combined fixed joint guard',
        nominal_target_m=.115,geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,training=False,final_holdout_opened=False,
        summary=summary,rows=results,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}),
        ensure_ascii=False,indent=2)+'\n')
    print(summary)
    if summary['radial_plus_joint']['normal_full_gate']<6:
        raise AssertionError('关节角预测制动未过六个正常场景联合门，停止该固定候选')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
