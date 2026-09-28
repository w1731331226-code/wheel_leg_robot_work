"""0.115 m单项2 kHz低位径向安全反射：公开面板同图零动作配对。"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "wheelleg_warp"), str(ROOT / "wheelleg_ppo/tools")]
import numpy as np
import mujoco_warp as mjw
import warp as wp
from native.controller import D, V2, allowed, control, fk, polar_jac
from native.environment import NativeEnv, after, begin, command_step, reduce_contacts, rotate_y, wheel_center
from native.terrain import HEIGHT_115_GEOMETRIC_MIN, bank_height_115
from probe_height_115_margin import cases


@wp.kernel
def radial_guard(q:wp.array2d[float],v:wp.array2d[float],ctrl:wp.array2d[float],
                 ids:wp.array[int],active:wp.array[int],state:wp.array2d[D],
                 mode:wp.array[int],stats:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0 or mode[w]==0:return
    triggered=False
    for side in range(2):
        a=D(q[w,ids[2*side]]);b=D(q[w,ids[2*side+1]])
        va=D(v[w,ids[4+2*side]]);vb=D(v[w,ids[5+2*side]])
        jac=polar_jac(a,b);length=fk(a,b)[3]
        rate=jac[0,0]*va+jac[1,0]*vb
        closing=wp.max(D(0),-rate)
        predicted=length-D(.02)*closing
        if predicted<D(HEIGHT_115_GEOMETRIC_MIN):
            force=wp.clamp(D(500)*(D(HEIGHT_115_GEOMETRIC_MIN)-predicted)+D(500)*closing,D(0),D(40))
            torque=jac*V2(force,D(0))
            stats[w,1]=wp.max(stats[w,1],force)
            for j in range(2):
                motor=2*side+j
                speed=va
                if j==1:speed=vb
                bound=allowed(speed,True)
                request=D(ctrl[w,motor])+torque[j]
                applied=wp.clamp(request,-bound,bound)
                stats[w,2]=wp.max(stats[w,2],wp.abs(applied-D(ctrl[w,motor])))
                if wp.abs(request-applied)>D(1.e-8):stats[w,3]=stats[w,3]+D(1)
                ctrl[w,motor]=float(applied)
            triggered=True
    if triggered:
        stats[w,0]=stats[w,0]+D(1)
        if stats[w,7]<D(0):stats[w,7]=state[w,0]*D(.0005)


@wp.kernel
def audit(q:wp.array2d[float],v:wp.array2d[float],ctrl:wp.array2d[float],
          ids:wp.array[int],passive:wp.array[int],active:wp.array[int],
          offsets:wp.array3d[wp.vec3d],b_offsets:wp.array3d[wp.vec3d],stats:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    margin=D(1)
    for j in range(4):
        margin=wp.min(margin,D(1.5)-wp.abs(D(q[w,ids[j]])))
        margin=wp.min(margin,D(2.5)-wp.abs(D(q[w,passive[j]])))
        rpm=wp.abs(D(v[w,ids[4+j]]))*D(60)/(D(2)*D(3.141592653589793))
        stats[w,5]=wp.max(stats[w,5],rpm)
        stats[w,6]=wp.max(stats[w,6],wp.abs(D(ctrl[w,j])))
    stats[w,4]=wp.min(stats[w,4],margin)
    rotation=wp.quatd(D(q[w,4]),D(q[w,5]),D(q[w,6]),D(q[w,3]))
    root=wp.vec3d(D(q[w,0]),D(q[w,1]),D(q[w,2]))
    for side in range(2):
        hip=root+wp.quat_rotate(rotation,wp.vec3d(D(.075),offsets[w,side,0][1],D(0)))
        wheel=wheel_center(q,ids,offsets,w,side)
        actual=wp.length(wheel-hip)
        virtual=fk(D(q[w,ids[2*side]]),D(q[w,ids[2*side+1]]))[3]
        stats[w,8]=wp.min(stats[w,8],actual)
        stats[w,9]=wp.max(stats[w,9],wp.abs(actual-virtual))
        alpha=D(q[w,ids[2*side]]);beta=D(q[w,ids[2*side+1]])
        pass_a=D(q[w,passive[2*side]]);pass_c=D(q[w,passive[2*side+1]])
        a_chain=offsets[w,side,0]+rotate_y(offsets[w,side,1],alpha)+rotate_y(offsets[w,side,2],alpha+pass_a)
        b_chain=b_offsets[w,side,0]+rotate_y(b_offsets[w,side,1],beta)+rotate_y(b_offsets[w,side,2],beta+pass_c)
        stats[w,10]=wp.max(stats[w,10],wp.length(a_chain-b_chain))


def run(output):
    assert not output.exists()
    base=cases()
    assert len(base)==14
    scenarios=base+base
    n=len(scenarios)
    env=NativeEnv(n=n,scenario=scenarios,bank_factory=bank_height_115,
                  height_conditioned=True,height_design="range115",residual_scale=0)
    env.reset()
    mode=wp.array(np.array([0]*len(base)+[1]*len(base),np.int32))
    initial=np.zeros((n,11));initial[:,4]=1.;initial[:,7]=-1.;initial[:,8]=1.
    stats=wp.array(initial,dtype=D)
    passive_names=("passA_L","passC_L","passA_R","passC_R")
    passive=wp.array([env.cpu.joint(name).qposadr[0] for name in passive_names],dtype=wp.int32)
    b_offsets=wp.array(np.tile(np.asarray([[
        env.cpu.body("leg"+side+"_D").pos,
        env.cpu.body("kneeB_"+side).pos,
        env.cpu.site("couplerB_"+side+"_end").pos] for side in ("L","R")]),(n,1,1,1)),dtype=wp.vec3d)
    data=env.data
    with wp.ScopedCapture() as capture:
        wp.launch(begin,n,[env.reward])
        for _ in range(40):
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,
                data.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,
                env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],
                env.k['reference'],env.k['yaw'],data.ctrl,env.diag,0,0],block_dim=32)
            wp.launch(radial_guard,n,[data.qpos,data.qvel,data.ctrl,env.ids,env.active,env.state,mode,stats])
            mjw.step(env.model,data)
            wp.launch(audit,n,[data.qpos,data.qvel,data.ctrl,env.ids,passive,env.active,env.wheel_offsets,b_offsets,stats])
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,data.contact.worldid,data.contact.geom,
                env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,
                env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,
                env.residual,env.active,env.done,env.reward,env.obs,env.history,
                env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    env.graph=capture.graph
    zero=np.zeros((n,3),np.float32)
    results=[None]*n
    for _ in range(700):
        env.step_async(zero)
        done_before=env.done.numpy();stats_before=stats.numpy()
        _,_,_,infos=env.step_wait()
        for i in np.flatnonzero(done_before):
            if results[i] is None:
                info=infos[i];s=stats_before[i]
                results[i]=dict(scenario=asdict(scenarios[i]),mode="baseline" if i<len(base) else "radial_guard",
                    success=bool(info["success"]),reason=info["reason"],peak_deg=info["peak_deg"],
                    height_rmse_m=info["height_rmse_m"],min_fk_leg_m=info["min_leg_m"],
                    min_actual_chain_leg_m=float(s[8]),max_chain_vs_fk_m=float(s[9]),
                    max_closed_chain_error_m=float(s[10]),
                    min_joint_margin_rad=float(s[4]),peak_hip_rpm=float(s[5]),peak_hip_torque_Nm=float(s[6]),
                    trigger_physical_steps=int(s[0]),peak_requested_force_N=float(s[1]),
                    peak_executed_extra_hip_Nm=float(s[2]),clipped_hip_physical_commands=int(s[3]),
                    first_trigger_s=float(s[7]) if s[7]>=0 else None,
                    full_gate=bool(info["success"] and s[8]>=HEIGHT_115_GEOMETRIC_MIN and s[4]>=0))
        if all(row is not None for row in results):break
    assert all(row is not None for row in results)
    assert all(row["trigger_physical_steps"]==0 for row in results[:len(base)])
    summary={}
    for name,rows in (("baseline",results[:len(base)]),("radial_guard",results[len(base):])):
        summary[name]=dict(total=len(rows),task_success=sum(row["success"] for row in rows),
                           full_gate=sum(row["full_gate"] for row in rows),
                           normal_full_gate=sum(row["full_gate"] for row in rows[:6]),
                           worst_fk_leg_m=min(row["min_fk_leg_m"] for row in rows),
                           worst_actual_chain_leg_m=min(row["min_actual_chain_leg_m"] for row in rows),
                           worst_joint_margin_rad=min(row["min_joint_margin_rad"] for row in rows),
                           max_closed_chain_error_m=max(row["max_closed_chain_error_m"] for row in rows))
    names=("wheelleg_warp/probe_height_115_safety_reflex.py","wheelleg_warp/probe_height_115_margin.py",
           "wheelleg_warp/native/controller.py","wheelleg_warp/native/environment.py",
           "wheelleg_warp/native/terrain.py","wheelleg_warp/native/models.py",
           "wheelleg_ppo/tools/rm_controller.py","wheelleg_ppo/tools/model_lqr.py",
           "wheelleg_ppo/tools/wheelleg_sim.py","wheelleg_ppo/tools/hardware_profile.py",
           "wheelleg_ppo/tools/state_estimation.py",
           "wheelleg_ppo/xml/wheelleg.xml")
    output.mkdir(parents=True)
    (output/"verification.json").write_text(json.dumps(dict(protocol="0.115m 14 public cases paired zero vs one fixed predictive radial guard",
        target_leg_m=.115,geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,training=False,final_holdout_opened=False,
        summary=summary,rows=results,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}),
        ensure_ascii=False,indent=2)+"\n")
    print(summary)
    if summary["radial_guard"]["normal_full_gate"]<6:
        raise AssertionError("低位安全反射未过六个正常场景门，停止该固定候选")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
