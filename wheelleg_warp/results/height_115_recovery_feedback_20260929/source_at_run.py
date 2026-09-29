"""恢复阶段当前状态预警/LP反馈；host-in-loop诊断，不作2kHz部署声明。"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import warp as wp
import mujoco_warp as mjw
from probe_height_115_action_predict_loow import ROOT, DT, ACTIVE, forecast, sha, START_LIMITS
from probe_height_115_continuous_common_1nm import GAMMA, LENGTH_SCALE_M, JOINT_SCALE_RAD
from probe_height_115_live_braking import solve_current
from probe_height_115_live_common_local import align_start, common_basis
from probe_height_115_radial_authority import torque_box, pose
from probe_height_115_contact_action_pair import margins, contact_sets
from probe_height_115_passive import geometry
from native.controller import control
from native.environment import NativeEnv, begin, command_step, reduce_contacts, after
from native.terrain import HeightTerrainScenario, bank_height_115, HEIGHT_115_GEOMETRIC_MIN


def run():
    folder=ROOT/'wheelleg_warp/results';output=folder/'height_115_recovery_feedback_20260929';assert not output.exists()
    paths=[folder/n/'verification.json' for n in ('height_115_contact_response_2nm_20260929',
        'height_115_action_predict_1nm_single_graph_20260929','height_115_local_states_20260928',
        'height_115_contact_hold_v2_20260929')]
    response,fit,archive,prior=[json.loads(p.read_text()) for p in paths]
    prior_path=paths[3].parent/'trace.npz';assert sha(prior_path)==prior['trace_sha256']
    with np.load(prior_path) as z:old={k:z[k] for k in z.files}
    scenario=HeightTerrainScenario(**archive['events'][fit['selected_event_ids'][3]]['scenario'])
    env=NativeEnv(n=7,scenario=[scenario]*7,bank_factory=bank_height_115,height_conditioned=True,height_design='range115',residual_scale=0)
    env.reset();d=env.data;m=env.cpu;n=7
    qa=np.array([m.joint(x).qposadr[0] for x in ACTIVE]);va=np.array([m.joint(x).dofadr[0] for x in ACTIVE])
    limits=np.array([m.jnt_range[m.joint(x).id] for x in ACTIVE]);gain=np.array(fit['folds'][3]['gain'])
    rs=fit['folds'][3]['reserves'];reserve=(rs['actual_A_length_m'],rs['eight_joint_margin_rad'])
    start=response['rows'][2]['start_step'];assert response['rows'][2]['world']==3
    with wp.ScopedCapture() as prepare:
        wp.launch(begin,n,[env.reward])
        wp.launch(command_step,n,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,
            env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
        wp.launch(control,n,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
            env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0],block_dim=32)
    with wp.ScopedCapture() as advance:
        mjw.step(env.model,d)
        wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags])
        wp.launch(after,n,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,
            env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,
            env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    for _ in range(start//40):wp.capture_launch(env.graph)
    for _ in range(start%40):wp.capture_launch(prepare.graph);wp.capture_launch(advance.graph)
    assert np.all(env.state.numpy()[:,0]==start)
    wp.launch(align_start,n,[wp.array(np.full(n,start,np.int32)),d.qpos,d.qvel,d.qacc_warmstart,d.sensordata,
        env.k['state'],env.state,env.command,env.contact_flags,env.stopped_q,env.stopped_v,env.stopped_w,env.active])
    output.mkdir();rows=[];traces=[];timings=[];prev=None;last=np.zeros((7,3));failure=None
    status='completed_1020ms';first_alarm=None;prefix_errors=dict(qpos=0.,qvel=0.,ctrl=0.)
    wheels={m.geom('wheel_collide_'+s).id for s in ('L','R')}
    for t in range(2040):
        if t==40:
            # Diagnostic paired fork: copy physical/controller memory, not future trajectory.
            for arr in (d.qpos,d.qvel,d.qacc_warmstart,d.sensordata,env.k['state'],env.state,env.command,
                        env.contact_flags,env.stopped_q,env.stopped_v,env.stopped_w,env.active):
                a=arr.numpy();a[2]=a[1];arr.assign(a)
            prev[2]=prev[1];last[2]=last[1]
        wp.capture_launch(prepare.graph)
        q=d.qpos.numpy().astype(float);v=d.qvel.numpy().astype(float);nom=d.ctrl.numpy().astype(float)
        if t==40:
            assert np.array_equal(q[1],q[2]) and np.array_equal(v[1],v[2]) and np.array_equal(nom[1],nom[2])
        if t<40:
            for key,err in (('qpos',abs(q-old['pre'][:,t,:m.nq]).max()),
                            ('qvel',abs(v-old['pre'][:,t,m.nq:m.nq+m.nv]).max()),
                            ('ctrl',abs(nom-old['pre'][:,t,m.nq+m.nv:m.nq+m.nv+6]).max())):
                prefix_errors[key]=max(prefix_errors[key],float(err))
            if any(prefix_errors[k]>START_LIMITS[k] for k in prefix_errors):
                status='failed_initial_prefix_pairing';failure=dict(step=t,errors=prefix_errors);break
        actions=old['schedule'][t].copy() if t<40 else np.zeros((7,3));alarm=False;predicted=None
        if t>=41:
            tic=perf_counter()
            corrected_previous=prev[1]+DT*(gain@last[1])
            pl,pj=forecast(dict(q0=q[1,qa],v0=v[1,va],vprev=corrected_previous,
                coeff=np.zeros((1,3)),active_joint_limits=limits),np.zeros((4,3)))
            predicted=[float(pl.min()-HEIGHT_115_GEOMETRIC_MIN-reserve[0]-GAMMA*LENGTH_SCALE_M),
                       float(pj.min()-reserve[1]-GAMMA*JOINT_SCALE_RAD)]
            alarm=min(predicted)<0
            if alarm:
                if first_alarm is None:first_alarm=t
                c,bs,a0,error=solve_current(m,q[1],v[1],prev[1],last[1],gain,reserve,nom[1])
                if error:
                    failure=dict(step=t,time_ms=t*.5,reason=error,barriers=[dict(name=b['name'],h=b['h'],rate=b['rate']) for b in bs],
                        true_A_margin_m=(geometry(m,q[1])[1]-HEIGHT_115_GEOMETRIC_MIN).tolist())
                    status='failed_model_feasibility';timings.append((perf_counter()-tic)*1000);break
                actions[1]=c
            timings.append((perf_counter()-tic)*1000)
        issued=nom.copy()
        for arm in range(7):issued[arm]+=common_basis(m,q[arm],v[arm])@actions[arm]
        motor_ok=all(np.max(abs(issued[a])-torque_box(m,v[a]))<=1e-6 for a in range(7))
        if not motor_ok:status='failed_motor_box';failure=dict(step=t);break
        d.ctrl.assign(issued.astype(np.float32));wp.capture_launch(advance.graph)
        qn=d.qpos.numpy().astype(float);vn=d.qvel.numpy().astype(float);pairs=contact_sets(d,7);checks=[]
        for arm in (1,2):
            L=geometry(m,qn[arm])[1];J=float(margins(m,qn[arm]).min());att=max(pose(qn[arm],scenario)['world_abs_deg'])
            nonwheel=any(not wheels.intersection(p) for p in pairs[arm]);peak=float(abs(issued[arm]-nom[arm]).max())
            safe=bool(L.min()>=HEIGHT_115_GEOMETRIC_MIN and J>=0 and att<=5 and not nonwheel and env.active.numpy()[arm])
            checks.append(dict(arm=arm,safe=safe,true_A_margin_m=(L-HEIGHT_115_GEOMETRIC_MIN).tolist(),
                joint_margin_rad=J,attitude_deg=att,nonwheel=nonwheel,peak_extra_motor_Nm=peak))
        rows.append(dict(step=t,time_ms=(t+1)*.5,alarm=alarm,predicted_margins=predicted,action=actions[1].tolist(),checks=checks))
        traces.append((q,v,nom,actions,issued.astype(np.float32),qn,vn))
        prev=v[:,va].copy();last=actions.copy()
        if not checks[0]['safe']:status='failed_feedback_physical_gate';failure=rows[-1];break
    if traces:
        np.savez_compressed(output/'trace.npz',**{key:np.stack([r[i] for r in traces]) for i,key in enumerate(
            ('pre_q','pre_v','nominal','actions','issued','post_q','post_v'))})
    result=dict(role='current_state_recovery_feedback_host_in_loop',status=status,failure=failure,
        completed_steps=len(rows),first_alarm_ms=None if first_alarm is None else first_alarm*.5,
        recovery_nonzero_action_steps=sum(np.any(r['action']) for r in rows[40:]),
        initial_prefix_errors=prefix_errors,paired_recovery_fork_step=40,
        host_compute_median_ms=float(np.median(timings)) if timings else None,
        host_compute_max_ms=max(timings) if timings else None,host_compute_excludes_gpu_transfer=True,
        rows=rows,limitations='Initial20ms action and onset remain offline oracle. Recovery uses current active q/dq, previous dq and previous issued action, with frozen unvalidated-error G. Ideal simulation observations; no real-time, hardware or full-episode claim.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[prior_path]},
        trace_sha256=sha(output/'trace.npz') if traces else None,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_live_braking.py',
            ROOT/'wheelleg_warp/probe_height_115_action_predict_loow.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False)+'\n')
    print({k:v for k,v in result.items() if k not in ('rows','input_sha256','source_sha256')},flush=True)


if __name__=='__main__':run()
