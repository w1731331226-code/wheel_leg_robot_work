"""当前仿真状态求解：反向平地零/一次制动/滚动制动的10ms诊断。"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import numpy as np
import mujoco_warp as mjw
import warp as wp
from scipy.optimize import linprog
from native.controller import control
from native.environment import NativeEnv, begin, command_step, reduce_contacts, after
from native.terrain import HeightTerrainScenario, bank_height_115, HEIGHT_115_GEOMETRIC_MIN
from probe_height_115_braking_budget import barriers, DT, ACTIVE
from probe_height_115_contact_action_pair import basis, margins, contact_sets
from probe_height_115_live_common_local import align_start
from probe_height_115_radial_authority import torque_box, pose
from probe_height_115_passive import geometry
from probe_height_115_action_predict_loow import sha


def solve_current(m, q, v, previous_v, previous_action, gain, reserve, nominal):
    va = np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    # Measured acceleration already contains the previously issued extra action.
    a0 = (v[va] - previous_v) / DT - gain @ previous_action
    return solve_from_acceleration(m, q, v, a0, gain, reserve, nominal)


def solve_from_acceleration(m, q, v, a0, gain, reserve, nominal):
    qa = np.array([m.joint(n).qposadr[0] for n in ACTIVE])
    va = np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    bs, _ = barriers(q[qa], v[va], a0, gain, reserve)
    if any(b['h'] <= 0 for b in bs):
        return None, bs, a0, 'tightened_position_already_outside'
    B = basis(m, q, v); bound = torque_box(m, v)
    A, rhs = [], []
    for b in bs:
        if b['rate'] < 0:
            required = b['rate']**2 / (2*b['h'])
            A.append(np.r_[-b['g'], 0.]); rhs.append(b['a0'] - required)
    for j, row in enumerate(B):
        A.extend((np.r_[row, -1.], np.r_[-row, -1.], np.r_[row, 0.], np.r_[-row, 0.]))
        rhs.extend((0., 0., bound[j]-nominal[j], bound[j]+nominal[j]))
    result = linprog([0.,0.,0.,1.], A_ub=A, b_ub=rhs,
                     bounds=[(None,None)]*3+[(0.,1.)], method='highs')
    if not result.success:
        return None, bs, a0, result.message
    c = result.x[:3]
    assert np.max(abs(B@c)) <= 1. + 1e-6
    assert np.max(abs(nominal+B@c)-bound) <= 1e-6
    assert all(b['a0']+b['g']@c >= b['rate']**2/(2*b['h'])-1e-5 for b in bs if b['rate']<0)
    return c, bs, a0, None


def run(output, horizon_steps=20):
    assert not output.exists(), output
    assert horizon_steps in (20,40)
    folder = ROOT/'wheelleg_warp/results'
    alarm_path = folder/'height_115_alarm_timing_integer_20260929/verification.json'
    fit_path = folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    archive_path = folder/'height_115_local_states_20260928/verification.json'
    alarm, fit, archive = [json.loads(p.read_text()) for p in (alarm_path,fit_path,archive_path)]
    assert sha(fit_path)==alarm['fitted_verification_sha256']
    world=5;event=archive['events'][fit['selected_event_ids'][world]]
    scenario=HeightTerrainScenario(**event['scenario'])
    env=NativeEnv(n=3,scenario=[scenario]*3,bank_factory=bank_height_115,
                  height_conditioned=True,height_design='range115',residual_scale=0)
    env.reset();d=env.data;m=env.cpu
    va=np.array([m.joint(n).dofadr[0] for n in ACTIVE]);qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE])
    gain=np.asarray(fit['folds'][world]['gain']);r=fit['folds'][world]['reserves']
    reserve=(r['actual_A_length_m'],r['eight_joint_margin_rad'])
    start=round(alarm['rows'][world]['first_alarm_s']/DT)
    with wp.ScopedCapture() as prepared:
        wp.launch(begin,3,[env.reward])
        wp.launch(command_step,3,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,
            d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
        wp.launch(control,3,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,
            env.k['state'],env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],
            env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0],block_dim=32)
    with wp.ScopedCapture() as advanced:
        mjw.step(env.model,d)
        wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags])
        wp.launch(after,3,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,
            env.ids,env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,
            env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    for _ in range(start//40):wp.capture_launch(env.graph)
    previous=None
    for _ in range(start%40):
        previous=d.qvel.numpy()[0,va].copy()
        wp.capture_launch(prepared.graph);wp.capture_launch(advanced.graph)
    assert previous is not None and np.all(env.state.numpy()[:,0]==start)
    wp.launch(align_start,3,[wp.array(np.full(3,start,np.int32)),d.qpos,d.qvel,d.qacc_warmstart,
        d.sensordata,env.k['state'],env.state,env.command,env.contact_flags,
        env.stopped_q,env.stopped_v,env.stopped_w,env.active])
    prev=np.tile(previous,(3,1));last=np.zeros((3,3));held=None
    complete_status=f'completed_{horizon_steps/2:g}ms'
    output.mkdir(parents=True);rows=[];traces=[];timings=[];status=complete_status;failure=None
    for step in range(horizon_steps):
        wp.capture_launch(prepared.graph)
        q=d.qpos.numpy().astype(float);v=d.qvel.numpy().astype(float);nominal=d.ctrl.numpy().astype(float)
        if step==0:
            assert np.array_equal(q[0],q[1]) and np.array_equal(q[0],q[2])
            assert np.array_equal(v[0],v[1]) and np.array_equal(v[0],v[2])
            assert np.array_equal(nominal[0],nominal[1]) and np.array_equal(nominal[0],nominal[2])
        controls=nominal.copy();actions=np.zeros((3,3));models=[]
        for arm in (1,2):
            if arm==1 and step>0:
                c=held;error=None
                a0=(v[arm,va]-prev[arm])/DT-gain@last[arm]
                bs,_=barriers(q[arm,qa],v[arm,va],a0,gain,reserve)
            else:
                tic=perf_counter()
                c,bs,a0,error=solve_current(m,q[arm],v[arm],prev[arm],last[arm],gain,reserve,nominal[arm])
                timings.append((perf_counter()-tic)*1000)
            if error is not None:
                failure=dict(step=step,arm=arm,reason=error,time_after_start_ms=step/2,
                             q_active=q[arm,qa].tolist(),v_active=v[arm,va].tolist(),
                             estimated_nominal_active_acceleration=a0.tolist(),previous_action=last[arm].tolist(),
                             tightened_barrier_distances={b['name']:b['h'] for b in bs},
                             actual_A_length_m=geometry(m,q[arm])[1].tolist(),
                             eight_joint_min_margin_rad=float(margins(m,q[arm]).min()))
                status='failed_receding_model_feasibility';break
            if arm==1 and step==0:held=c.copy()
            actions[arm]=c;controls[arm]+=basis(m,q[arm],v[arm])@c
            if np.max(abs(controls[arm])-torque_box(m,v[arm]))>1e-6:
                failure=dict(step=step,arm=arm,reason='dynamic_motor_box_exceeded');status='failed_motor_gate';break
            models.append((arm,bs,a0+gain@c))
        if status!=complete_status:break
        issued=controls.astype(np.float32);d.ctrl.assign(issued)
        wp.capture_launch(advanced.graph)
        qnext=d.qpos.numpy().astype(float);vnext=d.qvel.numpy().astype(float)
        pairs=contact_sets(d,3);observed=[]
        for arm,bs,predacc in models:
            nxt,_=barriers(qnext[arm,qa],vnext[arm,va],np.zeros(4),gain,reserve)
            errors=[float(b['a0']+b['g']@actions[arm]-(nb['rate']-b['rate'])/DT) for b,nb in zip(bs,nxt)]
            observed.append(dict(arm=arm,active_acceleration_prediction_error=(predacc-(vnext[arm,va]-v[arm,va])/DT).tolist(),
                maximum_optimistic_leg_acceleration_error=max(errors[:2]),
                maximum_optimistic_joint_acceleration_error=max(errors[2:])))
        arms=[]
        wheels={m.geom('wheel_collide_'+s).id for s in ('L','R')}
        for arm in range(3):
            L=geometry(m,qnext[arm])[1];J=margins(m,qnext[arm]);P=pose(qnext[arm],scenario)
            nonwheel=any(not wheels.intersection(p) for p in pairs[arm])
            arms.append(dict(arm=('zero','current_once','current_receding')[arm],action=actions[arm].tolist(),
                min_actual_A_m=float(L.min()),min_eight_joint_margin_rad=float(J.min()),
                attitude=P,contact_pairs=[list(p) for p in sorted(pairs[arm])],
                nonwheel_contact=nonwheel,
                actual_safe=bool(L.min()>=HEIGHT_115_GEOMETRIC_MIN and J.min()>=0 and max(P['world_abs_deg'])<=5 and not nonwheel),
                peak_extra_motor_Nm=float(max(abs(issued[arm]-nominal[arm])))))
        rows.append(dict(step=step,time_after_start_ms=(step+1)/2,arms=arms,acceleration_errors=observed))
        traces.append((qnext,vnext,issued,nominal))
        prev=v[:,va].copy();last=actions.copy()
        if not np.all(env.active.numpy()):
            status='failed_physical_active_gate';failure=dict(step=step);break
    names=('wheelleg_warp/probe_height_115_live_braking.py','wheelleg_warp/probe_height_115_braking_budget.py',
        'wheelleg_warp/probe_height_115_live_common_local.py','wheelleg_warp/probe_height_115_contact_action_pair.py',
        'wheelleg_warp/probe_height_115_radial_authority.py','wheelleg_warp/probe_height_115_passive.py',
        'wheelleg_warp/native/controller.py','wheelleg_warp/native/environment.py','wheelleg_warp/native/models.py',
        'wheelleg_warp/native/terrain.py','wheelleg_ppo/tools/state_estimation.py','wheelleg_ppo/tools/wheelleg_sim.py',
        'wheelleg_ppo/tools/hardware_profile.py','wheelleg_ppo/xml/wheelleg.xml')
    if traces:np.savez_compressed(output/'trace.npz',q=np.stack([t[0] for t in traces]),v=np.stack([t[1] for t in traces]),
        issued=np.stack([t[2] for t in traces]),nominal=np.stack([t[3] for t in traces]))
    summary=dict(status=status,planned_physical_steps=horizon_steps,completed_physical_steps=len(rows),failure=failure,
        solve_python_host_median_ms=float(np.median(timings)),solve_python_host_max_ms=max(timings),
        solve_time_excludes_gpu_transfer=True,
        all_scored_steps_safe=[bool(rows) and all(r['arms'][a]['actual_safe'] for r in rows) for a in range(3)],
        completed_10ms_safe=[len(rows)>=20 and all(r['arms'][a]['actual_safe'] for r in rows[:20]) for a in range(3)],
        completed_requested_window_safe=[len(rows)==horizon_steps and all(r['arms'][a]['actual_safe'] for r in rows) for a in range(3)])
    payload=dict(role='current_state_host_in_loop_reverse_flat_braking',world=5,
        requested_horizon_ms=horizon_steps/2,previous_10ms_source_git_revision='9bb8726',
        start_s=start*DT,trigger_time_selected_offline=True,training=False,final_holdout_opened=False,
        no_hardware_realtime_claim=True,acceleration_uncertainty_margin=0.,
        previous_action_removed_from_last_measured_acceleration=True,
        summary=summary,rows=rows,fit_sha256=sha(fit_path),alarm_sha256=sha(alarm_path),
        trace_sha256=sha(output/'trace.npz') if traces else None,source_sha256={n:sha(ROOT/n) for n in names})
    (output/'verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration-ms',type=int,choices=(10,20),default=10);args=p.parse_args();target=args.output
    try:run(target,args.duration_ms*2)
    except Exception as exc:
        if not (target/'verification.json').exists():
            target.mkdir(parents=True,exist_ok=True)
            (target/'verification.json').write_text(json.dumps(dict(status='failed_execution_or_fixed_gate',
                error=f'{type(exc).__name__}: {exc}',training=False,final_holdout_opened=False,
                source_sha256=sha(Path(__file__))),ensure_ascii=False,indent=2)+'\n')
        raise
