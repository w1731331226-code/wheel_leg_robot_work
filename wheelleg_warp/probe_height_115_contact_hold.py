"""固定已归档粗糙候选，零/单步/持续20ms同图比较；不重选动作。"""
import argparse
import json
from pathlib import Path
import numpy as np
from probe_height_115_action_predict_loow import ROOT, DT, sha, compare_start
from probe_height_115_live_common_local import simulate, common_basis
from probe_height_115_radial_authority import torque_box, pose
from probe_height_115_contact_action_pair import margins
from probe_height_115_passive import geometry
from probe_height_115_braking_budget import ACTIVE, barriers
from native.terrain import model, HeightTerrainScenario, HEIGHT_115_GEOMETRIC_MIN


def run(recovery_steps=0):
    folder=ROOT/'wheelleg_warp/results';source=folder/'height_115_contact_candidate_20260929'
    assert recovery_steps in (0,200,2000)
    output=folder/{0:'height_115_contact_hold_v2_20260929',200:'height_115_contact_release_20260929',
                   2000:'height_115_contact_release_1s_20260929'}[recovery_steps]
    assert not output.exists()
    paths=[source/'verification.json',folder/'height_115_contact_response_2nm_20260929/verification.json',
        folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json',
        folder/'height_115_local_states_20260928/verification.json']
    candidate,response,fit,archive=[json.loads(p.read_text()) for p in paths]
    old=np.load(source/'trace.npz');assert sha(source/'trace.npz')==candidate['trace_sha256']
    c=np.array(candidate['checks'][1]['action']);assert candidate['checks'][1]['braking_gate_pass']
    scenario=HeightTerrainScenario(**archive['events'][fit['selected_event_ids'][3]]['scenario'])
    m=model(scenario);start=response['rows'][2]['start_step'];assert response['rows'][2]['world']==3
    # Preserve every first-step arm from the original paired run; only extend arm 1.
    schedule=np.zeros((40,7,3));schedule[0]=old['coefficients'];schedule[1:,1]=c
    rec,states,stats,_=simulate([scenario],[start],np.zeros((7,3)),schedule=schedule,followup_steps=recovery_steps)
    qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    wheels={m.geom('wheel_collide_'+s).id for s in ('L','R')};rsv=fit['folds'][3]['reserves']
    reserve=(rsv['actual_A_length_m'],rsv['eight_joint_margin_rad']);rows=[];summaries=[]
    for arm,r in enumerate(rec):
        compare_start(states[arm],states[0],m.nq,m.nv,f'hold arm{arm}')
    output.mkdir();np.savez_compressed(output/'trace.npz',schedule=schedule,starts=states,
        **{k:np.stack([r[k] for r in rec]) for k in ('pre','post','post_velocity','applied')},contact_raw=rec[0]['contact_raw'])
    first_q_error=float(abs(rec[1]['post'][0]-old['post'][1,0]).max())
    first_v_error=float(abs(rec[1]['post_velocity'][0]-old['post_velocity'][1,0]).max())
    if recovery_steps==2000:
        (output/'verification.json').write_text(json.dumps(dict(status='collected_for_independent_audit',
            recovery_steps=recovery_steps,first_q_error=first_q_error,first_v_error=first_v_error,
            kernel_clipping_events=float(stats[:,1].sum()),trace_sha256=sha(output/'trace.npz'),
            previous_source_git_revision='053d8e8',
            input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[source/'trace.npz']},
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_live_common_local.py',
                ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')}),indent=2)+'\n')
        print('Collected 20ms hold and 1000ms release; physical results require independent audit.',flush=True)
        return
    if first_q_error!=0 or first_v_error!=0:
        (output/'verification.json').write_text(json.dumps(dict(status='failed_before_scoring',
            first_q_error=first_q_error,first_v_error=first_v_error,
            trace_sha256=sha(output/'trace.npz'),source_sha256=sha(Path(__file__))),indent=2)+'\n')
        raise AssertionError((first_q_error,first_v_error))
    pulse={k:old[k][1] for k in ('pre','post','post_velocity','applied')}
    pulse['contacts']=[{tuple(sorted((int(a),int(b)))) for w,a,b in frame if int(w)==1} for frame in old['contact_raw']]
    zero_error=float(abs(rec[0]['post'][:40]-old['post'][0]).max());assert zero_error<=2e-6
    if recovery_steps:
        prior_path=folder/'height_115_contact_hold_v2_20260929/verification.json'
        prior=json.loads(prior_path.read_text());prior_trace=prior_path.parent/'trace.npz'
        assert sha(prior_trace)==prior['trace_sha256']
        prior_data=np.load(prior_trace)
        for key in ('pre','post','post_velocity','applied'):
            assert np.array_equal(np.stack([r[key][:40] for r in rec]),prior_data[key]),key
        paths.extend((prior_path,prior_trace))
    for arm,r in enumerate((rec[0],rec[1],pulse)):
        one=[]
        for t in range(len(r['pre'])):
            pre=r['pre'][t];q=pre[:m.nq];v=pre[m.nq:m.nq+m.nv];nom=pre[m.nq+m.nv:m.nq+m.nv+6]
            nxt=r['post'][t];vn=r['post_velocity'][t];ctrl=r['applied'][t]
            leg=geometry(m,nxt)[1];prior_leg=geometry(m,q)[1];jm=float(margins(m,nxt).min());att=pose(nxt,scenario)
            nonwheel=any(not wheels.intersection(p) for p in r['contacts'][t])
            peak=float(abs(ctrl-nom).max());box=float((abs(ctrl)-torque_box(m,v)).max())
            action=c if (arm==1 and t<40) or (arm==2 and t==0) else np.zeros(3)
            match=float(abs(ctrl-nom-common_basis(m,q,v)@action).max())
            b,_=barriers(q[qa],v[va],np.zeros(4),np.zeros((4,3)),reserve)
            bn,_=barriers(nxt[qa],vn[va],np.zeros(4),np.zeros((4,3)),reserve)
            slack={x['name']:float((y['rate']-x['rate'])/DT-x['rate']**2/(2*x['h']))
                   for x,y in zip(b,bn) if x['rate']<0 and x['h']>0}
            gates=dict(geometry=bool(leg.min()>=HEIGHT_115_GEOMETRIC_MIN and jm>=0),
                pose=bool(max(att['world_abs_deg'])<=5),nonwheel=not nonwheel,
                motor=bool(peak<=2+1e-5 and box<=1e-6 and match<=1e-5),active=bool(pre[-1]==1))
            one.append(dict(step=t,time_ms=(t+1)*.5,gates=gates,all_gates=all(gates.values()),
                true_A_margin_m=(leg-HEIGHT_115_GEOMETRIC_MIN).tolist(),joint_min_margin_rad=jm,
                interval_true_A_rate_m_s=((leg-prior_leg)/DT).tolist(),
                instantaneous_FK_rate_m_s=[x['rate'] for x in bn[:2]],
                tightened_FK_distance_m=[x['h'] for x in bn[:2]],
                actual_constant_acceleration_slack=slack,
                maximum_world_attitude_deg=max(att['world_abs_deg']),peak_extra_motor_Nm=peak,
                absolute_motor_box_excess_Nm=box,command_error_Nm=match,
                contacts_equal_zero=r['contacts'][t]==rec[0]['contacts'][t]))
        failed=next((x for x in one if not x['all_gates']),None)
        summaries.append(dict(arm=('zero','hold20ms_then_release' if recovery_steps else 'hold20ms','archived_pulse0.5ms')[arm],
            scored_duration_ms=len(one)*.5,
            complete_scored_window_safe=all(x['all_gates'] for x in one),
            complete_20ms_safe=all(x['all_gates'] for x in one[:40]),first_failure=failed,
            recovery_100ms_safe=bool(len(one)==240 and all(x['all_gates'] for x in one[40:])),
            terminal_true_A_target_error_m=[x+HEIGHT_115_GEOMETRIC_MIN-.115 for x in one[-1]['true_A_margin_m']],
            complete_5ms_safe=all(x['all_gates'] for x in one[:10]),
            complete_10ms_safe=all(x['all_gates'] for x in one[:20]),
            minimum_true_A_margin_m=min(min(x['true_A_margin_m']) for x in one),
            peak_extra_motor_Nm=max(x['peak_extra_motor_Nm'] for x in one),
            contacts_equal_zero_steps=sum(x['contacts_equal_zero'] for x in one),terminal=one[-1]))
        rows.append(one)
    result=dict(role='fixed_rough_contact_candidate_20ms_hold',recovery_steps=recovery_steps,
        previous_20ms_source_git_revision='087f86f',action=c.tolist(),summaries=summaries,rows=rows,
        first_step_q_v_exact_match=True,zero_replay_q_max_error=zero_error,
        pulse_comparison_source='Archived candidate trace arm1; current trace arm2 remains original +F pulse.',
        limitations='Fixed candidate selected with future paired-state calibration, offline contact timing. No online trigger/estimator or full-episode claim. Constant-acceleration slack is a separate diagnostic, not used to replace physical safety gates.',
        kernel_clipping_events=float(stats[:,1].sum()),trace_sha256=sha(output/'trace.npz'),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[source/'trace.npz']},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_live_common_local.py',
            ROOT/'wheelleg_warp/probe_height_115_braking_budget.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(summaries,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--release-observation',action='store_true')
    parser.add_argument('--release-1s',action='store_true');args=parser.parse_args()
    assert not (args.release_1s and args.release_observation)
    run(2000 if args.release_1s else 200 if args.release_observation else 0)
