"""反向平地首次警报：位置短窗动作与模型制动动作的同图10ms对照。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import numpy as np
from native.terrain import HEIGHT_115_GEOMETRIC_MIN, HeightTerrainScenario, model
from probe_height_115_action_predict_loow import compare_start, sha
from probe_height_115_live_common_local import simulate, common_basis
from probe_height_115_contact_action_pair import command, margins
from probe_height_115_passive import geometry
from probe_height_115_radial_authority import pose
from probe_height_115_braking_budget import barriers


def run(output):
    assert not output.exists(), output
    folder = ROOT / 'wheelleg_warp/results'
    budget_path = folder / 'height_115_braking_budget_bounded_20260929/verification.json'
    alarm_path = folder / 'height_115_alarm_timing_integer_20260929/verification.json'
    fit_path = folder / 'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    archive_path = folder / 'height_115_local_states_20260928/verification.json'
    budget, alarm, fit, archive = [json.loads(p.read_text()) for p in (budget_path, alarm_path, fit_path, archive_path)]
    assert budget['alarm_sha256'] == sha(alarm_path) and budget['fitted_sha256'] == sha(fit_path)
    assert all(sha(ROOT / n) == h for n, h in budget['source_sha256'].items())
    row = next(r for r in budget['rows'] if r['world'] == 5 and r['state'] == 'first_alarm')
    braking = row['assessments'][0]
    assert braking['within_1Nm_calibration'] and braking['delay_ms'] == 0
    event = archive['events'][fit['selected_event_ids'][5]]
    scenario = HeightTerrainScenario(**event['scenario']); m = model(scenario)
    actions = np.zeros((9, 3))
    actions[1] = alarm['rows'][5]['candidate']['action']
    actions[2] = braking['action']
    records, starts, _, _ = simulate([scenario], [round(row['time_s'] / .0005)], actions)
    for arm in range(9):
        compare_start(starts[arm], starts[0], m.nq, m.nv, f'world5 arm{arm}')
        assert np.all(records[arm]['pre'][:20, -1] == 1)
    qa = np.array([m.joint(n).qposadr[0] for n in ('alphaL','betaL','alphaR','betaR')])
    va = np.array([m.joint(n).dofadr[0] for n in ('alphaL','betaL','alphaR','betaR')])
    first = records[0]['pre'][0]
    q, v = first[qa], first[m.nq + va]
    acc = (v - records[0]['previous_qvel'][va]) / .0005
    fold = fit['folds'][5]; reserve = (fold['reserves']['actual_A_length_m'], fold['reserves']['eight_joint_margin_rad'])
    bs, _ = barriers(q, v, acc, np.asarray(fold['gain']), reserve)
    live_slack = min(b['a0'] + b['g'] @ actions[2] - b['rate']**2/(2*b['h'])
                     for b in bs if b['rate'] < 0 and b['h'] > 0)
    assert all(b['h'] > 0 for b in bs) and live_slack >= -1e-5, live_slack
    wheels = {m.geom('wheel_collide_' + s).id for s in ('L', 'R')}
    rows, traces = [], {}
    for arm, name in enumerate(('zero', 'position_horizon', 'stopping_condition')):
        record = records[arm]; clip = match = peak = 0.
        for t in range(20):
            pre = record['pre'][t]; fullq, fullv = pre[:m.nq], pre[m.nq:m.nq+m.nv]
            base = pre[m.nq+m.nv:m.nq+m.nv+6]
            request = base + common_basis(m, fullq, fullv) @ actions[arm]
            expected, _, _ = command(m, fullq, fullv, base, actions[arm])
            clip = max(clip, float(max(abs(request-expected))))
            match = max(match, float(max(abs(record['applied'][t]-expected))))
            peak = max(peak, float(max(abs(record['applied'][t]-base))))
        assert clip <= 1e-6 and match <= 1e-5, (name, clip, match)
        L = geometry(m, record['post'][:20])[1].min(axis=1)
        J = margins(m, record['post'][:20]).min(axis=1)
        P = np.asarray([pose(z, scenario)['world_abs_deg'] for z in record['post'][:20]])
        nonwheel = sum(any(not wheels.intersection(p) for p in pairs) for pairs in record['contacts'][:20])
        rows.append(dict(arm=name, action=actions[arm].tolist(), min_actual_A_length_m=float(L.min()),
            min_eight_joint_margin_rad=float(J.min()), peak_abs_attitude_deg=float(P.max()),
            nonwheel_steps=nonwheel, contact_equal_to_zero_steps=sum(a==b for a,b in zip(records[0]['contacts'][:20],record['contacts'][:20])),
            peak_motor_delta_Nm=peak, clipping_Nm=clip, motor_match_error_Nm=match,
            safe_5ms=bool(L[:10].min()>=HEIGHT_115_GEOMETRIC_MIN and J[:10].min()>=0 and P[:10].max()<=5 and not nonwheel),
            safe_10ms=bool(L.min()>=HEIGHT_115_GEOMETRIC_MIN and J.min()>=0 and P.max()<=5 and not nonwheel),
            terminal_10ms_leg_rate_m_s=float((L[-1]-L[-2])/.0005)))
        traces[name+'_qpos'] = record['post'][:20]
        traces[name+'_ctrl'] = record['applied'][:20]
    output.mkdir(parents=True)
    np.savez_compressed(output/'trace.npz', **traces)
    sources = ('wheelleg_warp/probe_height_115_braking_response.py','wheelleg_warp/probe_height_115_braking_budget.py',
        'wheelleg_warp/probe_height_115_live_common_local.py','wheelleg_warp/probe_height_115_contact_action_pair.py',
        'wheelleg_warp/probe_height_115_passive.py','wheelleg_warp/probe_height_115_radial_authority.py',
        'wheelleg_warp/native/controller.py','wheelleg_warp/native/environment.py','wheelleg_warp/native/models.py',
        'wheelleg_warp/native/terrain.py','wheelleg_ppo/tools/state_estimation.py','wheelleg_ppo/tools/wheelleg_sim.py',
        'wheelleg_ppo/tools/hardware_profile.py','wheelleg_ppo/xml/wheelleg.xml')
    payload = dict(role='one_public_reverse_flat_alarm_stopping_action_10ms_not_online_controller',
        training=False, final_holdout_opened=False, world=5, start_s=alarm['rows'][5]['first_alarm_s'],
        model_live_stopping_acceleration_min_slack=live_slack, scoring_horizon_s=.01, helper_action_window_s=.02,
        calibration_limitation='heldout constant acceleration model at this new state; no proven acceleration error bound',
        rows=rows, budget_sha256=sha(budget_path), alarm_sha256=sha(alarm_path), fit_sha256=sha(fit_path),
        trace_sha256=sha(output/'trace.npz'), source_sha256={n:sha(ROOT/n) for n in sources})
    (output/'verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for r in rows:print(r,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
