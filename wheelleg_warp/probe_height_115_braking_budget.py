"""固定留出模型中的制动距离必要条件；不构成真实系统不变集证明。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import numpy as np
from scipy.optimize import linprog
from native.terrain import HEIGHT_115_GEOMETRIC_MIN, HeightTerrainScenario, model
from state_estimation import leg_kinematics
from probe_height_115_action_predict_loow import sha
from probe_height_115_margin import lengths
from probe_height_115_continuous_common_1nm import GAMMA, LENGTH_SCALE_M, JOINT_SCALE_RAD

DT = .0005
ACTIVE = ('alphaL', 'betaL', 'alphaR', 'betaR')


def barriers(q, v, accel, gain, reserve):
    rows = []
    hessian_error = 0.
    for side in range(2):
        sl = slice(2 * side, 2 * side + 2)
        x, velocity = q[sl], v[sl]
        j = leg_kinematics(x, velocity)[3][:, 0]
        terms = []
        for eps in (1e-5, 5e-6):
            plus = leg_kinematics(x + eps * velocity, velocity)[3][:, 0]
            minus = leg_kinematics(x - eps * velocity, velocity)[3][:, 0]
            terms.append(float((plus - minus) @ velocity / (2 * eps)))
        hessian_error = max(hessian_error, abs(terms[1] - terms[0]))
        rows.append(dict(name=('left_leg', 'right_leg')[side],
            h=float(lengths(*x) - HEIGHT_115_GEOMETRIC_MIN - reserve[0] - GAMMA * LENGTH_SCALE_M),
            rate=float(j @ velocity), a0=float(j @ accel[sl] + terms[1]), g=j @ gain[sl]))
    for k, name in enumerate(ACTIVE):
        for sign, suffix in ((-1., 'upper'), (1., 'lower')):
            rows.append(dict(name=name + '_' + suffix,
                h=float(1.5 - reserve[1] - GAMMA * JOINT_SCALE_RAD + sign * q[k]),
                rate=float(sign * v[k]), a0=float(sign * accel[k]), g=sign * gain[k]))
    assert hessian_error < 1e-4, hessian_error
    return rows, hessian_error


def evaluate(q, v, previous, gain, reserve, delay):
    basis = np.zeros((6, 3))
    for side in range(2):
        sl = slice(side * 2, side * 2 + 2)
        basis[sl, :2] = leg_kinematics(q[sl], v[sl])[3]
    basis[4:, 2] = 1.
    constraints, rhs = [], []
    rows, error = barriers(q, v, (v - previous) / DT, gain, reserve)
    checked = []
    crossed = False
    for b in rows:
        h = b['h'] + delay * b['rate'] + .5 * delay * delay * b['a0']
        rate = b['rate'] + delay * b['a0']
        if h <= 0:
            crossed = True
        if rate < 0 and h > 0:
            required = rate * rate / (2 * h)
            capacity = linprog(-b['g'], A_ub=np.r_[basis, -basis], b_ub=np.ones(12),
                               bounds=[(None, None)] * 3, method='highs')
            assert capacity.success
            best = b['a0'] + b['g'] @ capacity.x
            constraints.append(np.r_[-b['g'], 0.])
            rhs.append(b['a0'] - required)
            checked.append(dict(name=b['name'], distance_after_delay=float(h),
                closing_speed=float(-rate), required_constant_braking_acceleration=float(required),
                maximum_model_acceleration_in_1Nm_box=float(best),
                scalar_stopping_condition_possible=bool(best >= required)))
    for row in basis:
        constraints.extend((np.r_[row, -1.], np.r_[-row, -1.]))
        rhs.extend((0., 0.))
    solution = None if crossed else linprog([0., 0., 0., 1.], A_ub=constraints, b_ub=rhs,
        bounds=[(None, None)] * 3 + [(0., 1.)], method='highs')
    feasible = solution is not None and solution.success
    return dict(delay_ms=delay * 1000, tightened_boundary_crossed_during_delay=crossed,
        coupled_stopping_constraints_feasible=bool(feasible),
        minimum_model_peak_motor_delta_Nm=float(solution.x[3]) if feasible else None,
        within_1Nm_calibration=bool(feasible),
        solver_status=int(solution.status) if solution is not None else None,
        action=solution.x[:3].tolist() if feasible else None,
        directional_hessian_step_halving_error=error, closing_constraints=checked)


def run(output):
    assert not output.exists(), output
    folder = ROOT / 'wheelleg_warp/results'
    alarm_path = folder / 'height_115_alarm_timing_integer_20260929/verification.json'
    fit_path = folder / 'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    archive_path = folder / 'height_115_local_states_20260928/verification.json'
    alarm, fitted, archive = [json.loads(p.read_text()) for p in (alarm_path, fit_path, archive_path)]
    path = archive_path.parent / 'windows.npz'
    assert sha(path) == archive['windows_sha256'] == alarm['windows_sha256']
    assert sha(fit_path) == alarm['fitted_verification_sha256']
    data = np.load(path); cols = archive['columns']; rows = []
    for world, event_id in enumerate(fitted['selected_event_ids']):
        event = archive['events'][event_id]
        m = model(HeightTerrainScenario(**event['scenario']))
        pre = data[f'event_{event_id}_pre']
        qids = np.array([m.joint(n).qposadr[0] for n in ACTIVE])
        vids = np.array([m.joint(n).dofadr[0] for n in ACTIVE])
        fold = fitted['folds'][world]; gain = np.asarray(fold['gain'])
        reserve = (fold['reserves']['actual_A_length_m'], fold['reserves']['eight_joint_margin_rad'])
        for label, time in (('15ms_before_failure', event['reference_s'] - .015),
                            ('first_alarm', alarm['rows'][world]['first_alarm_s'])):
            k = int(np.argmin(abs(pre[:, 0] - time)))
            assert k > 0 and abs(pre[k, 0] - time) < 1e-6
            q = pre[k, cols['qpos_start'] + qids]
            v = pre[k, cols['qvel_start'] + vids]
            previous = pre[k - 1, cols['qvel_start'] + vids]
            rows.append(dict(world=world, state=label, time_s=float(pre[k, 0]),
                assessments=[evaluate(q, v, previous, gain, reserve, delay) for delay in (0., DT, .001)]))
    names = ('wheelleg_warp/probe_height_115_braking_budget.py', 'wheelleg_warp/probe_height_115_alarm_timing.py',
        'wheelleg_warp/probe_height_115_continuous_common_1nm.py', 'wheelleg_ppo/tools/state_estimation.py',
        'wheelleg_warp/native/terrain.py', 'wheelleg_warp/native/models.py',
        'wheelleg_ppo/tools/wheelleg_sim.py', 'wheelleg_ppo/xml/wheelleg.xml')
    output.mkdir(parents=True)
    payload = dict(role='frozen_model_constant_acceleration_stopping_budget_not_viability_proof',
        training=False, final_holdout_opened=False, physical_action_executed=False,
        constraints='two FK leg distances and four active-joint upper/lower distances; prior positional reserves retained',
        assumptions='last-step nominal acceleration and action gain remain constant; no acceleration-error bound',
        limitations='scalar stopping and simultaneous acceleration LP are necessary within this frozen model only; dynamic absolute motor box omitted; optimization never extrapolates beyond 1Nm start increment',
        summary={label: {str(delay): sum(r['assessments'][k]['within_1Nm_calibration'] for r in rows if r['state'] == label)
            for k, delay in enumerate((0., .5, 1.))} for label in ('15ms_before_failure', 'first_alarm')},
        rows=rows, archive_sha256=sha(archive_path), windows_sha256=sha(path), fitted_sha256=sha(fit_path),
        alarm_sha256=sha(alarm_path), source_sha256={n: sha(ROOT / n) for n in names})
    (output / 'verification.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    for r in rows:
        print(r['world'], r['state'], [(a['delay_ms'], a['minimum_model_peak_motor_delta_Nm'], a['within_1Nm_calibration']) for a in r['assessments']])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
