"""同一 Warp 图九臂分叉：0.115 m 六世界 5 ms 动作预测留出诊断。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import numpy as np
from native.terrain import HEIGHT_115_GEOMETRIC_MIN, HeightTerrainScenario, model
from probe_height_115_action_predict_loow import (
    ARMS, DT, STEPS, MOTOR_CLIP_LIMIT, MOTOR_MATCH_LIMIT, actions, compare_start,
    fit_gain, optimistic_error, score, summarize, require, sha)
from probe_height_115_contact_action_pair import command, margins
from probe_height_115_live_common_local import common_basis, simulate
from probe_height_115_margin import cases
from probe_height_115_passive import geometry

ACTIVE = ('alphaL', 'betaL', 'alphaR', 'betaR')


def one_start(world, event_id, lag, archive, windows, peak_nm=.08):
    event = archive['events'][event_id]
    scenario = HeightTerrainScenario(**event['scenario'])
    m = model(scenario)
    nq, nv = m.nq, m.nv
    step = int(round(event['reference_s'] / DT)) - lag
    source = windows[f'event_{event_id}_pre']
    index = int(np.argmin(abs(source[:, 0] - step * DT)))
    require(abs(float(source[index, 0] - step * DT)) < 2e-6,
            f'world{world} {lag} archive event time missing')
    old = source[index]
    cols = archive['columns']
    old_q = old[cols['qpos_start']:cols['qpos_start'] + nq]
    old_v = old[cols['qvel_start']:cols['qvel_start'] + nv]
    coeff, eps = actions(m, old_q, np.zeros(nv))  # B(q) only; no future state or archived velocity.
    coeff *= peak_nm / .08
    eps *= peak_nm / .08
    records, start_state, stats, _ = simulate([scenario], [step], coeff)
    qa = np.array([m.joint(name).qposadr[0] for name in ACTIVE])
    va = np.array([m.joint(name).dofadr[0] for name in ACTIVE])
    joint_limits = np.array([m.jnt_range[m.joint(name).id] for name in ACTIVE])
    for arm, row in enumerate(records):
        compare_start(start_state[arm], start_state[0], nq, nv,
                      f'world{world} {lag} same-graph arm{arm}')
        require(np.array_equal(row['previous_qvel'], records[0]['previous_qvel']),
                f'world{world} {lag} previous same-graph dq mismatch arm{arm}')
        require(np.all(row['pre'][:STEPS, -1] == 1),
                f'world{world} {lag} inactive in first 5ms arm{arm}')
        pre0 = row['pre'][0]
        initial = np.r_[pre0[:nq + nv], pre0[nq + nv:nq + nv + 6]]
        reference = np.r_[start_state[arm, 1:1 + nq + nv],
                          start_state[arm, 1 + nq + 2 * nv:1 + nq + 2 * nv + 6]]
        require(float(np.max(abs(initial - reference))) <= 1e-6,
                f'world{world} {lag} start snapshot mismatch arm{arm}')
    prior_v = records[0]['previous_qvel']
    require(np.isfinite(prior_v).all(), f'world{world} {lag} previous dq nonfinite')
    actual = np.stack([geometry(m, row['post'][:STEPS])[1] for row in records])
    joints = np.stack([margins(m, row['post'][:STEPS]) for row in records])
    sampled = np.stack([row['summary'][:STEPS, 1] for row in records])
    require(float(np.max(abs(actual.min(axis=-1) - sampled))) < 1e-5,
            f'world{world} {lag} true A-chain sample mismatch')
    motor_match = 0.
    motor_clip = 0.
    peak_delta = 0.
    initial_request = 0.
    initial_applied = 0.
    initial_request_by_arm = np.zeros(len(ARMS))
    initial_applied_by_arm = np.zeros(len(ARMS))
    for arm, row in enumerate(records):
        for t in range(STEPS):
            pre = row['pre'][t]
            q, v = pre[:nq], pre[nq:nq + nv]
            base = pre[nq + nv:nq + nv + 6]
            request = base + common_basis(m, q, v) @ coeff[arm]
            expected, _, _ = command(m, q, v, base, coeff[arm])
            motor_match = max(motor_match, float(np.max(abs(row['applied'][t] - expected))))
            motor_clip = max(motor_clip, float(np.max(abs(request - expected))))
            peak_delta = max(peak_delta, float(np.max(abs(row['applied'][t] - base))))
            if t == 0:
                initial_request_by_arm[arm] = float(np.max(abs(request - base)))
                initial_applied_by_arm[arm] = float(np.max(abs(row['applied'][t] - base)))
                initial_request = max(initial_request, initial_request_by_arm[arm])
                initial_applied = max(initial_applied, initial_applied_by_arm[arm])
    require(np.all(abs(initial_request_by_arm[1:7] - peak_nm) <= 1e-5) and
            np.all(abs(initial_applied_by_arm[1:7] - peak_nm) <= 1e-5) and
            initial_request <= peak_nm + 1e-5 and initial_applied <= peak_nm + 1e-5,
            f'world{world} {lag} initial single-axis action is not {peak_nm} Nm: '
            f'request={initial_request:.9g} applied={initial_applied:.9g}')
    require(motor_match <= MOTOR_MATCH_LIMIT,
            f'world{world} {lag} applied motor command mismatch {motor_match:.9g}')
    require(motor_clip <= MOTOR_CLIP_LIMIT,
            f'world{world} {lag} motor clipped during 5ms {motor_clip:.9g}')
    base = records[0]
    return dict(world=world, event_id=event_id, event=event['kind'], lag_ms=lag / 2,
                start_s=step * DT, q0=base['pre'][0, qa].copy(),
                v0=base['pre'][0, nq + va].copy(), vprev=prior_v[va].copy(),
                active_joint_limits=joint_limits, coeff=coeff, eps=eps,
                first_v=np.stack([row['pre'][1, nq + va] for row in records]),
                actual=actual, joints=joints,
                contact_changed=np.array([sum(a != b for a, b in zip(base['contacts'][:STEPS],
                                                                      row['contacts'][:STEPS]))
                                            for row in records]),
                first_contact_changed=np.array([base['contacts'][0] != row['contacts'][0]
                                                for row in records]),
                max_abs_attitude=np.max(np.stack([row['summary'][:STEPS, 3:6] for row in records]), axis=1),
                historical_start_qpos_max_abs_difference=float(np.max(abs(base['pre'][0, :nq] - old_q))),
                historical_start_qvel_max_abs_difference=float(np.max(abs(base['pre'][0, nq:nq + nv] - old_v))),
                initial_requested_delta_Nm=initial_request, initial_applied_delta_Nm=initial_applied,
                initial_requested_by_arm_Nm=initial_request_by_arm,
                initial_applied_by_arm_Nm=initial_applied_by_arm,
                motor_match=motor_match, motor_clip=motor_clip, peak_delta=peak_delta,
                same_graph_start_max_abs_difference=float(np.max(abs(start_state - start_state[0]))),
                previous_qvel_full=prior_v.copy(), full_40_step_kernel_stats=stats.copy())


def run(output, peak_nm=.08):
    require(not output.exists(), f'output already exists: {output}')
    require(peak_nm in (.08, 1.), f'unregistered initial motor peak {peak_nm} Nm')
    source = ROOT / 'wheelleg_warp/results/height_115_local_states_20260928/verification.json'
    archive = json.loads(source.read_text())
    data_path = source.parent / 'windows.npz'
    require(sha(data_path) == archive['windows_sha256'], 'input archive data hash mismatch')
    selected = [min((i for i, e in enumerate(archive['events']) if e['world'] == world),
                    key=lambda i: archive['events'][i]['reference_s']) for world in range(6)]
    require([archive['events'][i]['scenario'] for i in selected] ==
            [asdict(s) for s in cases()[:6]], 'six public normal worlds changed')
    windows = np.load(data_path)
    preflight = one_start(1, selected[1], 10, archive, windows, peak_nm)
    print('same-graph world1 late preflight passed', flush=True)
    samples = [preflight]
    for world, event_id in enumerate(selected):
        for lag in (30, 10):
            if world == 1 and lag == 10:
                continue
            samples.append(one_start(world, event_id, lag, archive, windows, peak_nm))
        print(f'captured world {world}', flush=True)
    samples.sort(key=lambda s: (s['world'], -s['lag_ms']))
    require(len(samples) == 12, 'two starts per six worlds missing')
    folds, heldout_rows = [], []
    for heldout in range(6):
        training_worlds = set(range(6)) - {heldout}
        residuals, inner = [], []
        for inner_world in sorted(training_worlds):
            gain = fit_gain(samples, training_worlds - {inner_world})
            errors = [optimistic_error(s, gain) for s in samples if s['world'] == inner_world]
            require(len(errors) == 2, 'inner leave-one-world-out missing two starts')
            residuals.extend(errors)
            inner.append(dict(world=inner_world, max_optimistic_leg_m=max(e[0] for e in errors),
                              max_optimistic_joint_rad=max(e[1] for e in errors)))
        reserve = np.maximum(0., np.max(np.asarray(residuals), axis=0))
        gain = fit_gain(samples, training_worlds)
        rows = score(samples, heldout, gain, reserve)
        folds.append(dict(heldout_world=heldout, gain=gain.tolist(), inner_world_residuals=inner,
                          reserves=dict(actual_A_length_m=float(reserve[0]),
                                        eight_joint_margin_rad=float(reserve[1])),
                          summary=summarize(rows)))
        heldout_rows.extend(rows)
    late_actual_only = []
    for world in range(6):
        late = [r for r in heldout_rows if r['world'] == world and r['lag_ms'] == 5.]
        zero = next(r for r in late if r['action'] == 'zero')
        if not zero['actual_combined_safe']:
            late_actual_only.extend(r for r in late if r['action'] != 'zero' and r['actual_combined_safe'])
    late_positive = sum(r['predicted_leg_margin_after_reserve_m'] > 0 and
                        r['predicted_joint_margin_after_reserve_rad'] > 0
                        for r in heldout_rows if r['lag_ms'] == 5. and r['action'] != 'zero')
    old_failures = []
    for directory in ('height_115_action_predict_loow_20260928',
                      'height_115_action_predict_loow_v2_20260928',
                      'height_115_action_predict_loow_v3_20260928',
                      'height_115_action_predict_current_20260928',
                      'height_115_action_predict_current_v2_20260928'):
        path = ROOT / 'wheelleg_warp/results' / directory / 'verification.json'
        old = json.loads(path.read_text())
        old_failures.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                                 status=old.get('status'), error=old.get('error'),
                                 scientific_loow_result=False))
    old_simulator = hashlib.sha256(subprocess.check_output(
        ['git', 'show', '2b870ee:wheelleg_warp/probe_height_115_live_common_local.py'], cwd=ROOT)).hexdigest()
    old_single_graph = hashlib.sha256(subprocess.check_output(
        ['git', 'show', '666716a:wheelleg_warp/probe_height_115_action_predict_single_graph.py'], cwd=ROOT)).hexdigest()
    output.mkdir(parents=True)
    trace = output / 'heldout_inputs.npz'
    np.savez_compressed(trace, q0=np.stack([s['q0'] for s in samples]),
        v0=np.stack([s['v0'] for s in samples]), vprev=np.stack([s['vprev'] for s in samples]),
        actions=np.stack([s['coeff'] for s in samples]),
        first_step_active_v=np.stack([s['first_v'] for s in samples]),
        actual_A_length_m=np.stack([s['actual'] for s in samples]),
        actual_eight_joint_margin_rad=np.stack([s['joints'] for s in samples]))
    sources = ('wheelleg_warp/probe_height_115_action_predict_single_graph.py',
               'wheelleg_warp/probe_height_115_action_predict_loow.py',
               'wheelleg_warp/probe_height_115_live_common_local.py',
               'wheelleg_warp/probe_height_115_contact_action_pair.py',
               'wheelleg_warp/probe_height_115_passive.py',
               'wheelleg_warp/probe_height_115_margin.py',
               'wheelleg_warp/probe_height_115_warp_local_lp.py',
               'wheelleg_warp/native/controller.py', 'wheelleg_warp/native/environment.py',
               'wheelleg_warp/native/terrain.py', 'wheelleg_warp/native/models.py',
               'wheelleg_ppo/tools/state_estimation.py', 'wheelleg_ppo/tools/wheelleg_sim.py',
               'wheelleg_ppo/tools/hardware_profile.py', 'wheelleg_ppo/xml/wheelleg.xml')
    if peak_nm == 1.:
        sources += ('wheelleg_warp/probe_height_115_action_predict_1nm.py',)
    payload = dict(role=('public_115m_single_graph_common3_1Nm_5ms_world_holdout_prediction'
                         if peak_nm == 1. else 'public_115m_single_graph_common3_5ms_world_holdout_prediction'),
        status='completed', training=False, final_holdout_opened=False,
        nominal_height_m=.115, simulated_geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,
        physical_step_s=DT, horizon_steps=STEPS, arms=ARMS,
        action_rule=f'archived current active q only sets B(q) peak{peak_nm}Nm per F/H/W single axis; +/- mixed (F,-H,W)/3',
        initial_motor_peak_Nm=peak_nm,
        preflight='world1 step late, all nine same-graph starts and first 10 steps; clipping or mismatch stops run',
        preflight_passed=True,
        predictor='single-graph current active q/dq, same-graph previous active dq, current common3 command, constant 4x3 G',
        no_contact_or_passive_or_future_state_predictor_input=True,
        historical_archive_use='six scenario specs, approximate event times, current q for action sizing; q/v start differences diagnostic only',
        same_graph_previous_qvel_captured_before_align=True,
        old_simulate_git_revision='2b870ee', old_simulate_source_sha256=old_simulator,
        previous_single_graph_git_revision='666716a', previous_single_graph_source_sha256=old_single_graph,
        previous_single_graph_verification_sha256=sha(ROOT / 'wheelleg_warp/results/height_115_action_predict_single_graph_20260928/verification.json'),
        previous_single_graph_inputs_sha256=sha(ROOT / 'wheelleg_warp/results/height_115_action_predict_single_graph_20260928/heldout_inputs.npz'),
        old_cross_run_preflight_failures=old_failures,
        calibration='outer leave one world; inner leave one world of five training worlds; max positive inner residual',
        simulated_sensors_not_hardware_noise_or_latency_validated=True,
        source_state_verification_sha256=sha(source), source_windows_sha256=archive['windows_sha256'],
        heldout_inputs_sha256=sha(trace), selected_event_ids=selected,
        max_historical_start_qpos_difference=max(s['historical_start_qpos_max_abs_difference'] for s in samples),
        max_historical_start_qvel_difference=max(s['historical_start_qvel_max_abs_difference'] for s in samples),
        max_same_graph_start_difference=max(s['same_graph_start_max_abs_difference'] for s in samples),
        max_initial_requested_motor_delta_Nm=max(s['initial_requested_delta_Nm'] for s in samples),
        max_initial_applied_motor_delta_Nm=max(s['initial_applied_delta_Nm'] for s in samples),
        max_initial_requested_by_arm_Nm=np.max(np.stack([s['initial_requested_by_arm_Nm'] for s in samples]), axis=0).tolist(),
        max_initial_applied_by_arm_Nm=np.max(np.stack([s['initial_applied_by_arm_Nm'] for s in samples]), axis=0).tolist(),
        max_5ms_motor_command_match_error_Nm=max(s['motor_match'] for s in samples),
        max_5ms_motor_clip_Nm=max(s['motor_clip'] for s in samples),
        max_5ms_applied_motor_delta_Nm=max(s['peak_delta'] for s in samples),
        first_step_contact_changed_branches=sum(int(np.count_nonzero(s['first_contact_changed'])) for s in samples),
        late_zero_actually_failed_worlds=sum(not next(r for r in heldout_rows
            if r['world'] == w and r['lag_ms'] == 5. and r['action'] == 'zero')['actual_combined_safe']
            for w in range(6)),
        late_actual_only_safe_nonzero_actions=len(late_actual_only),
        critical_late_positive_predicted_nonzero_actions=late_positive,
        critical_no_positive_candidate_warning=('No late nonzero action has strictly positive leg and joint predicted margins after reserve'
                                                if late_positive == 0 else None),
        late_predicted_and_actual_safe_actions=sum(r['predicted_leg_margin_after_reserve_m'] > 0 and
            r['predicted_joint_margin_after_reserve_rad'] > 0 for r in late_actual_only),
        summary=summarize(heldout_rows), folds=folds, heldout_rows=heldout_rows,
        source_sha256={name: sha(ROOT / name) for name in sources})
    (output / 'verification.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(payload['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    target = parser.parse_args().output
    try:
        run(target)
    except Exception as exc:
        if not (target / 'verification.json').exists():
            target.mkdir(parents=True, exist_ok=True)
            (target / 'verification.json').write_text(json.dumps(dict(
                role='public_115m_single_graph_common3_5ms_world_holdout_prediction',
                status='failed_pre_registered_gate_or_execution', error=f'{type(exc).__name__}: {exc}',
                training=False, final_holdout_opened=False,
                source_sha256=sha(Path(__file__)),
                previous_simulate_git_revision='2b870ee'), ensure_ascii=False, indent=2) + '\n')
        raise
