"""0.115 m 首次失效前：六世界整体留出的 5 ms 共模动作预测诊断。"""
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
from probe_height_115_contact_action_pair import command, margins
from probe_height_115_live_common_local import common_basis, simulate
from probe_height_115_margin import cases, lengths
from probe_height_115_passive import geometry

DT = .0005
STEPS = 10
ARMS = ('zero', 'F+', 'F-', 'H+', 'H-', 'W+', 'W-', 'mix+', 'mix-')
ACTIVE = ('alphaL', 'betaL', 'alphaR', 'betaR')
START_LIMITS = dict(qpos=1e-7, qvel=1e-6, warmstart=1e-5, ctrl=1e-6)
MOTOR_MATCH_LIMIT = 1e-5
MOTOR_CLIP_LIMIT = 1e-6
CONTROLLER_AT_ARCHIVE = '54004e5dc718e9b6a6a97669bf5dcf59cb0743729bfb0dcacd08fb2f3cf42af3'
CONTROLLER_NOW = '03d56704bf41ab4987a01affef434bc776771ea24077b6ac6fd5f44bed1bd15f'


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controller_audit():
    name = 'wheelleg_warp/native/controller.py'
    archived = json.loads((ROOT / 'wheelleg_warp/results/height_115_local_states_20260928/verification.json').read_text())
    try:
        historical = hashlib.sha256(subprocess.check_output(
            ['git', 'show', 'c4fed2c:' + name], cwd=ROOT)).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        historical = None
    current = sha(ROOT / name)
    return dict(archive_sha256=archived['source_sha256'].get(name), c4fed2c_sha256=historical,
                current_sha256=current,
                archive_matches_c4fed2c=archived['source_sha256'].get(name) == historical == CONTROLLER_AT_ARCHIVE,
                current_matches_2b870ee_version=current == CONTROLLER_NOW,
                archive_git_revision='c4fed2c', current_change='2b870ee optional four-column target floor',
                experiment_reference_columns=3)


def compare_start(actual, reference, nq, nv, label, strict_warm=True):
    """start_state 与归档 pre 均为 time, q, v, warmstart, base ctrl。"""
    require(abs(float(actual[0] - reference[0])) < 2e-6, f'{label}: time mismatch')
    sections = (('qpos', 1, nq), ('qvel', 1 + nq, nv),
                ('warmstart', 1 + nq + nv, nv), ('ctrl', 1 + nq + 2 * nv, 6))
    errors = {}
    for name, start, width in sections:
        error = float(np.max(abs(actual[start:start + width] - reference[start:start + width])))
        errors[name] = error
        if name != 'warmstart' or strict_warm:
            require(error <= START_LIMITS[name], f'{label}: {name} mismatch {error:.9g}')
    return errors


def archive_difference(live, archived, nq, nv, with_warm):
    """旧归档只作差异审计，不参与当前来源的通过门。"""
    shift = 1 if with_warm else 0
    result = dict(qpos=float(np.max(abs(live[shift:shift + nq] - archived[1:1 + nq]))),
                  qvel=float(np.max(abs(live[shift + nq:shift + nq + nv] - archived[1 + nq:1 + nq + nv]))))
    if with_warm:
        result['warmstart'] = float(np.max(abs(live[1 + nq + nv:1 + nq + 2 * nv] -
                                              archived[1 + nq + nv:1 + nq + 2 * nv])))
    result['ctrl'] = float(np.max(abs(live[shift + nq + nv + (nv if with_warm else 0):
                                      shift + nq + nv + (nv if with_warm else 0) + 6] -
                                      archived[1 + nq + 2 * nv:1 + nq + 2 * nv + 6])))
    return result


def pair_current_5ms(reference, candidate, m, nq, nv, label, offset=0):
    prior_pre = reference['pre'][offset:offset + STEPS]
    prior_post = reference['post'][offset:offset + STEPS]
    now_pre = candidate['pre'][:STEPS]
    now_post = candidate['post'][:STEPS]
    require(len(prior_pre) == STEPS and len(now_pre) == STEPS, f'{label}: incomplete 5ms')
    q_error = float(np.max(abs(prior_post - now_post)))
    v_error = float(np.max(abs(prior_pre[:, nq:nq + nv] - now_pre[:, nq:nq + nv])))
    ctrl_error = float(np.max(abs(prior_pre[:, nq + nv:nq + nv + 6] -
                                  now_pre[:, nq + nv:nq + nv + 6])))
    length_error = float(np.max(abs(geometry(m, prior_post)[1] - geometry(m, now_post)[1])))
    joint_error = float(np.max(abs(margins(m, prior_post) - margins(m, now_post))))
    contacts = sum(a == b for a, b in zip(reference['contacts'][offset:offset + STEPS],
                                          candidate['contacts'][:STEPS]))
    require(q_error <= 2e-6 and v_error <= 1e-6 and ctrl_error <= 1e-6 and length_error <= 1e-6 and
            joint_error <= 1e-4 and contacts == STEPS,
            f'{label}: current-source 5ms mismatch q={q_error:.9g}, v={v_error:.9g}, ctrl={ctrl_error:.9g}, '
            f'A={length_error:.9g}, joint={joint_error:.9g}, contacts={contacts}/{STEPS}')
    return dict(qpos_max_abs_error=q_error, qvel_max_abs_error=v_error,
                ctrl_max_abs_error_Nm=ctrl_error,
                actual_A_max_abs_error_m=length_error, eight_joint_max_abs_error_rad=joint_error,
                contact_pair_steps_equal=contacts)


def archive_at(pre, step):
    index = int(np.argmin(abs(pre[:, 0] - step * DT)))
    require(abs(float(pre[index, 0] - step * DT)) < 2e-6, f'archive step {step} missing')
    return pre[index]


def actions(m, q, v):
    basis = common_basis(m, q, v)
    scale = np.max(abs(basis), axis=0)
    require(np.isfinite(basis).all() and np.all(scale > 0), 'invalid current common basis')
    eps = .08 / scale  # Only the current state sets action size; no future basis is read.
    coeff = np.zeros((len(ARMS), 3))
    for channel in range(3):
        coeff[1 + 2 * channel, channel] = eps[channel]
        coeff[2 + 2 * channel, channel] = -eps[channel]
    coeff[7] = eps * np.array((1., -1., 1.)) / 3
    coeff[8] = -coeff[7]
    require(np.max(abs(basis @ coeff.T)) <= .08000001, 'initial action exceeds fixed 0.08 Nm trust box')
    return coeff, eps


def capture_world(world, event_id, archive, windows):
    event = archive['events'][event_id]
    scenario = HeightTerrainScenario(**event['scenario'])
    m = model(scenario)
    nq, nv = m.nq, m.nv
    source = windows[f'event_{event_id}_pre']
    event_step = int(round(event['reference_s'] / DT))
    zero_step = event_step - 32  # 16 ms before the first failure.
    zero_coeff = np.zeros((len(ARMS), 3))
    zero, zero_state, _, _ = simulate([scenario], [zero_step], zero_coeff)
    zero_source = archive_at(source, zero_step)
    for arm in range(len(ARMS)):
        archive_difference(zero_state[arm], zero_source, nq, nv, with_warm=True)
        compare_start(zero_state[arm], zero_state[0], nq, nv, f'world{world} zero-run equal arm{arm}')
    qa = np.array([m.joint(name).qposadr[0] for name in ACTIVE])
    va = np.array([m.joint(name).dofadr[0] for name in ACTIVE])
    joint_limits = np.array([m.jnt_range[m.joint(name).id] for name in ACTIVE])
    samples = []
    for lag in (30, 10):  # 15 ms and 5 ms before the first failure.
        step = event_step - lag
        offset = step - zero_step
        prior = zero[0]['pre'][offset - 1]
        current = zero[0]['pre'][offset]
        archived = archive_at(source, step)
        archive_difference(current, archived, nq, nv, with_warm=False)
        coeff, eps = actions(m, current[:nq], current[nq:nq + nv])
        independent_zero, independent_state, _, _ = simulate([scenario], [step], zero_coeff)
        records, start_state, stats, _ = simulate([scenario], [step], coeff)
        warm_zero_error = 0.
        for arm in range(len(ARMS)):
            archive_difference(start_state[arm], archived, nq, nv, with_warm=True)
            compare_start(start_state[arm], start_state[0], nq, nv, f'world{world} {lag} arm{arm} equal')
            compare_start(independent_state[arm], independent_state[0], nq, nv,
                          f'world{world} {lag} independent-zero arm{arm} equal')
            archive_difference(independent_state[arm], archived, nq, nv, with_warm=True)
            warm_zero_error = max(warm_zero_error, compare_start(
                start_state[arm], independent_state[arm], nq, nv,
                f'world{world} {lag} arm{arm} independent-zero')['warmstart'])
            live = records[arm]['pre'][0]
            for name, start, width in (('qpos', 0, nq), ('qvel', nq, nv), ('ctrl', nq + nv, 6)):
                error = float(np.max(abs(live[start:start + width] - current[start:start + width])))
                require(error <= START_LIMITS[name], f'world{world} {lag} arm{arm} zero-run {name} mismatch {error:.9g}')
            require(np.all(records[arm]['pre'][:STEPS, -1] == 1), f'world{world} {lag} arm{arm} inactive in 5ms')
        archive_differences = archive_difference(start_state[0], archived, nq, nv, with_warm=True)
        require(np.max(abs(records[0]['pre'][:STEPS, -1] - 1)) == 0, f'world{world} {lag} baseline inactive')
        base = records[0]
        other = independent_zero[0]
        zero_to_independent = pair_current_5ms(zero[0], other, m, nq, nv,
                                               f'world{world} {lag} zero-to-independent', offset)
        zero_to_action = pair_current_5ms(zero[0], base, m, nq, nv,
                                          f'world{world} {lag} zero-to-action', offset)
        independent_to_action = pair_current_5ms(other, base, m, nq, nv,
                                                 f'world{world} {lag} independent-to-action')
        actual = np.stack([geometry(m, row['post'][:STEPS])[1] for row in records])
        joints = np.stack([margins(m, row['post'][:STEPS]) for row in records])
        sample_actual = np.stack([row['summary'][:STEPS, 1] for row in records])
        require(float(np.max(abs(actual.min(axis=-1) - sample_actual))) < 1e-5,
                f'world{world} {lag} actual A-chain sample mismatch')
        motor_match = 0.
        motor_clip = 0.
        peak_delta = 0.
        for arm, row in enumerate(records):
            for t in range(STEPS):
                pre = row['pre'][t]
                base = pre[nq + nv:nq + nv + 6]
                q = pre[:nq]
                v = pre[nq:nq + nv]
                expected, _, _ = command(m, q, v, base, coeff[arm])
                request = base + common_basis(m, q, v) @ coeff[arm]
                motor_match = max(motor_match, float(np.max(abs(row['applied'][t] - expected))))
                motor_clip = max(motor_clip, float(np.max(abs(request - expected))))
                peak_delta = max(peak_delta, float(np.max(abs(row['applied'][t] - base))))
        require(motor_match <= MOTOR_MATCH_LIMIT, f'world{world} {lag} applied command mismatch {motor_match:.9g}')
        require(motor_clip <= MOTOR_CLIP_LIMIT, f'world{world} {lag} motor clipped in 5ms {motor_clip:.9g}')
        sample = dict(world=world, event_id=event_id, event=event['kind'], lag_ms=lag / 2,
                      start_s=step * DT, q0=current[qa].copy(), v0=current[nq + va].copy(),
                      vprev=prior[nq + va].copy(), active_joint_limits=joint_limits,
                      coeff=coeff, eps=eps,
                      first_v=np.stack([row['pre'][1, nq + va] for row in records]),
                      actual=actual, joints=joints,
                      contact_changed=np.array([sum(a != b for a, b in zip(records[0]['contacts'][:STEPS],
                                                                          row['contacts'][:STEPS]))
                                                for row in records]),
                      first_contact_changed=np.array([records[0]['contacts'][0] != row['contacts'][0]
                                                      for row in records]),
                      max_abs_attitude=np.max(np.stack([row['summary'][:STEPS, 3:6] for row in records]), axis=1),
                      motor_match=motor_match, motor_clip=motor_clip, peak_delta=peak_delta,
                      historical_start_differences=archive_differences,
                      independent_zero_warmstart_start_mismatch=warm_zero_error,
                      zero_to_independent_5ms=zero_to_independent,
                      zero_to_action_5ms=zero_to_action,
                      independent_to_action_5ms=independent_to_action,
                      full_40_step_kernel_stats=stats.copy())
        samples.append(sample)
    return samples


def preflight_world1_late(event, archived):
    """先核曾失败的台阶晚窗，同来源零臂配对才进入六世界试验。"""
    scenario = HeightTerrainScenario(**event['scenario'])
    m = model(scenario)
    nq, nv = m.nq, m.nv
    step = int(round(event['reference_s'] / DT)) - 10
    zero_coeff = np.zeros((len(ARMS), 3))
    zero, zero_state, _, _ = simulate([scenario], [step], zero_coeff)
    q = zero[0]['pre'][0, :nq]
    v = zero[0]['pre'][0, nq:nq + nv]
    coeff, _ = actions(m, q, v)
    branches, branch_state, _, _ = simulate([scenario], [step], coeff)
    proximity = archive_difference(zero_state[0], archive_at(archived, step), nq, nv,
                                   with_warm=True)
    for arm in range(len(ARMS)):
        compare_start(zero_state[arm], zero_state[0], nq, nv, f'world1 late preflight zero arm{arm}')
        compare_start(branch_state[arm], branch_state[0], nq, nv, f'world1 late preflight action arm{arm}')
        compare_start(branch_state[arm], zero_state[arm], nq, nv,
                      f'world1 late preflight independent/action arm{arm}')
    pair = pair_current_5ms(zero[0], branches[0], m, nq, nv, 'world1 late preflight')
    return dict(world=1, event=event['kind'], start_s=step * DT,
                historical_start_differences=proximity, current_source_zero_pair=pair)


def fit_gain(samples, worlds):
    responses = []
    for sample in samples:
        if sample['world'] not in worlds:
            continue
        one = np.stack([(sample['first_v'][1 + 2 * k] - sample['first_v'][2 + 2 * k]) /
                        (2 * DT * sample['eps'][k]) for k in range(3)], axis=1)
        responses.append(one)
    require(len(responses) == 2 * len(worlds), f'gain fit has {len(responses)} instead of {2 * len(worlds)} starts')
    return np.mean(responses, axis=0)


def forecast(sample, gain):
    acceleration = (sample['v0'] - sample['vprev']) / DT + sample['coeff'] @ gain.T
    return forecast_acceleration(sample['q0'], sample['v0'], acceleration, sample['active_joint_limits'])


def forecast_acceleration(q0, v0, acceleration, limits):
    t = np.arange(1, STEPS + 1) * DT
    q = q0[None, None, :] + v0[None, None, :] * t[None, :, None] + \
        .5 * acceleration[:, None, :] * t[None, :, None] ** 2
    leg = np.minimum(lengths(q[..., 0], q[..., 1]), lengths(q[..., 2], q[..., 3]))
    active_margin = np.minimum(q - limits[None, None, :, 0], limits[None, None, :, 1] - q).min(axis=-1)
    return leg, active_margin


def optimistic_error(sample, gain):
    predicted_leg, predicted_joint = forecast(sample, gain)
    true_leg = sample['actual'].min(axis=-1)
    true_joint = sample['joints'].min(axis=-1)
    return float(np.max(predicted_leg - true_leg)), float(np.max(predicted_joint - true_joint))


def score(samples, heldout, gain, reserve):
    rows = []
    for sample in samples:
        if sample['world'] != heldout:
            continue
        predicted_leg, predicted_joint = forecast(sample, gain)
        actual_leg = sample['actual'].min(axis=-1)
        actual_joint = sample['joints'].min(axis=-1)
        for arm, name in enumerate(ARMS):
            leg_margin = float(predicted_leg[arm].min() - reserve[0] - HEIGHT_115_GEOMETRIC_MIN)
            joint_margin = float(predicted_joint[arm].min() - reserve[1])
            true_leg_margin = float(actual_leg[arm].min() - HEIGHT_115_GEOMETRIC_MIN)
            true_joint_margin = float(actual_joint[arm].min())
            leg_safe = leg_margin >= 0
            joint_safe = joint_margin >= 0
            true_leg_safe = true_leg_margin >= 0
            true_joint_safe = true_joint_margin >= 0
            rows.append(dict(world=heldout, event=sample['event'], lag_ms=sample['lag_ms'],
                             start_s=sample['start_s'], action=name,
                             predicted_leg_margin_after_reserve_m=leg_margin,
                             predicted_joint_margin_after_reserve_rad=joint_margin,
                             actual_leg_margin_m=true_leg_margin, actual_eight_joint_margin_rad=true_joint_margin,
                             leg_false_safe=bool(leg_safe and not true_leg_safe),
                             joint_false_safe=bool(joint_safe and not true_joint_safe),
                             combined_false_safe=bool(leg_safe and joint_safe and not (true_leg_safe and true_joint_safe)),
                             predicted_combined_safe=bool(leg_safe and joint_safe),
                             actual_combined_safe=bool(true_leg_safe and true_joint_safe),
                             contact_pair_change_steps=int(sample['contact_changed'][arm]),
                             first_step_contact_changed=bool(sample['first_contact_changed'][arm]),
                             max_abs_roll_pitch_yaw_rad=sample['max_abs_attitude'][arm].tolist()))
    require(len(rows) == 2 * len(ARMS), f'heldout world {heldout} has {len(rows)} score rows')
    return rows


def summarize(rows):
    return dict(total=len(rows), leg_false_safe=sum(r['leg_false_safe'] for r in rows),
                joint_false_safe=sum(r['joint_false_safe'] for r in rows),
                combined_false_safe=sum(r['combined_false_safe'] for r in rows),
                predicted_combined_safe=sum(r['predicted_combined_safe'] for r in rows),
                actual_combined_safe=sum(r['actual_combined_safe'] for r in rows),
                strictly_positive_combined_margin=sum(r['predicted_leg_margin_after_reserve_m'] > 0 and
                                                   r['predicted_joint_margin_after_reserve_rad'] > 0 for r in rows),
                max_contact_pair_change_steps=max(r['contact_pair_change_steps'] for r in rows))


def run(output):
    require(not output.exists(), f'output already exists: {output}')
    source = ROOT / 'wheelleg_warp/results/height_115_local_states_20260928/verification.json'
    archive = json.loads(source.read_text())
    data_path = source.parent / 'windows.npz'
    require(sha(data_path) == archive['windows_sha256'], 'input archive data hash mismatch')
    audit = controller_audit()
    selected = [min((i for i, e in enumerate(archive['events']) if e['world'] == world),
                    key=lambda i: archive['events'][i]['reference_s']) for world in range(6)]
    require([archive['events'][i]['scenario'] for i in selected] ==
            [asdict(x) for x in cases()[:6]], 'selected worlds differ from public six normal scenarios')
    windows = np.load(data_path)
    failed_preflights = []
    for directory in ('height_115_action_predict_loow_20260928',
                      'height_115_action_predict_loow_v2_20260928',
                      'height_115_action_predict_loow_v3_20260928'):
        path = ROOT / 'wheelleg_warp/results' / directory / 'verification.json'
        old = json.loads(path.read_text())
        failed_preflights.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                                      status=old.get('status'), error=old.get('error'),
                                      scientific_loow_result=False))
    preflight = preflight_world1_late(archive['events'][selected[1]],
                                      windows[f'event_{selected[1]}_pre'])
    print('current-source world1 late preflight passed', preflight, flush=True)
    samples = []
    for world, event_id in enumerate(selected):
        samples.extend(capture_world(world, event_id, archive, windows))
        print(f'captured world {world}: {len(samples)}/12 starts', flush=True)
    folds = []
    heldout_rows = []
    for heldout in range(6):
        train_worlds = set(range(6)) - {heldout}
        inner = []
        errors = []
        for inner_holdout in sorted(train_worlds):
            inner_gain = fit_gain(samples, train_worlds - {inner_holdout})
            one = [optimistic_error(s, inner_gain) for s in samples if s['world'] == inner_holdout]
            require(len(one) == 2, 'inner world must have both starts')
            errors.extend(one)
            inner.append(dict(world=inner_holdout, max_optimistic_leg_m=max(x[0] for x in one),
                              max_optimistic_joint_rad=max(x[1] for x in one)))
        reserve = np.maximum(0., np.max(np.asarray(errors), axis=0))
        gain = fit_gain(samples, train_worlds)
        one_rows = score(samples, heldout, gain, reserve)
        folds.append(dict(heldout_world=heldout, fitted_common3_acceleration_gain=gain.tolist(),
                          inner_world_residuals=inner, reserves=dict(actual_A_length_m=float(reserve[0]),
                                                                      eight_joint_margin_rad=float(reserve[1])),
                          summary=summarize(one_rows)))
        heldout_rows.extend(one_rows)
    late_rescue = []
    for world in range(6):
        late = [r for r in heldout_rows if r['world'] == world and r['lag_ms'] == 5.]
        zero = next(r for r in late if r['action'] == 'zero')
        if not zero['actual_combined_safe']:
            late_rescue.extend([r for r in late if r['action'] != 'zero' and r['actual_combined_safe']])
    arrays = dict(q0=np.stack([s['q0'] for s in samples]),
                  v0=np.stack([s['v0'] for s in samples]),
                  vprev=np.stack([s['vprev'] for s in samples]),
                  actions=np.stack([s['coeff'] for s in samples]),
                  first_step_active_v=np.stack([s['first_v'] for s in samples]),
                  actual_A_length_m=np.stack([s['actual'] for s in samples]),
                  actual_eight_joint_margin_rad=np.stack([s['joints'] for s in samples]))
    output.mkdir(parents=True)
    trace = output / 'heldout_inputs.npz'
    np.savez_compressed(trace, **arrays)
    sources = ('wheelleg_warp/probe_height_115_action_predict_loow.py',
               'wheelleg_warp/probe_height_115_live_common_local.py',
               'wheelleg_warp/probe_height_115_contact_action_pair.py',
               'wheelleg_warp/probe_height_115_passive.py',
               'wheelleg_warp/probe_height_115_margin.py',
               'wheelleg_warp/probe_height_115_warp_local_lp.py',
               'wheelleg_warp/native/controller.py', 'wheelleg_warp/native/environment.py',
               'wheelleg_warp/native/terrain.py', 'wheelleg_warp/native/models.py',
               'wheelleg_ppo/tools/state_estimation.py', 'wheelleg_ppo/tools/wheelleg_sim.py',
               'wheelleg_ppo/tools/hardware_profile.py', 'wheelleg_ppo/xml/wheelleg.xml')
    def pair_max(key):
        return dict(qpos=max(s[key]['qpos_max_abs_error'] for s in samples),
                    qvel=max(s[key]['qvel_max_abs_error'] for s in samples),
                    ctrl_Nm=max(s[key]['ctrl_max_abs_error_Nm'] for s in samples),
                    actual_A_m=max(s[key]['actual_A_max_abs_error_m'] for s in samples),
                    eight_joint_rad=max(s[key]['eight_joint_max_abs_error_rad'] for s in samples),
                    contact_pair_steps_equal=sum(s[key]['contact_pair_steps_equal'] for s in samples))
    payload = dict(role='public_115m_live_common3_5ms_world_holdout_action_predictability',
                   status='completed', training=False, final_holdout_opened=False,
                   nominal_height_m=.115, simulated_geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,
                   physical_step_s=DT, horizon_steps=STEPS, arms=ARMS,
                   action_rule='current_start_B(q) peak0.08Nm per axis; mixed +/- (F,-H,W)/3',
                   predictor='current active q/dq, previous live baseline active dq, current issued common3 action; constant 4x3 G',
                   no_contact_or_passive_or_future_state_predictor_input=True,
                   current_simulated_sensors_not_hardware_latency_validated=True,
                   authoritative_control_and_physics='current controller same-shape n9 zero and action graphs',
                   historical_archive_use='public scenarios and approximate first-failure times; start q/v/ctrl/warm differences diagnostics only',
                   historical_preflights=failed_preflights, current_source_world1_late_preflight=preflight,
                   calibration='outer leave one world; five inner leave one world fits on four worlds; max positive inner residual',
                   source_state_verification_sha256=sha(source), source_windows_sha256=archive['windows_sha256'],
                   controller_source_audit=audit,
                   heldout_inputs_sha256=sha(trace), selected_event_ids=selected,
                   max_5ms_motor_command_match_error_Nm=max(s['motor_match'] for s in samples),
                   max_5ms_motor_clip_Nm=max(s['motor_clip'] for s in samples),
                   max_historical_start_differences={key: max(s['historical_start_differences'][key]
                       for s in samples) for key in ('qpos', 'qvel', 'ctrl', 'warmstart')},
                   max_start_independent_zero_warmstart_mismatch=max(
                       s['independent_zero_warmstart_start_mismatch'] for s in samples),
                   zero_to_independent_5ms_max_errors=pair_max('zero_to_independent_5ms'),
                   zero_to_action_5ms_max_errors=pair_max('zero_to_action_5ms'),
                   independent_to_action_5ms_max_errors=pair_max('independent_to_action_5ms'),
                   max_5ms_applied_motor_delta_Nm=max(s['peak_delta'] for s in samples),
                   first_step_contact_changed_branches=sum(int(np.count_nonzero(s['first_contact_changed'])) for s in samples),
                   late_zero_actually_failed_worlds=sum(not next(r for r in heldout_rows
                       if r['world'] == w and r['lag_ms'] == 5. and r['action'] == 'zero')['actual_combined_safe']
                       for w in range(6)),
                   late_actual_only_safe_nonzero_actions=len(late_rescue),
                   late_predicted_and_actual_safe_actions=sum(
                       r['predicted_leg_margin_after_reserve_m'] > 0 and
                       r['predicted_joint_margin_after_reserve_rad'] > 0 for r in late_rescue),
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
            failed = dict(
                role='public_115m_live_common3_5ms_world_holdout_action_predictability',
                status='failed_pre_registered_gate_or_execution', error=f'{type(exc).__name__}: {exc}',
                training=False, final_holdout_opened=False,
                script_sha256=sha(Path(__file__)))
            try:
                failed['controller_source_audit'] = controller_audit()
            except Exception as audit_error:
                failed['controller_audit_error'] = str(audit_error)
            failed['historical_preflights'] = []
            for directory in ('height_115_action_predict_loow_20260928',
                              'height_115_action_predict_loow_v2_20260928',
                              'height_115_action_predict_loow_v3_20260928'):
                path = ROOT / 'wheelleg_warp/results' / directory / 'verification.json'
                if path.exists():
                    old = json.loads(path.read_text())
                    failed['historical_preflights'].append(dict(path=str(path.relative_to(ROOT)),
                        sha256=sha(path), status=old.get('status'), error=old.get('error'),
                        scientific_loow_result=False))
            (target / 'verification.json').write_text(json.dumps(failed, ensure_ascii=False, indent=2) + '\n')
        raise
