"""Three flat 20 ms windows: offline common3 choice, live nominal control and live B(q)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import mujoco
import mujoco_warp as mjw
import numpy as np
from scipy.optimize import linprog
import warp as wp
from native.controller import D, V2, allowed, control, polar_jac
from native.environment import NativeEnv, after, begin, command_step, reduce_contacts
from native.terrain import HEIGHT_115_GEOMETRIC_MIN, HeightTerrainScenario, bank_height_115, model
from probe_height_115_local_lp import EPS, GAMMA, HORIZON, metric
from probe_height_115_passive import JOINTS
from probe_height_115_warp_local_lp import capture_start, sample
from state_estimation import leg_kinematics
from trace_first_divergence_v2 import ordered_contacts

EVENTS = (1, 9, 10)


@wp.kernel
def align_start(start: wp.array[int], q: wp.array2d[float], v: wp.array2d[float],
                warm: wp.array2d[float], sensor: wp.array2d[float],
                controller: wp.array2d[D], state: wp.array2d[D], command: wp.array[D],
                contacts: wp.array2d[int], stopped_q: wp.array2d[float],
                stopped_v: wp.array2d[float], stopped_w: wp.array2d[float],
                active: wp.array[int]):
    w = wp.tid()
    if w == 0 or active[w] == 0 or int(state[w, 0]) != start[w]:
        return
    for j in range(q.shape[1]):
        q[w, j] = q[0, j]
        stopped_q[w, j] = stopped_q[0, j]
    for j in range(v.shape[1]):
        v[w, j] = v[0, j]
        warm[w, j] = warm[0, j]
        stopped_v[w, j] = stopped_v[0, j]
        stopped_w[w, j] = stopped_w[0, j]
    for j in range(sensor.shape[1]):
        sensor[w, j] = sensor[0, j]
    for j in range(controller.shape[1]):
        controller[w, j] = controller[0, j]
    for j in range(state.shape[1]):
        state[w, j] = state[0, j]
    for j in range(contacts.shape[1]):
        contacts[w, j] = contacts[0, j]
    command[w] = command[0]


@wp.kernel
def snapshot_pre(slot: int, q: wp.array2d[float], v: wp.array2d[float],
                 ctrl: wp.array2d[float], active: wp.array[int], out: wp.array3d[D]):
    w = wp.tid()
    nq = q.shape[1]
    nv = v.shape[1]
    for j in range(nq):
        out[slot, w, j] = D(q[w, j])
    for j in range(nv):
        out[slot, w, nq + j] = D(v[w, j])
    for j in range(ctrl.shape[1]):
        out[slot, w, nq + nv + j] = D(ctrl[w, j])
    out[slot, w, nq + nv + ctrl.shape[1]] = D(active[w])


@wp.kernel
def snapshot_ctrl(slot: int, ctrl: wp.array2d[float], out: wp.array3d[D]):
    w = wp.tid()
    for j in range(ctrl.shape[1]):
        out[slot, w, j] = D(ctrl[w, j])


@wp.kernel
def snapshot_post(slot: int, q: wp.array2d[float], out: wp.array3d[D]):
    w = wp.tid()
    for j in range(q.shape[1]):
        out[slot, w, j] = D(q[w, j])


@wp.kernel
def snapshot_previous_velocity(start: wp.array[int], state: wp.array2d[D],
                               v: wp.array2d[float], out: wp.array[D], hit: wp.array[int]):
    if int(state[0, 0]) != start[0] - 1:
        return
    for j in range(v.shape[1]):
        out[j] = D(v[0, j])
    hit[0] = 1


@wp.kernel
def live_common(q: wp.array2d[float], v: wp.array2d[float], ctrl: wp.array2d[float],
                state: wp.array2d[D], active: wp.array[int], ids: wp.array[int],
                start: wp.array[int], coeff: wp.array2d[D], stats: wp.array2d[D]):
    w = wp.tid()
    step = int(state[w, 0])
    if active[w] == 0 or step < start[w] or step >= start[w] + HORIZON:
        return
    stats[w, 0] += D(1)
    for side in range(2):
        a = D(q[w, ids[2 * side]])
        b = D(q[w, ids[2 * side + 1]])
        torque = polar_jac(a, b) * V2(coeff[w, 0], coeff[w, 1])
        for j in range(2):
            motor = 2 * side + j
            speed = D(v[w, ids[4 + motor]])
            delta = torque[j]
            before = D(ctrl[w, motor])
            request = before + delta
            applied = wp.clamp(request, -allowed(speed, True), allowed(speed, True))
            ctrl[w, motor] = float(applied)
            stats[w, 1] += D(wp.abs(request - applied) > D(1.e-8))
            stats[w, 2] = wp.max(stats[w, 2], wp.abs(delta))
            stats[w, 3] = wp.max(stats[w, 3], wp.abs(applied - before))
    for motor in range(4, 6):
        speed = D(v[w, ids[4 + motor]])
        delta = coeff[w, 2]
        before = D(ctrl[w, motor])
        request = before + delta
        applied = wp.clamp(request, -allowed(speed, False), allowed(speed, False))
        ctrl[w, motor] = float(applied)
        stats[w, 1] += D(wp.abs(request - applied) > D(1.e-8))
        stats[w, 2] = wp.max(stats[w, 2], wp.abs(delta))
        stats[w, 3] = wp.max(stats[w, 3], wp.abs(applied - before))


def common_basis(m, q, v):
    basis = np.zeros((6, 3))
    for side in ('L', 'R'):
        joints = [m.joint(name + side).id for name in ('alpha', 'beta')]
        jac = leg_kinematics(q[m.jnt_qposadr[joints]], v[m.jnt_dofadr[joints]])[3]
        motors = [m.actuator('motor_' + name + side).id for name in ('alpha', 'beta')]
        basis[motors, :2] = jac
        basis[m.actuator('motor_wheel' + side).id, 2] = 1.
    return basis


def simulate(base, starts, coefficients):
    assert len(base) == 1
    arms = len(coefficients) // len(base)
    scenarios = base * arms
    n = len(scenarios)
    env = NativeEnv(n=n, scenario=scenarios, bank_factory=bank_height_115,
                    height_conditioned=True, height_design='range115', residual_scale=0)
    env.reset()
    start = wp.array(np.tile(starts, arms), dtype=wp.int32)
    coeff = wp.array(np.asarray(coefficients), dtype=D)
    stats = wp.zeros((n, 4), dtype=D)
    start_state = wp.zeros((n, 1 + env.cpu.nq + 2 * env.cpu.nv + env.cpu.nu), dtype=D)
    previous_qvel = wp.zeros(env.cpu.nv, dtype=D)
    previous_hit = wp.zeros(1, dtype=wp.int32)
    passive = wp.array([env.cpu.joint(name).qposadr[0] for name in
                        ('passA_L', 'passC_L', 'passA_R', 'passC_R')], dtype=wp.int32)
    d = env.data
    pre = wp.zeros((40, n, env.cpu.nq + env.cpu.nv + env.cpu.nu + 1), dtype=D)
    applied = wp.zeros((40, n, 6), dtype=D)
    post = wp.zeros((40, n, env.cpu.nq), dtype=D)
    summary = wp.zeros((40, n, 7), dtype=D)
    pairs = wp.zeros((40, d.naconmax, 3), dtype=wp.int32)
    with wp.ScopedCapture() as captured:
        wp.launch(begin, n, [env.reward])
        for slot in range(40):
            wp.launch(snapshot_previous_velocity, 1, [start, env.state, d.qvel,
                previous_qvel, previous_hit])
            wp.launch(align_start, n, [start, d.qpos, d.qvel, d.qacc_warmstart, d.sensordata,
                env.k['state'], env.state, env.command, env.contact_flags,
                env.stopped_q, env.stopped_v, env.stopped_w, env.active])
            wp.launch(command_step, n, [env.state, env.param, env.command, env.active, d.qpos, d.qvel,
                d.qacc_warmstart, env.stopped_q, env.stopped_v, env.stopped_w, env.contact_flags])
            wp.launch(control, n, [d.qpos, d.qvel, d.sensordata, env.targets, env.command, env.active,
                env.k['state'], env.ids, env.k['heights'], env.k['gains'], env.k['feed'], env.k['angles'],
                env.k['reference'], env.k['yaw'], d.ctrl, env.diag, 0, 0], block_dim=32)
            wp.launch(capture_start, n, [d.time, d.qpos, d.qvel, d.qacc_warmstart, d.ctrl,
                env.state, env.active, start, start_state])
            wp.launch(snapshot_pre, n, [slot, d.qpos, d.qvel, d.ctrl, env.active, pre])
            wp.launch(live_common, n, [d.qpos, d.qvel, d.ctrl, env.state, env.active, env.ids,
                start, coeff, stats])
            wp.launch(snapshot_ctrl, n, [slot, d.ctrl, applied])
            mjw.step(env.model, d)
            wp.launch(sample, n, [slot, d.qpos, env.state, env.active, env.ids,
                passive, env.wheel_offsets, summary])
            wp.launch(snapshot_post, n, [slot, d.qpos, post])
            wp.launch(ordered_contacts, d.naconmax, [slot, d.nacon, d.contact.worldid, d.contact.geom, pairs])
            wp.launch(reduce_contacts, d.naconmax, [d.nacon, d.contact.worldid, d.contact.geom,
                env.ids, env.contact_flags])
            wp.launch(after, n, [d.qpos, d.qvel, d.sensordata, d.qacc_warmstart, d.time,
                env.contact_flags, env.ids, env.param, env.command, env.state, env.k['state'], env.diag,
                env.residual, env.active, env.done, env.reward, env.obs, env.history,
                env.stopped_q, env.stopped_v, env.stopped_w, env.wheel_offsets], block_dim=32)
    env.targets.assign(np.zeros((n, 3), np.float32))
    records = [dict(pre=[], applied=[], post=[], summary=[], contacts=[], contact_raw=[]) for _ in range(n)]
    last_step = int(max(starts)) + HORIZON
    for period in range((last_step + 39) // 40):
        wp.capture_launch(captured.graph)
        global_step = period * 40
        if not any(global_step < s + HORIZON and global_step + 40 > s for s in starts):
            continue
        p0, p1, p2, ps, pc = pre.numpy(), applied.numpy(), post.numpy(), summary.numpy(), pairs.numpy()
        for w in range(n):
            s = starts[w % len(base)]
            slots = np.flatnonzero((np.arange(40) + global_step >= s) &
                                   (np.arange(40) + global_step < s + HORIZON))
            for slot in slots:
                r = records[w]
                r['pre'].append(p0[slot, w].copy())
                r['applied'].append(p1[slot, w].copy())
                r['post'].append(p2[slot, w].copy())
                r['summary'].append(ps[slot, w].copy())
                raw = pc[slot].copy()
                r['contact_raw'].append(raw)
                r['contacts'].append({tuple(sorted((int(a), int(b)))) for world, a, b in raw if int(world) == w})
    assert previous_hit.numpy()[0] == 1
    prior_v = previous_qvel.numpy()
    for r in records:
        for key in ('pre', 'applied', 'post', 'summary', 'contact_raw'):
            r[key] = np.stack(r[key])
        assert len(r['pre']) == HORIZON and len(r['contacts']) == HORIZON
        r['previous_qvel'] = prior_v.copy()
    return records, start_state.numpy(), stats.numpy(), env.done.numpy()


def margins(m, records):
    d = mujoco.MjData(m)
    ids = np.array([m.joint(name).qposadr[0] for name in JOINTS])
    ranges = np.asarray([m.jnt_range[m.joint(name).id] for name in JOINTS])
    values = []
    for q in records['post']:
        d.qpos[:] = q
        values.append(metric(m, d, ids, ranges)[0])
    values = np.asarray(values)
    actual = HEIGHT_115_GEOMETRIC_MIN + .001 * values[:, :2]
    assert np.max(np.abs(np.min(actual, axis=1) - records['summary'][:, 1])) < 1.e-5
    return values


def solve(g, sensitivity, basis, pre, m):
    b = basis.reshape(-1, 3)
    upper = []
    lower = []
    for t in range(HORIZON):
        for aid in range(6):
            dof = m.jnt_dofadr[m.actuator_trnid[aid, 0]]
            speed = pre[t, m.nq + dof]
            rpm = abs(speed) * 60 / (2 * np.pi)
            if aid < 4:
                bound = 40. if rpm <= 175 else 40. * max(0., (280. - rpm) / 105.)
            else:
                bound = 4.5 if rpm <= 490 else 4.5 * max(0., (710. - rpm) / 220.)
            base = pre[t, m.nq + m.nv + aid]
            upper.append(min(EPS, bound - base))
            lower.append(min(EPS, bound + base))
    upper = np.asarray(upper)
    lower = np.asarray(lower)
    assert np.min(upper) >= -1.e-7 and np.min(lower) >= -1.e-7
    s = sensitivity.reshape(-1, 3)
    safe = np.column_stack((-s, np.zeros(len(g))))
    peak_pos = np.column_stack((b, -np.ones(len(b))))
    peak_neg = np.column_stack((-b, -np.ones(len(b))))
    box_pos = np.column_stack((b, np.zeros(len(b))))
    box_neg = np.column_stack((-b, np.zeros(len(b))))
    result = linprog(np.r_[np.zeros(3), 1.],
                     A_ub=np.vstack((safe, peak_pos, peak_neg, box_pos, box_neg)),
                     b_ub=np.r_[g - GAMMA, np.zeros(2 * len(b)), upper, lower],
                     bounds=[(None, None)] * 3 + [(0., EPS)], method='highs')
    return result


def run(output):
    assert not output.exists(), output
    state_path = ROOT / 'wheelleg_warp/results/height_115_local_states_20260928/verification.json'
    old_path = ROOT / 'wheelleg_warp/results/height_115_local_lp_20260928/verification.json'
    state = json.loads(state_path.read_text())
    old = json.loads(old_path.read_text())
    for record in (state, old):
        assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                   for name, digest in record['source_sha256'].items())
    assert hashlib.sha256((state_path.parent / 'windows.npz').read_bytes()).hexdigest() == state['windows_sha256']
    windows = np.load(state_path.parent / 'windows.npz')
    base = []
    starts = []
    reference = []
    eps = np.empty((3, 3))
    cols = state['columns']
    for i, event_id in enumerate(EVENTS):
        event = state['events'][event_id]
        assert event['world'] == old['rows'][i]['world'] and event['kind'] == old['rows'][i]['event']
        scenario = HeightTerrainScenario(**event['scenario'])
        base.append(scenario)
        s = int(round(old['rows'][i]['start_s'] / .0005))
        starts.append(s)
        pre = windows[f'event_{event_id}_pre']
        k = int(np.argmin(abs(pre[:, 0] - s * .0005)))
        assert k + HORIZON <= len(pre) and abs(pre[k, 0] - s * .0005) < 1.e-6
        reference.append(pre[k])
        m = model(scenario)
        basis = np.stack([common_basis(m, row[cols['qpos_start']:cols['qpos_start'] + m.nq],
                         row[cols['qvel_start']:cols['qvel_start'] + m.nv])
                         for row in pre[k:k + HORIZON]])
        eps[i] = .08 / np.max(abs(basis), axis=(0, 1))
    coefficients = np.zeros((7 * 3, 3))
    for i in range(3):
        for channel in range(3):
            coefficients[(1 + 2 * channel) * 3 + i, channel] = eps[i, channel]
            coefficients[(2 + 2 * channel) * 3 + i, channel] = -eps[i, channel]
    runs = [simulate([base[i]], [starts[i]], coefficients[i::3]) for i in range(3)]
    perturbed = [runs[i][0][arm] for arm in range(7) for i in range(3)]
    starts_live = np.stack([runs[i][1][arm] for arm in range(7) for i in range(3)])
    stats = np.stack([runs[i][2][arm] for arm in range(7) for i in range(3)])
    models = [model(scenario) for scenario in base]
    qpos_diffs = []
    for arm in range(7):
        for i in range(3):
            qpos_diffs.append(float(np.max(abs(starts_live[arm * 3 + i, 1:1 + models[i].nq] -
                                            reference[i][cols['qpos_start']:cols['qpos_start'] + models[i].nq]))))
    assert all(np.array_equal(starts_live[arm * 3 + i], starts_live[i])
               for arm in range(7) for i in range(3))
    assert max(qpos_diffs) <= 1.e-6, qpos_diffs
    rows = []
    chosen = np.zeros((6, 3))
    for i, m in enumerate(models):
        baseline = perturbed[i]
        g = margins(m, baseline)
        basis = np.stack([common_basis(m, row[:m.nq], row[m.nq:m.nq + m.nv]) for row in baseline['pre']])
        sensitivity = np.empty((HORIZON, g.shape[1], 3))
        contact_equal = 0
        clipped = 0
        perturb_peak = 0.
        active_perturb = True
        for channel in range(3):
            plus = perturbed[(1 + 2 * channel) * 3 + i]
            minus = perturbed[(2 + 2 * channel) * 3 + i]
            sensitivity[:, :, channel] = (margins(m, plus) - margins(m, minus)) / (2 * eps[i, channel])
            for arm in (plus, minus):
                contact_equal += sum(a == b for a, b in zip(baseline['contacts'], arm['contacts']))
                active_perturb &= bool(np.all(arm['pre'][:, -1] == 1))
            for arm_id in ((1 + 2 * channel) * 3 + i, (2 + 2 * channel) * 3 + i):
                clipped += int(stats[arm_id, 1])
                perturb_peak = max(perturb_peak, float(stats[arm_id, 2]))
        assert np.all(baseline['pre'][:, -1] == 1)
        valid_derivative = bool(active_perturb and clipped == 0 and perturb_peak <= EPS + 1.e-8 and contact_equal == 6 * HORIZON)
        result = solve(g.reshape(-1), sensitivity, basis, baseline['pre'], m) if valid_derivative else None
        candidate = None
        if result is not None and result.success:
            chosen[3 + i] = result.x[:3]
            candidate = dict(common_virtual_action=result.x[:3].tolist(),
                             linear_min_normalized_margin=float(np.min(g.reshape(-1) +
                                 sensitivity.reshape(-1, 3) @ result.x[:3])),
                             minimum_peak_motor_delta_Nm=float(result.x[3]))
        rows.append(dict(event_id=EVENTS[i], world=state['events'][EVENTS[i]]['world'],
                         event=state['events'][EVENTS[i]]['kind'], start_s=starts[i] * .0005,
                         max_start_vs_archive_qpos_m=max(qpos_diffs[i::3]),
                         finite_difference_coefficient_eps=eps[i].tolist(),
                         finite_difference_peak_motor_delta_Nm=perturb_peak,
                         finite_difference_clipped_motor_commands=clipped,
                         finite_difference_contact_pair_steps_equal=contact_equal,
                         finite_difference_all_40_steps_active=active_perturb,
                         finite_difference_valid=valid_derivative,
                         baseline_min_normalized_margin=float(np.min(g)),
                         linear_feasible=bool(result is not None and result.success),
                         linear_status=result.message if result is not None else 'invalid finite difference',
                         candidate=candidate))
    validation = [simulate([base[i]], [starts[i]],
                 np.stack((np.zeros(3), chosen[3 + i], np.zeros(3)))) for i in range(3)]
    tested = [validation[i][0][arm] for arm in range(2) for i in range(3)]
    tested_start = np.stack([validation[i][1][arm] for arm in range(2) for i in range(3)])
    tested_stats = np.stack([validation[i][2][arm] for arm in range(2) for i in range(3)])
    for i, m in enumerate(models):
        candidate = tested[3 + i]
        baseline = tested[i]
        assert np.array_equal(baseline['pre'], perturbed[i]['pre'])
        g = margins(m, candidate)
        r = rows[i]
        actual_motor_delta = np.max(abs(candidate['applied'] - candidate['pre'][:, m.nq + m.nv:m.nq + m.nv + 6]))
        r['nonlinear'] = dict(active_physical_steps=int(np.count_nonzero(candidate['pre'][:, -1])),
            min_normalized_margin=float(np.min(g)),
            min_actual_leg_m=float(np.min(candidate['summary'][:, 1])),
            min_joint_margin_rad=float(np.min(candidate['summary'][:, 2])),
            peak_abs_attitude_rad=np.max(candidate['summary'][:, 3:6], axis=0).tolist(),
            peak_actual_motor_delta_Nm=float(actual_motor_delta),
            clipped_motor_physical_commands=int(tested_stats[3 + i, 1]),
            applied_physical_steps=int(tested_stats[3 + i, 0]),
            contact_pair_steps_equal=sum(a == b for a, b in zip(baseline['contacts'], candidate['contacts'])),
            window_safe=bool(r['candidate'] and np.all(candidate['pre'][:, -1] == 1) and
                np.min(g) >= GAMMA - 1.e-7 and tested_stats[3 + i, 1] == 0 and
                tested_stats[3 + i, 2] <= EPS + 1.e-8))
    summary = dict(states=3, finite_difference_valid=sum(r['finite_difference_valid'] for r in rows),
                   linear_feasible=sum(r['linear_feasible'] for r in rows),
                   nonlinear_safe_same_contact=sum(r['nonlinear']['window_safe'] and
                       r['nonlinear']['contact_pair_steps_equal'] == HORIZON for r in rows),
                   baseline_window_safe=sum(bool(np.min(margins(m, tested[i])) >= 0) for i, m in enumerate(models)),
                   max_start_vs_archive_qpos_m=max(qpos_diffs))
    output.mkdir(parents=True)
    trace = output / 'trace.npz'
    np.savez_compressed(trace,
        perturbed_pre=np.stack([r['pre'] for r in perturbed]),
        perturbed_applied=np.stack([r['applied'] for r in perturbed]),
        perturbed_post=np.stack([r['post'] for r in perturbed]),
        perturbed_sample=np.stack([r['summary'] for r in perturbed]),
        perturbed_contacts=np.stack([r['contact_raw'] for r in perturbed]),
        tested_pre=np.stack([r['pre'] for r in tested]),
        tested_applied=np.stack([r['applied'] for r in tested]),
        tested_post=np.stack([r['post'] for r in tested]),
        tested_sample=np.stack([r['summary'] for r in tested]),
        tested_contacts=np.stack([r['contact_raw'] for r in tested]),
        start_state_perturbed=starts_live, start_state_tested=tested_start)
    names = ('wheelleg_warp/probe_height_115_live_common_local.py',
             'wheelleg_warp/probe_height_115_local_lp.py',
             'wheelleg_warp/probe_height_115_passive.py',
             'wheelleg_warp/probe_height_115_warp_local_lp.py',
             'wheelleg_warp/trace_first_divergence_v2.py',
             'wheelleg_warp/native/controller.py', 'wheelleg_warp/native/environment.py',
             'wheelleg_warp/native/terrain.py', 'wheelleg_warp/native/models.py',
             'wheelleg_ppo/tools/state_estimation.py', 'wheelleg_ppo/tools/wheelleg_sim.py',
             'wheelleg_ppo/tools/hardware_profile.py', 'wheelleg_ppo/xml/wheelleg.xml')
    (output / 'verification.json').write_text(json.dumps(dict(
        role='public_flat_20ms_live_nominal_live_jacobian_common3_offline_lp',
        nominal_target_m=.115, geometric_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,
        horizon_s=.02, finite_difference_motor_trust_Nm=EPS,
        numerical_pairing_margin_normalized=GAMMA,
        coefficients_chosen_offline_from_future_window=True,
        online_implementability_tested=False, full_episode_safety_tested=False,
        default_controller_changed=False, training=False, final_holdout_opened=False,
        summary=summary, rows=rows,
        source_state_windows_sha256=state['windows_sha256'],
        source_old_lp_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}),
        ensure_ascii=False, indent=2, default=lambda value: value.item()) + '\n')
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
