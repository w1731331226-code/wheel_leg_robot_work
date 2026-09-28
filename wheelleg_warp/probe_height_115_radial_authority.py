"""0.115 m 横坡首次接触：10 ms 内共模径向力的逐步电机盒上界诊断。"""
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
import warp as wp
import wheelleg_sim as sim
from native.terrain import HeightTerrainScenario, bank_height_115, model, HEIGHT_115_GEOMETRIC_MIN
from probe_height_115_contact_action_pair import basis, contact_sets, margins
from probe_height_115_cpu_step_pair import forces_cpu, forces_warp
from probe_height_115_passive import geometry, JOINTS

EVENT, START, STEPS = 4, 85, 20
PAIR_LIMITS = dict(qpos=5e-5, qvel=.02, actual_A_length=2e-6, joint_margin=1e-4)


def torque_box(m, v):
    out = np.empty(6)
    for aid in range(6):
        dof = m.jnt_dofadr[m.actuator_trnid[aid, 0]]
        out[aid] = min(sim.hw.torque_limit(float('inf'), float(v[dof]), aid < 4, 0., .0005)[0],
                       float(m.actuator_ctrlrange[aid, 1]))
    return out


def maximal_outward(m, q, v, base):
    """求交 -A_i <= base_i + J_i F <= A_i，取非负交集的右端点。"""
    radial = basis(m, q, v)[:, 0]
    bound = torque_box(m, v)
    if not (np.isfinite(radial).all() and np.isfinite(bound).all() and np.isfinite(base).all()):
        return None, 'nonfinite_input'
    lower, upper = 0., float('inf')
    for i in range(6):
        j = radial[i]
        if abs(j) < 1e-12:
            if abs(base[i]) > bound[i] + 1e-8:
                return None, 'uncontrolled_motor_outside_box'
            continue
        a, b = sorted(((-bound[i] - base[i]) / j, (bound[i] - base[i]) / j))
        lower, upper = max(lower, a), min(upper, b)
    if not np.isfinite(upper) or upper < lower - 1e-9:
        return None, 'no_nonnegative_common_radial_force'
    requested = base + radial * upper
    if np.max(np.abs(requested) - bound) > 1e-7:
        return None, 'analytic_motor_bound_error'
    return dict(requested_F_N=float(upper), interval_N=[float(lower), float(upper)],
                requested_ctrl_Nm=requested, radial_Nm_per_N=radial, bounds_Nm=bound), None


def pose(q, scenario):
    qw, qx, qy, qz = map(float, q[3:7])
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    terrain_roll = np.deg2rad(scenario.grade_deg) if abs(np.sign(scenario.speed) * q[0] - scenario.center) <= .65 else 0.
    return dict(world_abs_deg=np.rad2deg(np.abs([roll, pitch, yaw])).tolist(),
                terrain_relative_abs_deg=np.rad2deg(np.abs([roll - terrain_roll, pitch, yaw])).tolist())


def measured(m, q, base, requested, applied, bound, requested_F, radial, pairs, scenario):
    actual = geometry(m, q)[1]
    joint = margins(m, q)
    applied_F = float(np.dot(radial[:4], applied[:4] - base[:4]) / np.dot(radial[:4], radial[:4])) if requested_F else 0.
    wheel_ids = {m.geom('wheel_collide_' + side).id for side in ('L', 'R')}
    return dict(actual_A_length_m=actual.tolist(), min_actual_A_length_m=float(actual.min()),
                joint_margin_rad=dict(zip(JOINTS, map(float, joint))), min_eight_joint_margin_rad=float(joint.min()),
                attitude=pose(q, scenario), contact_pairs=[list(p) for p in sorted(pairs)],
                nonwheel_contact_pairs=[list(p) for p in sorted(p for p in pairs if not wheel_ids.intersection(p))],
                requested_F_N=float(requested_F), applied_F_N=applied_F,
                requested_ctrl_Nm=requested.tolist(), applied_ctrl_Nm=applied.tolist(),
                motor_bounds_at_command_Nm=bound.tolist(),
                max_motor_bound_excess_Nm=float(max(0., np.max(abs(applied) - bound))))


def pair(cpu, warp):
    d = dict(qpos=float(np.max(abs(cpu['q'] - warp['q']))), qvel=float(np.max(abs(cpu['v'] - warp['v']))),
             actual_A_length=float(np.max(abs(np.asarray(cpu['length']) - warp['length']))),
             joint_margin=float(np.max(abs(np.asarray(cpu['joint']) - warp['joint']))),
             contact_pairs_equal=cpu['pairs'] == warp['pairs'])
    d['passed'] = bool(d['contact_pairs_equal'] and all(d[k] <= PAIR_LIMITS[k] for k in PAIR_LIMITS))
    return d


def run(output):
    assert not output.exists(), output
    source_dir = ROOT / 'wheelleg_warp/results/height_115_local_states_20260928'
    source = source_dir / 'verification.json'
    record = json.loads(source.read_text())
    archive = source_dir / 'windows.npz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == record['windows_sha256']
    event = record['events'][EVENT]
    assert event['world'] == 2 and event['kind'] == 'actual_leg' and event['scenario']['terrain'] == 'cross_slope'
    arrays = np.load(archive)
    pre, post = arrays[f'event_{EVENT}_pre'], arrays[f'event_{EVENT}_post']
    old_contacts = arrays[f'event_{EVENT}_contacts']
    assert START + STEPS <= len(pre)
    scenario = HeightTerrainScenario(**event['scenario'])
    m = model(scenario)
    cols = record['columns']
    first = pre[START]
    q0 = first[cols['qpos_start']:cols['qpos_start'] + m.nq]
    v0 = first[cols['qvel_start']:cols['qvel_start'] + m.nv]
    warm0 = first[cols['warmstart_start']:cols['warmstart_start'] + m.nv]
    cpu = [mujoco.MjData(m) for _ in range(2)]
    for d in cpu:
        d.time = float(first[0]); d.qpos[:] = q0; d.qvel[:] = v0; d.qacc_warmstart[:] = warm0
    wp.init(); wp.set_device('cuda:0')
    _, gm, gd, _ = bank_height_115(2, scenario=[scenario] * 2)
    gd.time.assign(np.full(2, first[0], np.float32))
    gd.qpos.assign(np.tile(q0.astype(np.float32), (2, 1)))
    gd.qvel.assign(np.tile(v0.astype(np.float32), (2, 1)))
    gd.qacc_warmstart.assign(np.tile(warm0.astype(np.float32), (2, 1)))
    initial = dict(time_s=float(first[0]), baseline_ctrl_Nm=pre[START, cols['ctrl_start']:cols['ctrl_start'] + 6].tolist(),
                   actual_A_length_m=geometry(m, q0)[1].tolist(),
                   joint_margin_rad=dict(zip(JOINTS, map(float, margins(m, q0)))), attitude=pose(q0, scenario))
    rows = []
    status = 'completed_10ms'
    failure = None
    for step in range(STEPS):
        base = pre[START + step, cols['ctrl_start']:cols['ctrl_start'] + 6]
        qs = [np.asarray([d.qpos for d in cpu]), gd.qpos.numpy()]
        vs = [np.asarray([d.qvel for d in cpu]), gd.qvel.numpy()]
        commands, plans = [], []
        for backend, (q, v) in enumerate(zip(qs, vs)):
            plan, error = maximal_outward(m, q[1], v[1], base)
            if error:
                status = error
                failure = dict(step=step, backend=('cpu', 'warp')[backend], reason=error)
                break
            commands.append(np.stack((base, plan['requested_ctrl_Nm'])))
            plans.append(plan)
        if status != 'completed_10ms':
            break
        cpu_commands, warp_commands = commands
        for arm, d in enumerate(cpu):
            d.ctrl[:] = cpu_commands[arm]
            mujoco.mj_step(m, d)
        gd.ctrl.assign(np.asarray(warp_commands, np.float32))
        mjw.step(gm, gd)
        qg, vg = gd.qpos.numpy(), gd.qvel.numpy()
        gpu_pairs = contact_sets(gd, 2)
        one = dict(step=step, time_end_s=float(first[0] + (step + 1) * .0005),
                   legal_F_interval_cpu_N=plans[0]['interval_N'], legal_F_interval_warp_N=plans[1]['interval_N'],
                   arms={})
        for arm, name in enumerate(('baseline', 'max_outward_F')):
            cpairs = set(forces_cpu(m, cpu[arm]))
            wpairs = gpu_pairs[arm]
            radial_c = basis(m, qs[0][arm], vs[0][arm])[:, 0]
            radial_g = basis(m, qs[1][arm], vs[1][arm])[:, 0]
            requested_c = plans[0]['requested_F_N'] if arm else 0.
            requested_g = plans[1]['requested_F_N'] if arm else 0.
            c = measured(m, cpu[arm].qpos, base, cpu_commands[arm], cpu_commands[arm],
                         torque_box(m, vs[0][arm]), requested_c, radial_c, cpairs, scenario)
            g = measured(m, qg[arm], base, warp_commands[arm], np.asarray(warp_commands[arm], np.float32),
                         torque_box(m, vs[1][arm]), requested_g, radial_g, wpairs, scenario)
            cp = dict(q=cpu[arm].qpos.copy(), v=cpu[arm].qvel.copy(), length=c['actual_A_length_m'],
                      joint=list(c['joint_margin_rad'].values()), pairs=cpairs)
            gp = dict(q=qg[arm], v=vg[arm], length=g['actual_A_length_m'],
                      joint=list(g['joint_margin_rad'].values()), pairs=wpairs)
            one['arms'][name] = dict(cpu=c, warp=g, cpu_warp_pair=pair(cp, gp))
        for backend in ('cpu', 'warp'):
            base_pairs = {tuple(p) for p in one['arms']['baseline'][backend]['contact_pairs']}
            candidate_pairs = {tuple(p) for p in one['arms']['max_outward_F'][backend]['contact_pairs']}
            one[f'{backend}_candidate_contact_changed_vs_baseline'] = candidate_pairs != base_pairs
            one[f'{backend}_new_candidate_contacts'] = [list(p) for p in sorted(candidate_pairs - base_pairs)]
        source_q = post[START + step, cols['qpos_start']:cols['qpos_start'] + m.nq]
        source_v = post[START + step, cols['qvel_start']:cols['qvel_start'] + m.nv]
        one['baseline_vs_archive'] = dict(warp_qpos_max_abs_diff=float(np.max(abs(qg[0] - source_q))),
                                          warp_qvel_max_abs_diff=float(np.max(abs(vg[0] - source_v))),
                                          warp_contacts_equal=set(forces_warp(old_contacts[START + step], 2)) == gpu_pairs[0])
        rows.append(one)
        if not all(one['arms'][name]['cpu_warp_pair']['passed'] for name in one['arms']):
            status = 'cpu_warp_pair_invalid'
            failure = dict(step=step, reason=status,
                           pairs={name: one['arms'][name]['cpu_warp_pair'] for name in one['arms']})
            break
    values = [a[backend] for row in rows for a in row['arms'].values() for backend in ('cpu', 'warp')]
    candidate = [row['arms']['max_outward_F']['warp'] for row in rows]
    summary = dict(status=status, failure=failure, completed_steps=len(rows), planned_steps=STEPS,
                   min_candidate_actual_A_length_m=min((r['min_actual_A_length_m'] for r in candidate), default=None),
                   min_candidate_eight_joint_margin_rad=min((r['min_eight_joint_margin_rad'] for r in candidate), default=None),
                   max_motor_bound_excess_Nm=max((r['max_motor_bound_excess_Nm'] for r in values), default=None),
                   contact_change_steps=sum(r['warp_candidate_contact_changed_vs_baseline'] for r in rows),
                   candidate_10ms_gate=bool(status == 'completed_10ms' and candidate and
                       all(r['min_actual_A_length_m'] >= HEIGHT_115_GEOMETRIC_MIN and
                           r['min_eight_joint_margin_rad'] >= 0 and
                           max(r['attitude']['terrain_relative_abs_deg']) <= 5 and
                           max(r['attitude']['world_abs_deg'][:2]) <= 10 and
                           not r['nonwheel_contact_pairs'] and
                           r['max_motor_bound_excess_Nm'] <= 1e-5 for r in candidate)))
    sources = ('wheelleg_warp/probe_height_115_radial_authority.py',
               'wheelleg_warp/probe_height_115_contact_action_pair.py',
               'wheelleg_warp/probe_height_115_cpu_step_pair.py',
               'wheelleg_warp/probe_height_115_passive.py',
               'wheelleg_warp/native/terrain.py', 'wheelleg_warp/native/models.py',
               'wheelleg_ppo/tools/state_estimation.py', 'wheelleg_ppo/tools/wheelleg_sim.py',
               'wheelleg_ppo/tools/hardware_profile.py', 'wheelleg_ppo/xml/wheelleg.xml')
    output.mkdir(parents=True)
    (output / 'verification.json').write_text(json.dumps(dict(
        role='public_cross_slope_maximum_legal_common_radial_force_10ms_authority_diagnostic',
        training=False, final_holdout_opened=False, online_controller_evaluated=False,
        initial=initial, proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,
        protocol='event4 pre[85], 20 x 0.5 ms, archived base ctrl(t), H=W=0, each arm current q/dq analytic Fmax',
        pair_limits=PAIR_LIMITS, summary=summary, rows=rows,
        input_verification_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        input_windows_sha256=record['windows_sha256'],
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}),
        ensure_ascii=False, indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
