"""横坡首次接触附近，共模小动作的 CPU/Warp 5/10 ms 冻结命令配对。"""
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
from probe_height_115_cpu_step_pair import forces_cpu, forces_warp
from probe_height_115_passive import geometry, JOINTS
from state_estimation import leg_kinematics

EVENT = 4  # world 2: 3°横坡，首次真实 A 链轮心越代理线。
STARTS = (85, 91, 99)  # 首次新接触、+3 ms、越线前一步。
NAMES = ('base', 'F+', 'F-', 'H+', 'H-', 'W+', 'W-')
STEPS = 20


def basis(m, q, v):
    out = np.zeros((6, 3))
    for side in ('L', 'R'):
        joints = [m.joint(name + side).id for name in ('alpha', 'beta')]
        jac = leg_kinematics(q[m.jnt_qposadr[joints]], v[m.jnt_dofadr[joints]])[3]
        motors = [m.actuator('motor_' + name + side).id for name in ('alpha', 'beta')]
        out[motors, :2] = jac
        out[m.actuator('motor_wheel' + side).id, 2] = 1.
    return out


def command(m, q, v, base, action):
    delta = basis(m, q, v) @ action
    request = base + delta
    out = np.empty(6)
    for aid in range(6):
        dof = m.jnt_dofadr[m.actuator_trnid[aid, 0]]
        limit, _ = sim.hw.torque_limit(float('inf'), float(v[dof]), aid < 4, 0., .0005)
        limit = min(limit, float(m.actuator_ctrlrange[aid, 1]))
        out[aid] = np.clip(request[aid], -limit, limit)
    return out, int(np.count_nonzero(abs(out - request) > 1e-8)), float(max(abs(delta)))


def contact_sets(data, n):
    result = [set() for _ in range(n)]
    world = data.contact.worldid.numpy()
    geom = data.contact.geom.numpy()
    for i in range(int(data.nacon.numpy()[0])):
        w = int(world[i])
        result[w].add(tuple(sorted((int(geom[i, 0]), int(geom[i, 1])))))
    return result


def margins(m, q):
    ids = np.array([m.joint(name).qposadr[0] for name in JOINTS])
    limits = np.array([m.jnt_range[m.joint(name).id] for name in JOINTS])
    angles = q[..., ids]
    return np.minimum(angles - limits[:, 0], limits[:, 1] - angles)


def compare(cpu, gpu, steps, state):
    s = slice(state * 7, (state + 1) * 7)
    qerr = float(np.max(abs(cpu['q'][:steps, s] - gpu['q'][:steps, s])))
    verr = float(np.max(abs(cpu['v'][:steps, s] - gpu['v'][:steps, s])))
    lerr = float(np.max(abs(cpu['length'][:steps, s] - gpu['length'][:steps, s])))
    jerr = float(np.max(abs(cpu['joint'][:steps, s] - gpu['joint'][:steps, s])))
    same = sum(cpu['contacts'][t][w] == gpu['contacts'][t][w]
               for t in range(steps) for w in range(s.start, s.stop))
    return dict(physical_steps=steps, paired_contact_steps=same, total_contact_steps=steps * 7,
                max_qpos_abs_diff=qerr, max_qvel_abs_diff=verr,
                max_actual_A_length_abs_diff_m=lerr, max_eight_joint_margin_abs_diff_rad=jerr,
                paired=bool(same == steps * 7 and qerr <= 5e-5 and verr <= .02 and
                            lerr <= 2e-6 and jerr <= 1e-4))


def run(output):
    assert not output.exists(), output
    folder = ROOT / 'wheelleg_warp/results/height_115_local_states_20260928'
    source = folder / 'verification.json'
    record = json.loads(source.read_text())
    archive = folder / 'windows.npz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == record['windows_sha256']
    event = record['events'][EVENT]
    assert event['world'] == 2 and event['kind'] == 'actual_leg' and event['scenario']['terrain'] == 'cross_slope'
    arrays = np.load(archive)
    pre = arrays[f'event_{EVENT}_pre']
    post = arrays[f'event_{EVENT}_post']
    old_contacts = arrays[f'event_{EVENT}_contacts']
    cols = record['columns']
    assert max(STARTS) + STEPS <= len(pre)
    scenario = HeightTerrainScenario(**event['scenario'])
    m = model(scenario)
    actions, starts, bases, archived = [], [], [], []
    for start in STARTS:
        row = pre[start]
        q = row[cols['qpos_start']:cols['qpos_start'] + m.nq]
        v = row[cols['qvel_start']:cols['qvel_start'] + m.nv]
        b = basis(m, q, v)
        action = [np.zeros(3)]
        for k in range(3):
            one = np.eye(3)[k] * (.1 / max(abs(b[:, k])))
            action.extend((one, -one))
        actions.extend(action)
        starts.extend([row] * 7)
        bases.extend([pre[start:start + STEPS, cols['ctrl_start']:cols['ctrl_start'] + 6]] * 7)
        archived.append(dict(start_index=start, start_s=float(row[0]),
                             common_actions=[a.tolist() for a in action]))
    actions = np.asarray(actions)
    bases = np.asarray(bases)
    n = len(actions)
    cpu_data = []
    for row in starts:
        d = mujoco.MjData(m)
        d.time = float(row[0])
        d.qpos[:] = row[cols['qpos_start']:cols['qpos_start'] + m.nq]
        d.qvel[:] = row[cols['qvel_start']:cols['qvel_start'] + m.nv]
        d.qacc_warmstart[:] = row[cols['warmstart_start']:cols['warmstart_start'] + m.nv]
        cpu_data.append(d)
    wp.init()
    wp.set_device('cuda:0')
    _, warp_model, warp_data, _ = bank_height_115(n, scenario=[scenario] * n)
    warp_data.time.assign(np.asarray([row[0] for row in starts], np.float32))
    warp_data.qpos.assign(np.asarray([d.qpos for d in cpu_data], np.float32))
    warp_data.qvel.assign(np.asarray([d.qvel for d in cpu_data], np.float32))
    warp_data.qacc_warmstart.assign(np.asarray([d.qacc_warmstart for d in cpu_data], np.float32))
    traces = {side: {name: [] for name in ('q', 'v', 'length', 'joint', 'contacts', 'clipped', 'peak_delta')}
              for side in ('cpu', 'warp')}
    for step in range(STEPS):
        for w, d in enumerate(cpu_data):
            d.ctrl[:], clipped, peak = command(m, d.qpos, d.qvel, bases[w, step], actions[w])
            mujoco.mj_step(m, d)
            traces['cpu']['clipped'].append((step, w, clipped))
            traces['cpu']['peak_delta'].append(peak)
        q = warp_data.qpos.numpy()
        v = warp_data.qvel.numpy()
        controls = []
        for w in range(n):
            control, clipped, peak = command(m, q[w], v[w], bases[w, step], actions[w])
            controls.append(control)
            traces['warp']['clipped'].append((step, w, clipped))
            traces['warp']['peak_delta'].append(peak)
        warp_data.ctrl.assign(np.asarray(controls, np.float32))
        mjw.step(warp_model, warp_data)
        for side, q_now, v_now, pairs in (
            ('cpu', np.asarray([d.qpos for d in cpu_data]), np.asarray([d.qvel for d in cpu_data]),
             [set(forces_cpu(m, d)) for d in cpu_data]),
            ('warp', warp_data.qpos.numpy(), warp_data.qvel.numpy(), contact_sets(warp_data, n))):
            out = traces[side]
            out['q'].append(q_now.copy())
            out['v'].append(v_now.copy())
            out['length'].append(geometry(m, q_now)[1])
            out['joint'].append(margins(m, q_now))
            out['contacts'].append(pairs)
    for out in traces.values():
        for name in ('q', 'v', 'length', 'joint'):
            out[name] = np.asarray(out[name])
    rows = []
    for state, start in enumerate(STARTS):
        w = state * 7
        source_q = post[start, cols['qpos_start']:cols['qpos_start'] + m.nq]
        source_v = post[start, cols['qvel_start']:cols['qvel_start'] + m.nv]
        source_length = geometry(m, source_q)[1]
        source_joint = margins(m, source_q)
        source_pairs = set(forces_warp(old_contacts[start], 2))
        one = dict(source_contact_pairs_equal=bool(source_pairs == traces['cpu']['contacts'][0][w] == traces['warp']['contacts'][0][w]),
                   source_cpu_qpos_abs_diff=float(max(abs(traces['cpu']['q'][0, w] - source_q))),
                   source_cpu_qvel_abs_diff=float(max(abs(traces['cpu']['v'][0, w] - source_v))),
                   source_warp_qpos_abs_diff=float(max(abs(traces['warp']['q'][0, w] - source_q))),
                   source_warp_qvel_abs_diff=float(max(abs(traces['warp']['v'][0, w] - source_v))),
                   source_cpu_actual_A_length_abs_diff_m=float(max(abs(traces['cpu']['length'][0, w] - source_length))),
                   source_warp_actual_A_length_abs_diff_m=float(max(abs(traces['warp']['length'][0, w] - source_length))),
                   source_cpu_eight_joint_margin_abs_diff_rad=float(max(abs(traces['cpu']['joint'][0, w] - source_joint))),
                   source_warp_eight_joint_margin_abs_diff_rad=float(max(abs(traces['warp']['joint'][0, w] - source_joint))))
        one['paired'] = bool(one['source_contact_pairs_equal'] and
                             max(one['source_cpu_qpos_abs_diff'], one['source_warp_qpos_abs_diff']) <= 2e-6 and
                             max(one['source_cpu_qvel_abs_diff'], one['source_warp_qvel_abs_diff']) <= .001)
        checkpoints = []
        for steps in (10, 20):
            paired = compare(traces['cpu'], traces['warp'], steps, state)
            terminal = steps - 1
            responses = []
            for k, label in enumerate(('F', 'H', 'W')):
                pos, neg = w + 1 + 2 * k, w + 2 + 2 * k
                c = traces['cpu']['v'][terminal]
                g = traces['warp']['v'][terminal]
                cp, cm = c[pos] - c[w], c[w] - c[neg]
                gp, gm = g[pos] - g[w], g[w] - g[neg]
                cavg, gavg = (cp + cm) / 2, (gp + gm) / 2
                responses.append(dict(channel=label,
                    cpu_qvel_plus_minus_asymmetry=float(np.linalg.norm(cp - cm) / max(np.linalg.norm(cavg), 1e-12)),
                    warp_qvel_plus_minus_asymmetry=float(np.linalg.norm(gp - gm) / max(np.linalg.norm(gavg), 1e-12)),
                    cpu_warp_qvel_center_response_relative_l2=float(np.linalg.norm(cavg - gavg) / max(np.linalg.norm(cavg), 1e-12)),
                    cpu_warp_qvel_positive_response_relative_l2=float(np.linalg.norm(cp - gp) / max(np.linalg.norm(cp), 1e-12)),
                    cpu_warp_qvel_negative_response_relative_l2=float(np.linalg.norm(cm - gm) / max(np.linalg.norm(cm), 1e-12))))
            paired.update(horizon_ms=steps / 2,
                cpu_base_min_actual_A_length_m=float(np.min(traces['cpu']['length'][:steps, w])),
                warp_base_min_actual_A_length_m=float(np.min(traces['warp']['length'][:steps, w])),
                cpu_base_min_eight_joint_margin_rad=float(np.min(traces['cpu']['joint'][:steps, w])),
                warp_base_min_eight_joint_margin_rad=float(np.min(traces['warp']['joint'][:steps, w])),
                cpu_arms_min_actual_A_length_m=np.min(traces['cpu']['length'][:steps, w:w + 7], axis=(0, 2)).tolist(),
                warp_arms_min_actual_A_length_m=np.min(traces['warp']['length'][:steps, w:w + 7], axis=(0, 2)).tolist(),
                cpu_arms_min_eight_joint_margin_rad=np.min(traces['cpu']['joint'][:steps, w:w + 7], axis=(0, 2)).tolist(),
                warp_arms_min_eight_joint_margin_rad=np.min(traces['warp']['joint'][:steps, w:w + 7], axis=(0, 2)).tolist(),
                cpu_warp_baseline_contacts_equal_to_archive=sum(
                    traces['cpu']['contacts'][t][w] == set(forces_warp(old_contacts[start + t], 2)) and
                    traces['warp']['contacts'][t][w] == set(forces_warp(old_contacts[start + t], 2))
                    for t in range(steps)),
                cpu_perturbed_contact_steps_equal_to_base=sum(
                    traces['cpu']['contacts'][t][w + arm] == traces['cpu']['contacts'][t][w]
                    for t in range(steps) for arm in range(1, 7)),
                cpu_clipped_motor_commands=sum(count for t, arm, count in traces['cpu']['clipped']
                                               if t < steps and w <= arm < w + 7),
                warp_clipped_motor_commands=sum(count for t, arm, count in traces['warp']['clipped']
                                                if t < steps and w <= arm < w + 7),
                max_requested_motor_delta_Nm=float(max(traces['cpu']['peak_delta'][t * n + arm]
                    for t in range(steps) for arm in range(w, w + 7))), responses=responses)
            checkpoints.append(paired)
        rows.append(dict(**archived[state], source_one_step=one, checkpoints=checkpoints))
    summary = dict(source_one_step_paired=sum(row['source_one_step']['paired'] for row in rows),
                   five_ms_paired=sum(row['checkpoints'][0]['paired'] for row in rows),
                   ten_ms_paired=sum(row['checkpoints'][1]['paired'] for row in rows),
                   any_clipping=any(c['cpu_clipped_motor_commands'] or c['warp_clipped_motor_commands']
                                    for row in rows for c in row['checkpoints']),
                   any_actual_A_below_proxy=any(min(c['cpu_arms_min_actual_A_length_m']) < HEIGHT_115_GEOMETRIC_MIN
                                                for row in rows for c in row['checkpoints']))
    output.mkdir(parents=True)
    sources = ('wheelleg_warp/probe_height_115_contact_action_pair.py',
               'wheelleg_warp/probe_height_115_local_states.py',
               'wheelleg_warp/probe_height_115_passive.py',
               'wheelleg_warp/probe_height_115_cpu_step_pair.py',
               'wheelleg_warp/native/terrain.py', 'wheelleg_warp/native/models.py',
               'wheelleg_ppo/tools/state_estimation.py',
               'wheelleg_ppo/tools/wheelleg_sim.py', 'wheelleg_ppo/tools/hardware_profile.py',
               'wheelleg_ppo/xml/wheelleg.xml')
    (output / 'verification.json').write_text(json.dumps(dict(
        role='public_cross_slope_common_action_cpu_warp_5_10ms_pair', training=False,
        final_holdout_opened=False, experiment_height_m=.115,
        simulated_proxy_min_m=HEIGHT_115_GEOMETRIC_MIN,
        base_control_sequence='archived_ctrl_frozen_by_step',
        common_basis='recomputed_from_each_arm_current_active_q_each_step',
        action_scale='each_start_channel_normalized_to_0.1_Nm_initial_peak_motor_delta',
        motor_envelope='speed_dependent_hardware_profile_torque_limit',
        no_online_controller_or_policy_evaluated=True,
        one_step_qpos_tolerance_m=2e-6, one_step_qvel_tolerance=1e-3,
        horizon_qpos_tolerance_m=5e-5, horizon_qvel_tolerance=.02,
        horizon_actual_length_tolerance_m=2e-6, horizon_joint_margin_tolerance_rad=1e-4,
        summary=summary, rows=rows,
        input_verification_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        input_windows_sha256=record['windows_sha256'],
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}),
        ensure_ascii=False, indent=2) + '\n')
    print(summary)
    assert summary['source_one_step_paired'] == 3, '源状态一步配对失败，不能解释后续敏感度'
    assert summary['five_ms_paired'] == 3 and summary['ten_ms_paired'] == 3, '短窗接触/数值配对失败，停止敏感度结论'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
