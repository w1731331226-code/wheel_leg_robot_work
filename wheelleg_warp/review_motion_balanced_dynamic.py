"""Independent raw/prefix/geometry review; no controller or trajectory reruns."""
import ast
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wheelleg_ppo/tools'))
import mujoco
import numpy as np
import warp as wp
from native.terrain import HeightTerrainScenario, model
import wheelleg_sim as sim
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/motion_balanced_dynamic_v1'


def rotate(point, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c*point[0]+s*point[2], np.full_like(c, point[1]),
                     -s*point[0]+c*point[2]], axis=-1)


def run():
    assert not (OUT / 'review.json').exists()
    p, c, k, d = [json.loads((OUT / name).read_text()) for name in
                  ('proposal.json', 'completion.json', 'source_contract.json', 'delivery.json')]
    assert d['completion_sha256'] == sha(OUT / 'completion.json')
    assert d['summarizer_sha256'] == sha(ROOT / 'wheelleg_warp/summarize_motion_balanced_dynamic.py')
    assert c['source_contract_sha256'] == sha(OUT / 'source_contract.json')
    assert k['proposal_sha256'] == sha(OUT / 'proposal.json')
    assert c['primary_physical_steps'] == 80000 and c['calibration_physical_steps'] == 800
    assert all(sha(ROOT / n) == h for n, h in k['source_sha256'].items())
    # Confirm the recording fault in the actual CPU array API, not by assertion label.
    wp.init()
    probe = wp.array([1., 2.], device='cpu')
    view = probe.numpy()
    probe.assign(np.array([3., 4.], np.float32))
    np.testing.assert_array_equal(view, [3., 4.])
    for name, function in (('reference_role_probe.py', 'prepare'), ('phase_support_probe.py', 'select_anchor')):
        tree = ast.parse((ROOT / 'wheelleg_warp' / name).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        for n in ast.walk(fn):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AugAssign) else []
            assert all(not ast.unparse(t).startswith('memory[') for t in targets)
    models = [model(HeightTerrainScenario(**r['scenario'])) for r in k['cases']]
    qi, vi, sensors = [], [], []
    for m, r in zip(models, p['moving_references']):
        assert sha(ROOT / r['path']) == r['sha256']
        with np.load(ROOT / r['path']) as z:
            q, v = z['q'].astype(np.float32), z['v'].astype(np.float32)
        fresh = mujoco.MjData(m); fresh.qpos[:], fresh.qvel[:] = q, v
        mujoco.mj_forward(m, fresh)
        qi.append(q); vi.append(v); sensors.append(fresh.sensordata.astype(np.float32))
    qi, vi, sensors = map(np.array, (qi, vi, sensors))
    raw, checked = {}, []
    for artifact in c['completed']:
        path = OUT / artifact['path']
        assert sha(path) == artifact['sha256']
        with np.load(path, allow_pickle=False) as z: x = {n: z[n].copy() for n in z.files}
        backend, arm = artifact['backend'], artifact['arm']
        raw[backend, arm] = x
        if backend == 'CPU':
            # Demonstrate original view overwrite; originals remain unmodified.
            for pre, post in (('pre_q', 'post_q'), ('pre_v', 'post_v'), ('pre_sensor', 'post_sensor')):
                np.testing.assert_array_equal(x[pre], x[post].astype(np.float32))
            np.testing.assert_array_equal(x['memory_before'], x['memory_after'])
            repair = OUT / f'recovered_CPU_{arm}_prefix.npz'
            assert sha(repair) == d['recovered_CPU_prefix_sha256'][repair.name]
            with np.load(repair) as z: recovered = {n: z[n].copy() for n in z.files}
            for pre, post, initial in (('pre_q', 'post_q', qi), ('pre_v', 'post_v', vi),
                                       ('pre_sensor', 'post_sensor', sensors)):
                np.testing.assert_array_equal(recovered[pre][0], initial)
                np.testing.assert_array_equal(recovered[pre][1:], x[post][:-1].astype(np.float32))
            np.testing.assert_array_equal(recovered['memory_before'][1:], x['memory_after'][:-1])
            expected = raw['GPU', arm]['memory_before'][0].copy()
            gyro = int(models[0].sensor_adr[models[0].sensor('body_gyro').id])
            expected[:, 5:7], expected[:, 8] = sensors[:, gyro:gyro+2], sensors[:, gyro+2]
            np.testing.assert_array_equal(recovered['memory_before'][0], expected)
            x.update(recovered)
        np.testing.assert_array_equal(x['pre_q'][0], qi)
        np.testing.assert_array_equal(x['pre_v'][0], vi)
        for pre, post in (('pre_q', 'post_q'), ('pre_v', 'post_v'), ('pre_sensor', 'post_sensor')):
            np.testing.assert_array_equal(x[pre][1:], x[post][:-1].astype(np.float32))
        np.testing.assert_array_equal(x['memory_before'][1:], x['memory_after'][:-1])
        np.testing.assert_array_equal(x['pre_warm'][0], 0.)
        np.testing.assert_array_equal(x['pre_warm'][1:], x['post_warm'][:-1])
        np.testing.assert_array_equal(x['physical_state'][:, :, 37],
                                      np.broadcast_to(np.arange(1, 2001)[:, None], (2000, 10)))
        for i, (m, r) in enumerate(zip(models, p['moving_references'])):
            row = next(t for t in d['records'] if (t['backend'], t['arm'], t['world']) == (backend, arm, i))
            q = x['post_q'][:, i].astype(np.float32).astype(float)
            legs, loops = [], []
            for side in ('L', 'R'):
                adr = [m.jnt_qposadr[m.joint(n+side).id] for n in ('alpha', 'beta', 'passA_', 'passC_')]
                names = ('leg'+side, 'kneeA_'+side, 'wheel'+side, 'leg'+side+'_D', 'kneeB_'+side)
                offsets = np.array([m.body_pos[m.body(n).id] for n in names]+[
                    m.site_pos[m.site('couplerB_'+side+'_end').id]]).astype(np.float32).astype(float)
                a = offsets[0]+rotate(offsets[1], q[:, adr[0]])+rotate(offsets[2], q[:, adr[0]]+q[:, adr[2]])
                b = offsets[3]+rotate(offsets[4], q[:, adr[1]])+rotate(offsets[5], q[:, adr[1]]+q[:, adr[3]])
                mid = (offsets[0]+offsets[3])/2
                legs.extend([np.linalg.norm(a-mid, axis=1), np.linalg.norm(b-mid, axis=1)])
                loops.append(np.linalg.norm(a-b, axis=1))
            length, loop = float(np.min(legs)), float(np.max(loops))
            np.testing.assert_allclose([length, loop], [row['actual_leg_min'], row['loop_error_max']], rtol=0, atol=1e-12)
            names = ('alphaL', 'betaL', 'passA_L', 'passC_L', 'alphaR', 'betaR', 'passA_R', 'passC_R')
            ids = [m.joint(n).id for n in names]
            values = q[:, m.jnt_qposadr[ids]]
            bounds = m.jnt_range[ids].astype(np.float32).astype(float)
            margin = float(np.minimum(values-bounds[:, 0], bounds[:, 1]-values).min())
            active = float(1.4-np.max(abs(q[:, m.jnt_qposadr[[m.joint(n).id for n in ('alphaL','betaL','alphaR','betaR')]]])))
            np.testing.assert_allclose([margin, active], [row['eight_joint_margin'], row['active_joint_margin']], rtol=0, atol=1e-12)
            dofs = [m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
            rpm = abs(x['pre_v'][:, i, dofs].astype(float))*60/(2*np.pi)
            rated = np.array([sim.hw.HIP_RATED_RPM]*4+[sim.hw.MOTOR_RATED_RPM]*2)
            unloaded = np.array([sim.hw.HIP_NO_LOAD_RPM]*4+[sim.hw.MOTOR_NO_LOAD_RPM]*2)
            caps = np.array([sim.hw.HIP_PEAK_TORQUE]*4+[sim.hw.MOTOR_PEAK_TORQUE]*2)
            allowed = caps*np.where(rpm>rated, np.maximum(0., (unloaded-rpm)/(unloaded-rated)), 1.)
            assert sim.hw.IDEAL_POWER_MODEL
            excess = [float(np.maximum(abs(x[n][:, i])-allowed, 0.).max()) for n in ('actual_torque','ctrl')]
            np.testing.assert_allclose(excess, [row['actual_torque_excess'],row['command_torque_excess']], rtol=0, atol=1e-12)
            passed = length>=.1147044660616607 and margin>=0 and active>=0 and max(excess)<=1e-6
            assert row['physical_design_pass'] == passed
            checked.append(dict(backend=backend, arm=arm, world=i, passed=bool(passed),
                                actual_leg_min=length, loop_error_max=loop))
    for r in c['replay']:
        path = OUT / f"replay_{r['arm']}_{r['world']}.npz"
        assert sha(path) == d['replay_array_sha256'][path.name]
        with np.load(path) as z: errors = z['errors']
        assert errors.shape == (40, 3)
        assert r['max_qpos_error'] == errors[:, 0].max() and r['max_qvel_error'] == errors[:, 1].max()
        assert r['passed'] == bool(errors[:, 0].max()<=5e-5 and errors[:, 1].max()<=.02 and errors[:, 2].all())
    atomic_json(OUT / 'review.json', dict(verified=True, round=272, reviewer_sha256=sha(__file__),
        delivery_sha256=sha(OUT / 'delivery.json'), checked=checked,
        physical_design_passed=sum(r['passed'] for r in checked), replay_saved_arrays_passed=20,
        CPU_view_fault_reproduced=True, CPU_prefix_recovery_chain_passed=True,
        independent_geometry_and_torque_recomputed=True, fresh_initial_CPU_forward_calls=10,
        additional_controller_queries=0, integration_steps=0, transitionFD_calls=0, new_training_samples=0,
        production_admitted=False,
        decision='Close fixed-node dynamic qualification. Retain nominal engineering reference only; no more equilibrium trajectories. Assess continuous-command task coverage next, without assuming this small nominal-error reduction yields task or novel RL benefit.',
        limits='Review recomputes saved replay errors, not independent new physical replay. Initial CPU memory recovered from documented shared GPU initializer plus CPU gyro, not an independently captured initial CPU memory. No start/stop/perturbation/full-task/novel-method admission.'))
    print('PASS272 recoveredprefix/independentgeometry/torque40; savedreplay20; no new controller/physics/FD/PPO', flush=True)


if __name__ == '__main__':
    run()
