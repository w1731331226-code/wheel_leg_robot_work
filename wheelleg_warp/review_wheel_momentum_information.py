"""Independent saved-data review; no simulation or estimator calls."""
import json
import sys
from pathlib import Path
import numpy as np
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

sys.path.insert(0, str(ROOT / 'wheelleg_ppo/tools'))
import hardware_profile as hw
import model_lqr as ml
import wheelleg_sim as sim
from state_estimation import leg_kinematics

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/wheel_momentum_information_v1'
NAMES = ('relative_proxy_Nm', 'carrier_proxy_Nm', 'corrected_proxy_Nm')


def run():
    destination = OUT / 'review.json'
    assert not destination.exists(), 'Preserve the existing independent review'
    p = json.loads((OUT / 'proposal.json').read_text())
    a = json.loads((OUT / 'source_admission.json').read_text())
    c = json.loads((OUT / 'audit_completion.json').read_text())
    assert a['verified'] and c['verified'] and p['episodes'] == c['episodes'] == 80
    assert a['proposal_sha256'] == sha(OUT / 'proposal.json')
    assert c['source_admission_sha256'] == sha(OUT / 'source_admission.json')
    assert c['auditor_sha256'] == sha(ROOT / 'wheelleg_warp/audit_wheel_momentum_information.py')
    sources = {**p['source_sha256'], **a['source_sha256']}
    assert all(sha(ROOT / name) == digest for name, digest in sources.items())
    assert not a['exact_synchronous_absolute_momentum_source_qualified']
    m, _ = sim.load_model(ml.XML, True)
    J = hw.MOTOR_INERTIA + hw.TIRE_MASS * hw.WHEEL_RADIUS**2 + .5 * hw.HUB_MASS * hw.WHEEL_HUB_RADIUS**2
    scales = np.array([hw.HIP_PEAK_TORQUE] * 4 + [hw.MOTOR_PEAK_TORQUE] * 2)
    qids = [m.jnt_qposadr[m.joint(n).id] for n in ('alphaL', 'betaL', 'alphaR', 'betaR')]
    vids = [m.jnt_dofadr[m.joint(n).id] for n in ('alphaL', 'betaL', 'alphaR', 'betaR')]
    wheels = np.array([m.jnt_dofadr[m.joint(n).id] for n in ('wheel1', 'wheel2')])
    parents = [np.array([m.jnt_dofadr[m.joint(n + side).id] for n in ('alpha', 'passA_')]) for side in ('L', 'R')]
    np.testing.assert_array_equal(m.dof_damping[wheels], .005)
    records = {(r['arm'], r['seed']): r for r in c['records']}
    assert len(records) == 80
    errors = {arm: [] for arm in p['arms']}
    covers = {arm: [] for arm in p['arms']}
    improved = dict.fromkeys(p['arms'], 0)
    seen = set()
    gyro_pre_max = gyro_post_max = 0.
    for entry in p['input_results']:
        file = ROOT / entry['path']
        assert sha(file) == entry['sha256']
        rows = json.loads(file.read_text())['runs']
        assert len(rows) == 20
        for row in rows:
            key = (entry['arm'], row['seed'])
            assert key not in seen and row['scenario']['delay_ms'] == 0.
            seen.add(key)
            r = records[key]
            actor, dense = [file.parent / row[name]['path'] for name in ('actor_trace', 'complete_trace')]
            saved = OUT / r['path']
            assert sha(saved) == r['sha256']
            assert sha(actor) == r['actor_sha256'] == row['actor_trace']['sha256']
            assert sha(dense) == r['dense_sha256'] == row['complete_trace']['sha256']
            with np.load(actor, allow_pickle=False) as z:
                trace = z['trace']
                assert trace.ndim == 2 and trace.shape[1] == 968 and np.isfinite(trace).all()
                dt = trace[:, 470].astype(float) * .02
                k = np.flatnonzero((trace[:, 479:481] == 1).all(axis=1) & (dt > 0))
                np.testing.assert_array_equal(k, np.arange(1, len(trace)))
                np.testing.assert_array_equal(dt[k], .02)
                old = trace[k, 312:351].astype(float)
                new = trace[k, 351:390].astype(float)
                U = trace[k, 454:460].astype(float) * scales * dt[k, None]
            end, start = k * 40, (k - 1) * 40
            rates = [np.array([[leg_kinematics(x[12:14], x[16:18])[2],
                               leg_kinematics(x[14:16], x[18:20])[2]] for x in packet]) for packet in (old, new)]
            rotation = J * ((new[:, 20:22] + new[:, 4:5] + rates[1]) - (old[:, 20:22] + old[:, 4:5] + rates[0]))
            damping = .005 * (old[:, 20:22] + new[:, 20:22]) / 2 * dt[k, None]
            relative = (U[:, 4:6] / dt[k, None] - J * (new[:, 20:22] - old[:, 20:22]) / dt[k, None])
            relative = (relative / hw.MOTOR_PEAK_TORQUE).astype(np.float32).astype(float) * hw.MOTOR_PEAK_TORQUE
            estimates = [relative, (U[:, 4:6] - rotation) / dt[k, None], (U[:, 4:6] - rotation - damping) / dt[k, None]]
            bounds = [(f * U[:, 4:6] - rotation - damping) / dt[k, None] for f in (.95, 1.05)]
            lower, upper = np.minimum(*bounds), np.maximum(*bounds)
            with np.load(dense, allow_pickle=False) as z:
                pre, post = z['pre'], z['post']
                assert np.all(end <= len(post))
                np.testing.assert_array_equal(new[:, 12:16], post[end - 1][:, qids])
                np.testing.assert_array_equal(new[:, 16:20], post[end - 1][:, m.nq + np.array(vids)])
                np.testing.assert_array_equal(new[:, 20:22], post[end - 1][:, m.nq + wheels])
                np.testing.assert_array_equal(old[1:], new[:-1])
                gyro_pre_max = max(gyro_pre_max, float(abs(new[:, 4] - pre[end - 1, m.nq + 4]).max()))
                gyro_post_max = max(gyro_post_max, float(abs(new[:, 4] - post[end - 1, m.nq + 4]).max()))
                impulse = np.array([post[s:e, -2:].sum(axis=0) * .0005 for s, e in zip(start, end)])
                bimp = np.array([.005 * pre[s:e][:, m.nq + wheels].sum(axis=0) * .0005 for s, e in zip(start, end)])
                before = pre[start][:, m.nq + wheels] + pre[start, m.nq + 4, None]
                after = post[end - 1][:, m.nq + wheels] + post[end - 1, m.nq + 4, None]
                for side, ids in enumerate(parents):
                    before[:, side] += pre[start][:, m.nq + ids].sum(axis=1)
                    after[:, side] += post[end - 1][:, m.nq + ids].sum(axis=1)
                reference = (impulse - J * (after - before) - bimp) / dt[k, None]
                exact_U = np.array([pre[s:e, -6:].sum(axis=0) * .0005 for s, e in zip(start, end)])
                np.testing.assert_allclose(U, exact_U, rtol=1e-6, atol=1e-8)
            coverage = (reference >= lower) & (reference <= upper)
            es = np.stack([x - reference for x in estimates])
            expected = dict(actor_rows=k, sensor_end_steps=end, dt=dt[k], public_old=old, public_new=new,
                            public_command_integral=U, diagnostic_spin_axis_balance_Nm=reference,
                            diagnostic_actual_impulse=impulse, diagnostic_damping_impulse=bimp,
                            diagnostic_true_absolute_omega_before=before, diagnostic_true_absolute_omega_after=after,
                            gain_interval_lower_Nm=lower, gain_interval_upper_Nm=upper,
                            gain_interval_contains_diagnostic=coverage)
            expected.update(zip(NAMES, estimates))
            expected.update((n + '_error_Nm', e) for n, e in zip(NAMES, es))
            with np.load(saved, allow_pickle=False) as z:
                assert set(z.files) == set(expected)
                for name, value in expected.items():
                    np.testing.assert_allclose(z[name], value, rtol=1e-12, atol=1e-12, err_msg=name)
            assert len(k) == r['intervals'] and row['success'] == r['task_success']
            for n, e in zip(NAMES, es):
                for field, value in [('rmse_Nm', np.sqrt(np.mean(e**2))), ('mean_bias_Nm', e.mean()), ('max_abs_error_Nm', abs(e).max())]:
                    np.testing.assert_allclose(r[field][n], value, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(r['gain_interval_coverage'], coverage.mean(), rtol=1e-12)
            errors[key[0]].append(es)
            covers[key[0]].append(coverage)
            improved[key[0]] += int(np.mean(es[2]**2) < np.mean(es[0]**2))
        print('REVIEWED', entry['arm'], entry['batch'], len(seen), flush=True)
    assert seen == set(records)
    summary = {}
    for arm in p['arms']:
        es = np.concatenate(errors[arm], axis=1)
        coverage = np.concatenate(covers[arm])
        summary[arm] = dict(episodes=len(errors[arm]), intervals=es.shape[1],
                            pooled_rmse_Nm=dict(zip(NAMES, map(float, np.sqrt(np.mean(es**2, axis=(1, 2)))))),
                            corrected_rmse_lower_than_relative_cases=improved[arm],
                            gain_only_interval_coverage=float(coverage.mean()))
        assert summary[arm]['episodes'] == 40
        for name in NAMES:
            np.testing.assert_allclose(summary[arm]['pooled_rmse_Nm'][name], c['summary'][arm]['pooled_rmse_Nm'][name], rtol=1e-12)
        for field in ('intervals', 'corrected_rmse_lower_than_relative_cases', 'gain_only_interval_coverage'):
            np.testing.assert_allclose(summary[arm][field], c['summary'][arm][field], rtol=1e-12)
        assert summary[arm]['pooled_rmse_Nm'][NAMES[1]] > summary[arm]['pooled_rmse_Nm'][NAMES[0]]
        assert summary[arm]['gain_only_interval_coverage'] < 1
    assert sum(s['intervals'] for s in summary.values()) == c['intervals'] == 29708
    atomic_json(destination, dict(verified=True, round=293, summary=summary,
        completion_sha256=sha(OUT / 'audit_completion.json'), reviewer_sha256=sha(__file__),
        checked_all_saved_arrays_against_original_sources=True, estimator_functions_called=False,
        gyro_pre_max_rad_s=gyro_pre_max, gyro_post_max_rad_s=gyro_post_max,
        new_simulation=0, new_training_samples=0, new_optimizer=0, model_compile_only=True,
        information_audit_closed=True, known_damping_correction_supported=True,
        exact_force_or_normal_admitted=False, robust_traction_admitted=False,
        controller_admitted=False, formal_PPO_admitted=False, novelty_qualified=False,
        conclusion='Known damping removal explains the dominant improvement against a mechanical-balance diagnostic. Carrier correction alone worsens pooled error in both arms; gain-only intervals miss about 9.3%. No independently measured contact wrench or positive normal-force lower bound is established.',
        next='Close this audit without parameter fitting or new rollout. Reassess the method-level evidence gap in round294 before the scheduled round295 direction review and cleanup.'))
    print('PASS293 independent 80-episode source/array/statistic review; information audit closed', summary, flush=True)


if __name__ == '__main__':
    run()
