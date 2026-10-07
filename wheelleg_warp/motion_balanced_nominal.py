"""Isolated finite-point prototype; no production controller or gain changes."""
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

sys.path.insert(0, str(ROOT / 'wheelleg_ppo/tools'))
import model_lqr as ml
import wheelleg_sim as sim
from state_estimation import leg_kinematics

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/motion_balanced_nominal_v1'
ORIGINAL = ROOT / 'wheelleg_warp/reference_role_control.py'
COPY = ROOT / 'wheelleg_warp/motion_balanced_control.py'
THETA = '    theta_eq=(D(1)-ratio)*angles[index]+ratio*angles[index+1]\n'
SUPPORT = '    support=(D(1)-ratio)*feed[index,2]+ratio*feed[index+1,2]\n'
THETA_EXTRA = (
    '    if reference.shape[1]==21 and reference[w,20]==D(1) and cmd!=D(0):\n'
    '        theta_eq=reference[w,19]\n'
)
FEED_EXTRA = (
    '    if reference.shape[1]==21 and reference[w,20]==D(1) and cmd!=D(0):\n'
    '        wheel=reference[w,16];hub=reference[w,17];support=reference[w,18]\n'
)


def expected_source():
    source = ORIGINAL.read_text()
    assert source.count(THETA) == source.count(SUPPORT) == 1
    return source.replace(THETA, THETA + THETA_EXTRA).replace(SUPPORT, SUPPORT + FEED_EXTRA)


def reference_values(model, q, ctrl):
    """Current analytic J maps [support, hub] to each leg's motor torques."""
    q = np.asarray(q, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)
    if q.shape != (model.nq,) or ctrl.shape != (model.nu,) or not (
            np.isfinite(q).all() and np.isfinite(ctrl).all()):
        raise ValueError('Finite full reference q/ctrl required')
    qa, qb, qc, qd = [
        q[model.jnt_qposadr[model.joint(name).id]]
        for name in ('alphaL', 'betaL', 'alphaR', 'betaR')
    ]
    np.testing.assert_allclose([qa, qb], [qc, qd], rtol=0, atol=1e-12)
    left = ctrl[[model.actuator(name).id for name in
                 ('motor_wheelL', 'motor_alphaL', 'motor_betaL')]]
    right = ctrl[[model.actuator(name).id for name in
                  ('motor_wheelR', 'motor_alphaR', 'motor_betaR')]]
    np.testing.assert_allclose(left, right, rtol=0, atol=1e-12)
    jac = leg_kinematics([qa, qb], [0., 0.])[3]
    matrix = np.zeros((3, 3))
    matrix[0, 0] = 1.
    matrix[1:, 1] = jac[:, 1]
    matrix[1:, 2] = jac[:, 0]
    feed = np.linalg.solve(matrix, left)
    pitch = sim.euler(type('Pose', (), {'qpos': q})())[1]
    theta = sim.fk_joints(qa, qb)['phi5'] + math.pi / 2 - pitch
    return feed, theta, matrix, left


def private_reference(reference, height, speed, feed, theta):
    """Only registered nodes are enabled; zero speed retains the original path."""
    reference = np.asarray(reference, dtype=float)
    feed = np.asarray(feed, dtype=float)
    values = np.r_[height, speed, theta, feed]
    if reference.shape != (16,) or feed.shape != (3,) or not (
            np.isfinite(reference).all() and np.isfinite(values).all()):
        raise ValueError('Finite original16 reference and three feeds required')
    if height not in (.115, .16, .25, .30, .38) or speed not in (-.7, 0., .7):
        raise ValueError('Only exact registered height/speed nodes admitted')
    result = np.zeros(21)
    result[:16] = reference
    if speed != 0.:
        result[16:19] = feed
        result[19:21] = theta, 1.
    return result


def unit():
    assert not (OUT / 'source_admission.json').exists()
    p = json.loads((OUT / 'proposal.json').read_text())
    assert all(sha(ROOT / n) == value for n, value in p['source_sha256'].items())
    assert COPY.read_text() == expected_source()
    restored = COPY.read_text().replace(THETA_EXTRA, '').replace(FEED_EXTRA, '')
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(ORIGINAL.read_text()))
    assert 'reference.shape[1]' not in ORIGINAL.read_text().replace(
        'reference.shape[1]>', '')  # Original predicates are all lower-bound tests.
    model, _ = sim.load_model(ml.XML, True)
    assert (model.nq, model.nv, model.nu) == (17, 16, 6)
    assert abs(model.body_mass.sum() - 7.) < 1e-12
    records = []
    base = np.arange(16, dtype=float)
    for r in p['moving_references']:
        path = ROOT / r['path']
        assert sha(path) == r['sha256']
        with np.load(path, allow_pickle=False) as z:
            feed, theta, matrix, motor = reference_values(model, z['q'], z['ctrl'])
            np.testing.assert_allclose(matrix @ feed, motor, rtol=0, atol=1e-12)
            packet = private_reference(base, r['height'], r['speed'], feed, theta)
            np.testing.assert_array_equal(packet[:16], base)
            np.testing.assert_array_equal(packet[16:19], feed)
            assert packet[19] == theta and packet[20] == 1.
            zero = private_reference(base, r['height'], 0., feed, theta)
            np.testing.assert_array_equal(zero[:16], base)
            np.testing.assert_array_equal(zero[16:], 0.)
            records.append(dict(height=r['height'], speed=r['speed'],
                                reference_sha256=sha(path), feed=feed.tolist(),
                                theta=theta, matrix=matrix.tolist(),
                                motor=motor.tolist(),
                                reconstruction_error=float(abs(matrix @ feed - motor).max())))
    for height, speed in ((.2, .7), (.115, .8), (.115, float('nan'))):
        try:
            private_reference(base, height, speed, np.zeros(3), 0.)
        except ValueError:
            pass
        else:
            raise AssertionError('Unregistered/nonfinite input accepted')
    atomic_json(OUT / 'source_admission.json', dict(
        verified=True, proposal_sha256=sha(OUT / 'proposal.json'),
        source_sha256={str(f.relative_to(ROOT)): sha(f) for f in (Path(__file__), COPY, ORIGINAL)},
        records=records, source_reversal_exact=True,
        zero_speed_packet_tail_exact0=True, original_prefix_exact=True,
        zero_speed_runtime_equality_pending=True, individual_static_queries=0,
        integration_steps=0, transitionFD_calls=0, optimization_calls=0,
        new_training_samples=0, controller_admitted=False,
        limits='Source/algebra only. Absolute moving feed/theta replace interpolated static values; gains and all other original statements unchanged. Exact registered nominal nodes only. GPU compilation and 70 paired static queries pending; no dynamic stability or method advantage claimed.'))
    print('PASS268: ten current-J reconstructions; source reversal exact; zero-speed packet disabled; no controller queries', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--generate']:
        assert not COPY.exists()
        COPY.write_text(expected_source())
    else:
        unit()
