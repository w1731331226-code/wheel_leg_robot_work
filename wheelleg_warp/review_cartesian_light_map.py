"""Independent CPU mapping/parking receipt of sealed v2 records; no simulation."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'
RUNTIME = BASE / 'cartesian_training_interface_v2/runtime'
# Frozen363/366 tolerances and arithmetic, never selected from the v2 outcomes.
TOL = dict(geometry=1e-7, canonical=1e-6, motor=1e-6, alpha=1e-7,
           hip_limits=1e-12, projection=1e-7, accepted_residual=1e-12)
F0, H0, W0 = .1 * 7 * 9.81 / 2, 1., .3


def pairs(a):
    return np.stack((a[..., 0], -a[..., 0], a[..., 1], -a[..., 1],
                     a[..., 2], -a[..., 2]), axis=-1)


def geometry(q):
    """Vectorized frozen CPU leg_kinematics geometry and polar-J transpose."""
    q = np.asarray(q, float)
    if q.shape[-1] != 2 or not np.isfinite(q).all():
        raise ValueError('Finite active joint pairs required')
    p1, p4 = -2.3873937057493047 - q[..., 0], -.7541989478404885 - q[..., 1]
    bx, bz = .15 * np.cos(p1), .15 * np.sin(p1)
    dx, dz = .15 + .15 * np.cos(p4), .15 * np.sin(p4)
    aa, bb = .54 * (dx - bx), .54 * (dz - bz)
    cc = (dx - bx) ** 2 + (dz - bz) ** 2
    p2 = 2 * np.arctan2(bb - np.sqrt(np.maximum(0, aa * aa + bb * bb - cc * cc)), aa + cc)
    cx, cz = bx + .27 * np.cos(p2), bz + .27 * np.sin(p2)
    leg = np.stack((cx - .075, cz), axis=-1)
    length = np.linalg.norm(leg, axis=-1)
    cbx, cbz, cdx, cdz = cx - bx, cz - bz, cx - dx, cz - dz
    det = cbx * cdz - cbz * cdx
    if np.any(length <= 1e-6) or np.any(abs(det) <= 1e-12):
        raise ValueError('Degenerate frozen linkage')
    first = .15 * (cbx * np.sin(p1) - cbz * np.cos(p1))
    second = .15 * (cdx * np.sin(p4) - cdz * np.cos(p4))
    x1, x2 = cdz * first / det, -cbz * second / det
    z1, z2 = -cdx * first / det, cbx * second / det
    lx, lz = leg[..., 0], leg[..., 1]
    jac = np.empty(q.shape[:-1] + (2, 2))
    jac[..., 0, 0] = (lx * x1 + lz * z1) / length
    jac[..., 1, 0] = (lx * x2 + lz * z2) / length
    jac[..., 0, 1] = (-lz * x1 + lx * z1) / length ** 2
    jac[..., 1, 1] = (-lz * x2 + lx * z2) / length ** 2
    return leg, jac


def request(legs, latent):
    if legs.shape != latent.shape[:-1] + (2, 2) or latent.shape[-1] != 3:
        raise ValueError('Paired legs and latent3 required')
    length = np.linalg.norm(legs, axis=-1)
    if not np.isfinite(legs).all() or not np.isfinite(latent).all() or np.any(abs(latent) > 1) or np.any(length <= 1e-6):
        raise ValueError('Invalid geometry or latent')
    force = F0 * latent[..., :2]
    radial = np.sum(legs * force[..., None, :], axis=-1) / length
    moment = legs[..., 0] * force[..., None, 1] - legs[..., 1] * force[..., None, 0]
    u = np.stack((radial[..., 0] / F0, -radial[..., 1] / F0,
                  moment[..., 0] / H0, -moment[..., 1] / H0,
                  latent[..., 2], -latent[..., 2]), axis=-1)
    alpha = 1 / np.maximum(1, np.maximum(abs(u).max(axis=-1), abs(u[..., 2] + u[..., 3]) / .1))
    return u * alpha[..., None], alpha


def limits(speed):
    rpm = abs(speed.astype(float)) * 60 / (2 * np.pi)
    rated, free = np.array([175.] * 4 + [490.] * 2), np.array([280.] * 4 + [710.] * 2)
    return np.array([40.] * 4 + [4.5] * 2) * np.where(rpm > rated, np.maximum(0, (free - rpm) / (free - rated)), 1) / np.array([1.] * 4 + [1.05] * 2)


def projection(base, motor, bound):
    ratio = np.full_like(motor, np.inf)
    np.divide(bound - base, motor, out=ratio, where=motor > 0)
    np.divide(-bound - base, motor, out=ratio, where=motor < 0)
    return np.clip(np.minimum(1, ratio.min(axis=-1)), 0, 1)


def close(name, value, expected, maxima):
    error = float(abs(value - expected).max()) if value.size else 0.
    maxima[name] = max(maxima.get(name, 0.), error)
    np.testing.assert_allclose(value, expected, rtol=0, atol=TOL[name], err_msg=name)


def mapping(cart, guard, reference, mode, maxima):
    if not np.isfinite(cart).all() or not np.isfinite(guard).all() or np.any(cart[:, 33] != 0):
        raise ValueError('Invalid active control records')
    current, jac = geometry(guard[:, 50:54].reshape(-1, 2, 2))
    legs = current if mode == 0 else reference.reshape(-1, 2, 2)
    close('geometry', cart[:, 34:38], legs.reshape(-1, 4), maxima)
    u, alpha = request(legs, cart[:, 4:7])
    close('canonical', cart[:, 7:13], u, maxima)
    close('alpha', cart[:, 13], alpha, maxima)
    wrench = np.stack((F0 * u[:, :2], H0 * u[:, 2:4]), axis=-1)
    motor = np.empty((len(cart), 6))
    motor[:, :4] = np.einsum('nsij,nsj->nsi', jac, wrench).reshape(-1, 4)
    motor[:, 4:] = W0 * u[:, 4:]
    close('motor', cart[:, 15:21], motor, maxima)
    # Wheel velocities are absent from guard; recorded bounds remain available.
    close('hip_limits', cart[:, 27:31], limits(np.column_stack((guard[:, 54:58], np.zeros((len(cart), 2)))))[:, :4], maxima)
    bound = cart[:, 27:33]
    assert np.all((bound >= 0) & (bound <= np.array([40.] * 4 + [4.5 / 1.05] * 2) + 1e-12))
    close('projection', cart[:, 14], projection(guard[:, 2:8], motor, bound), maxima)
    np.testing.assert_array_equal(guard[:, 26], cart[:, 14])
    close('accepted_residual', guard[:, 8:14], cart[:, 14, None] * cart[:, 15:21], maxima)
    candidate = np.clip(guard[:, 2:8] + cart[:, 14, None] * cart[:, 15:21], -bound, bound).astype(np.float32)
    final = np.clip(guard[:, 14:20] + guard[:, 27, None] * guard[:, 8:14], -bound, bound).astype(np.float32)
    np.testing.assert_array_equal(candidate, cart[:, 21:27].astype(np.float32))
    np.testing.assert_array_equal(final, guard[:, 20:26].astype(np.float32))


def packet(record, mode, latent, seen, reference, maxima):
    c, g, p = record['trace'], record['guard'], record['parking']
    assert c.shape == (40, 10, 38) and g.shape == (40, 10, 62) and p.shape == (40, 10, 30)
    assert all(np.isfinite(record[k]).all() for k in ('trace','guard','parking','actor','action','reference','latent','shadow','terminal_packet'))
    np.testing.assert_array_equal(record['action'][:, 1::2], -record['action'][:, ::2])
    keep = c[:, :, 0] == 1
    np.testing.assert_array_equal(keep, g[:, :, 0] == 1)
    np.testing.assert_array_equal(keep, p[:, :, 0] == 1)
    counts = keep.sum(axis=0);assert np.all(counts > 0)
    np.testing.assert_array_equal(keep, np.arange(40)[:, None] < counts)
    cv, gv, pv = c[keep], g[keep], p[keep]
    np.testing.assert_array_equal(c[0, :, 1:4], latent)
    np.testing.assert_array_equal(record['actor'][:, 383:389], pairs(latent).astype(np.float32))
    np.testing.assert_array_equal(c[1:, :, 1:4][keep[1:]], c[:-1, :, 4:7][keep[1:]])
    observed_seen = np.maximum.accumulate(np.concatenate((seen[None, :], (p[:, :, 2] != 0) & keep)), axis=0)[1:]
    np.testing.assert_array_equal(p[:, :, 3][keep], observed_seen[keep])
    withdraw = observed_seen & (p[:, :, 2] == 0) & keep
    np.testing.assert_array_equal(p[:, :, 4][keep], withdraw[keep])
    requested = np.broadcast_to(record['action'], (40, 10, 6))
    np.testing.assert_array_equal(p[:, :, 5:11][keep], requested[keep])
    np.testing.assert_array_equal(p[:, :, 11:17][keep], np.where(withdraw[..., None], 0, requested)[keep])
    np.testing.assert_array_equal(cv[:, 4:7], cv[:, 1:4] + np.clip(pv[:, 11:17:2] - cv[:, 1:4], -.01, .01))
    np.testing.assert_array_equal(pv[:, 17:23], pairs(cv[:, 1:4]))
    np.testing.assert_array_equal(pv[:, 23:29], pairs(cv[:, 4:7]))
    np.testing.assert_array_equal(pv[:, 29], 0)
    close('geometry', record['reference'], reference, maxima)
    refs = np.broadcast_to(reference, (40, 10, 4))[keep]
    mapping(cv, gv, refs, mode, maxima)
    world = np.arange(10);last = c[counts - 1, world, 4:7]
    np.testing.assert_array_equal(record['latent'], last)
    np.testing.assert_array_equal(record['shadow'][:, 16:22], pairs(last))
    np.testing.assert_array_equal(record['terminal_packet'][:, 32:38], pairs(last).astype(np.float32))
    return last.copy(), observed_seen[-1].copy(), int(keep.sum()), int(withdraw.sum())


def self_check():
    legs, _ = geometry(np.zeros((1, 2, 2)))
    request(legs, np.zeros((1, 3)))
    for bad in (np.full((1, 2, 2), np.nan), np.zeros((1, 2, 2))):
        try:request(bad, np.zeros((1, 3)))
        except ValueError:pass
        else:raise AssertionError('Invalid geometry accepted')
    cart = np.zeros((1, 38));guard = np.zeros((1, 62))
    cart[:, 0] = guard[:, 0] = 1;cart[:, 13:15] = 1;guard[:, 26:28] = 1
    cart[:, 27:33] = limits(np.zeros((1, 6)));cart[:, 34:38] = legs.reshape(1, 4)
    mapping(cart, guard, legs.reshape(1, 4), 0, {})
    bad = cart.copy();bad[0, 34] += .001
    try:mapping(bad, guard, legs.reshape(1, 4), 0, {})
    except AssertionError:pass
    else:raise AssertionError('Wrong geometry accepted')


def review(runtime_path=None):
    self_check();runtime = Path(runtime_path) if runtime_path is not None else RUNTIME
    completion = json.loads((runtime / 'completion.json').read_text())
    frozen = json.loads((BASE / 'cartesian_pair_action_candidate_v1/runtime_proposal.json').read_text())['leaf_kernel_check']['cpu_tolerances']
    assert frozen['canonical_abs'] == TOL['canonical'] and frozen['motor_request_Nm'] == TOL['motor']
    reports = {}
    for mode, arm in enumerate(('CartLive3', 'CartReset3')):
        directory = runtime / arm;maxima = {};total = withdrawn = resets = 0
        with np.load(directory / 'initial.npz', allow_pickle=False) as z:
            initial = z['public'].copy();actor = np.concatenate((z['frames'].reshape(10, 390), z['inputs'].reshape(10, 72), z['elapsed'], z['valid']), axis=1)
        reference = geometry(initial[:, 12:16].reshape(10, 2, 2))[0].reshape(10, 4)
        with np.load(directory / 'zero_before_autoreset.npz', allow_pickle=False) as z:
            zero = {k:z[k].copy() for k in z.files}
        zero.update(actor=actor, action=np.zeros((10, 6), np.float32))
        _, _, prefix, _ = packet(zero, mode, np.zeros((10, 3)), np.zeros(10, bool), reference, maxima)
        latent = np.zeros((10, 3));seen = np.zeros(10, bool);stored_reference = zero['reference'].copy()
        for block in completion['reports'][arm]['blocks']:
            path = directory / block['path'];assert hashlib.sha256(path.read_bytes()).hexdigest() == block['sha256']
            with np.load(path, allow_pickle=False) as z:
                fields = {name:z[name] for name in ('trace','guard','parking','actor','action','reference','latent','shadow','terminal_packet','done','public_after','latent_after','shadow_after','reference_after')}
                assert all(len(values) == block['packets'] for values in fields.values())
                for k in range(block['packets']):
                    r = {name:values[k] for name,values in fields.items()}
                    np.testing.assert_array_equal(r['reference'], stored_reference)
                    latent, seen, count, park_count = packet(r, mode, latent, seen, reference, maxima)
                    total += count;withdrawn += park_count;done = r['done'];resets += int(done.sum())
                    for name in ('latent','shadow','reference'):
                        np.testing.assert_array_equal(r[name + '_after'][~done], r[name][~done])
                    np.testing.assert_array_equal(r['latent_after'][done], 0)
                    np.testing.assert_array_equal(r['shadow_after'][done, 16:22], 0)
                    np.testing.assert_array_equal(r['public_after'][done, 32:38], 0)
                    np.testing.assert_array_equal(r['public_after'][~done, 32:38], pairs(latent[~done]).astype(np.float32))
                    if done.any():
                        next_reference = geometry(r['public_after'][done, 12:16].reshape(-1, 2, 2))[0].reshape(-1, 4)
                        close('geometry', r['reference_after'][done], next_reference, maxima)
                        reference[done] = next_reference;latent[done] = 0;seen[done] = False
                    stored_reference = r['reference_after'].copy()
        with np.load(directory / 'final_cartesian.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['latent'], latent)
            np.testing.assert_array_equal(z['shadow'], r['shadow_after'])
            np.testing.assert_array_equal(z['reference'], stored_reference)
            close('geometry', z['reference'], reference, maxima)
        reports[arm] = dict(prefix_substeps=prefix,rollout_substeps=total,withdrawn_substeps=withdrawn,reset_rows=resets,max_abs_errors=maxima)
    return dict(verified=True,reports=reports,tolerances=TOL,
        scope='Saved Cartesian mapping, parking, latent and recorded Cart post-reset/private rows only.',
        limits=['Wheel velocities are not stored per substep: wheel speed-dependent limits cannot be independently reconstructed; stored limits checked against public upper bound.',
                'Intermediate post-wait guard/parking/phase/collector private arrays are absent; producer non-done checks cannot be independently repeated.',
                'terminal_history was replaced with producer Replay expectations; actual returned done infos481 are not independently archived.',
                'Per-substep historical Nom flag is absent; accepted-mode1 projection arithmetic checked without claiming flag1 exposure.'],
        new_physics_steps=0,new_learning_samples=0,full_interface_qualified=False)


if __name__ == '__main__':
    print(json.dumps(review(), ensure_ascii=False))
