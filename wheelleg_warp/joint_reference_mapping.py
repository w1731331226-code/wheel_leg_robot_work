"""Necessary reference geometry only; no controller or contact certificate."""
import numpy as np
from derive_joint_reference_authority import LOW, HIGH

EPS = .02
OFFSET_CAP = .035  # Original control_step roll-request cap, not a fitted threshold.


def select(height, baseline_difference, actions):
    h, d0, a = map(lambda x: np.asarray(x, dtype=float), (height, baseline_difference, actions))
    if h.ndim != 1 or d0.shape != h.shape or a.shape != (len(h), 3):
        raise ValueError('Expected height[n], baseline half-difference[n], actions[n,3]')
    if not all(np.isfinite(x).all() for x in (h, d0, a)) or np.any(abs(a) > 1):
        raise ValueError('Finite inputs and actions in [-1,1] required')
    if np.any((h < .115) | (h > HIGH)):
        raise ValueError('Height outside declared .115… .38 domain')
    room = np.minimum(h - LOW, HIGH - h)
    if np.any(abs(d0) > np.minimum(room, OFFSET_CAP)):
        raise ValueError('Baseline must satisfy original fixed-mean room and request cap')
    D = np.minimum(OFFSET_CAP, np.minimum((HIGH - LOW) / 2, np.minimum(h + EPS - LOW, HIGH - h + EPS)))
    d = d0 + a[:, 1] * np.where(a[:, 1] >= 0, D - d0, D + d0)
    lo, hi = np.maximum(LOW + abs(d), h - EPS), np.minimum(HIGH - abs(d), h + EPS)
    center = np.clip(h, lo, hi)
    m = center + a[:, 0] * np.where(a[:, 0] >= 0, hi - center, center - lo)
    # Preserve baseline coordinates exactly; full control identity still needs a bypass.
    zero = (a[:, :2] == 0).all(axis=1)
    m[zero], d[zero] = h[zero], d0[zero]
    return m, d, .3 * a[:, 2]


def unit():
    heights = np.linspace(.115, HIGH, 19)
    h, fraction, am, ad, ay = np.meshgrid(heights, np.linspace(-1, 1, 5),
        np.linspace(-1, 1, 11), np.linspace(-1, 1, 11), [-1., 0., 1.], indexing='ij')
    h, fraction = h.ravel(), fraction.ravel()
    d0 = fraction * np.minimum(OFFSET_CAP, np.minimum(h - LOW, HIGH - h))
    actions = np.column_stack((am.ravel(), ad.ravel(), ay.ravel()))
    m, d, wheel = select(h, d0, actions)
    # Float arithmetic tolerance, not a changed physical or task gate.
    assert np.all(m + d >= LOW - 1e-15) and np.all(m + d <= HIGH + 1e-15)
    assert np.all(m - d >= LOW - 1e-15) and np.all(m - d <= HIGH + 1e-15)
    assert np.all(abs(m - h) <= EPS + 1e-15) and np.all(abs(d) <= OFFSET_CAP + 1e-15)
    assert np.all(abs(wheel) <= .3)
    mz, dz, wz = select(h, d0, np.zeros_like(actions))
    np.testing.assert_array_equal(mz, h); np.testing.assert_array_equal(dz, d0)
    np.testing.assert_array_equal(wz, 0.)
    # Surjectivity onto each declared endpoint at the original request cap.
    D = np.minimum(OFFSET_CAP, np.minimum((HIGH - LOW)/2, np.minimum(h + EPS - LOW, HIGH - h + EPS)))
    np.testing.assert_allclose(d[actions[:, 1] == 1], D[actions[:, 1] == 1], rtol=0, atol=1e-17)
    np.testing.assert_allclose(d[actions[:, 1] == -1], -D[actions[:, 1] == -1], rtol=0, atol=1e-17)
    for args in [([.114], [0], [[0, 0, 0]]), ([.38], [.001], [[0, 0, 0]]),
                 ([.2], [0], [[0, 0, 1.1]]), ([.2], [0], [[np.nan, 0, 0]]),
                 ([.2], [0], [[0, 0]])]:
        try:
            select(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid mapping input accepted')
    print('PASS298 geometry/cap/zero-coordinate/endpoint/invalid-input checks', len(h), 'rows;0physics')
    return len(h)


if __name__ == '__main__':
    unit()
