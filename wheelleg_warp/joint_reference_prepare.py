"""Two extra reference parameters with original normalized 0.01/substep slew."""
import numpy as np
import warp as wp
from native.controller import D


def validate(actions):
    a = np.asarray(actions, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() or np.any(abs(a) > 1):
        raise ValueError('Finite actions[n,3] in [-1,1] required')
    return a


@wp.kernel
def prepare(base: wp.array2d[D], actions: wp.array2d[D], active: wp.array[int],
            filtered: wp.array2d[D], private: wp.array2d[D]):
    w = wp.tid()
    for j in range(16):
        private[w, j] = base[w, j]
    for j in range(2):
        if active[w] != 0:
            filtered[w, j] += wp.clamp(actions[w, j] - filtered[w, j], D(-.01), D(.01))
        private[w, 16+j] = filtered[w, j]


def unit(device):
    n = 3
    base = wp.array(np.arange(n*16).reshape(n, 16), dtype=D, device=device)
    request = validate([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])
    a = wp.array(request, dtype=D, device=device)
    active = wp.array([1, 1, 0], dtype=int, device=device)
    state = wp.zeros((n, 2), dtype=D, device=device)
    private = wp.zeros((n, 18), dtype=D, device=device)
    before = base.numpy().copy()
    for _ in range(40):
        wp.launch(prepare, n, [base, a, active, state, private])
    np.testing.assert_allclose(state.numpy(), request[:, :2]*.4, rtol=0, atol=3e-16)
    np.testing.assert_array_equal(private.numpy()[:, :16], before)
    np.testing.assert_array_equal(base.numpy(), before)
    for bad in ([[np.nan, 0, 0]], [[0, 0, 1.01]], [[0, 0]]):
        try:
            validate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid action accepted')
    print('PASS reference slew/inactive/copy and action boundary;0physics', flush=True)
