"""Algebra prototype only: accepting steering can change common wheel command."""
import numpy as np


def allocate(nominal, bounds, steering):
    b = np.asarray(nominal, dtype=float)
    limits = np.asarray(bounds, dtype=float)
    s = np.asarray(steering, dtype=float)
    if (b.ndim != 2 or b.shape[1] != 2 or not len(b) or limits.shape != b.shape
            or s.shape != (len(b),) or not all(np.isfinite(x).all() for x in (b, limits, s))
            or np.any(limits < 0) or np.any(abs(b) > limits) or np.any(abs(s) > .3)):
        raise ValueError('Expected feasible finite two-wheel Nom/box and steering within 0.3 Nm')
    r = np.c_[s, -s]
    fractions = np.ones_like(b)
    np.divide(limits-b, r, out=fractions, where=r > 0)
    np.divide(-limits-b, r, out=fractions, where=r < 0)
    fractions = np.clip(fractions, 0, 1)
    uniform = b + fractions.min(axis=1)[:, None]*r
    return np.clip(b+r, -limits, limits), uniform, fractions


def self_check():
    rng = np.random.default_rng(157)
    limits = rng.uniform(0, 4.5, (1024, 2)); b = rng.uniform(-1, 1, limits.shape)*limits
    s = rng.uniform(-.3, .3, len(b))
    limits[:4] = [[4.5, 4.5], [4.5, 4.5], [0, 0], [4.5, 4.5]]
    b[:4] = [[4.5, 4.5], [-4.5, -4.5], [0, 0], [0, 0]]; s[:4] = [.3, -.3, .3, 0]
    before = (b.copy(), limits.copy(), s.copy()); projected, uniform, f = allocate(b, limits, s)
    delta = projected-b; common = delta.mean(axis=1); differential = (delta[:, 0]-delta[:, 1])/2
    old_differential = ((uniform-b)[:, 0]-(uniform-b)[:, 1])/2
    assert np.all(abs(projected) <= limits) and np.all(abs(common) <= abs(s)/2+1e-12)
    assert np.all(np.sign(s)*(differential-old_differential) >= -1e-12)
    np.testing.assert_allclose(common,(f[:, 0]-f[:, 1])*s/2,rtol=0,atol=1e-12)
    np.testing.assert_allclose(differential,(f[:, 0]+f[:, 1])*s/2,rtol=0,atol=1e-12)
    nonbinding = np.all(f == 1, axis=1)
    np.testing.assert_array_equal(projected[nonbinding].astype(np.float32),uniform[nonbinding].astype(np.float32))
    np.testing.assert_array_equal(projected[s == 0],b[s == 0])
    # Box normal-cone inequalities at all four corners certify the projection.
    desired = b+np.c_[s, -s]
    for signs in ([1,1], [1,-1], [-1,1], [-1,-1]):
        assert np.all(np.sum((desired-projected)*(limits*signs-projected),axis=1) <= 1e-12)
    assert np.all(np.sum((projected-desired)**2,axis=1) <= np.sum((uniform-desired)**2,axis=1)+1e-12)
    for a, old in zip((b, limits, s), before): np.testing.assert_array_equal(a, old)
    np.testing.assert_allclose(uniform[0],[4.5,4.5],rtol=0,atol=1e-12)
    np.testing.assert_allclose(projected[0],[4.5,4.2],rtol=0,atol=1e-12)
    invalid = [(b[:, :1],limits,s), (b,limits,s[:,None]), (b,-limits,s),
               (b+20,limits,s), (b,limits,np.full(len(b),.31)), (b*float('nan'),limits,s)]
    for args in invalid:
        try: allocate(*args)
        except ValueError: pass
        else: raise AssertionError('Invalid algebra input accepted')
    return dict(verified=True,synthetic_rows=1024,max_abs_common_change_Nm=float(abs(common).max()),
                projection_normal_cone=True,signed_differential_nondegradation=True,
                nonbinding_and_zero_identity=True,input_arrays_unchanged=True,invalid_inputs_rejected=True,
                witness=dict(nominal=[4.5,4.5],request=[.3,-.3],uniform=[4.5,4.5],projected=[4.5,4.2],
                             common_change_Nm=-.15,differential_increment_Nm=.15),
                physics_steps=0,training_updates=0,
                limits='Command-box algebra, not actuator dynamics, yaw acceleration, closed-loop task or learning evidence')


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(),ensure_ascii=False))
