"""Public symmetric kinematic reference, not a dynamic equilibrium or safety set."""
import numpy as np
import wheelleg_sim as sim


def reference(packet):
    packet = np.asarray(packet)
    if packet.ndim != 2 or packet.shape[1] != 39 or not np.isfinite(packet).all():
        raise ValueError('Expected finite raw39 packet')
    height = packet[:, 11].astype(np.float64)+.3
    if np.any(height < .115-2e-7) or np.any(height > .38+2e-7):
        raise ValueError('Public commanded height outside declared range')
    height = np.clip(height, .115, .38)
    result = np.zeros(packet.shape, np.float32)
    # Native39 contract: command[9], commandedheight-.3[11], ownrequest[32:38], routey[38].
    result[:, 6] = result[:, 9] = packet[:, 9]
    result[:, 11] = packet[:, 11]
    # Condition both queries on identical delayed accepted/current own-request memory.
    # Otherwise an actor could keep a nonzero mean from its own residual context alone.
    result[:, 26:38] = packet[:, 26:38]
    result[:, 20:22] = packet[:, 9:10]/sim.hw.WHEEL_RADIUS
    result[:, 22:24] = height[:, None]
    for i, h in enumerate(height):
        qa, qb = sim.ik(float(h))
        result[i, 12:16] = [qa, qb, qa, qb]
    return result


def unit():
    x = np.zeros((15, 39), np.float32)
    x[:, 11] = np.repeat(np.array([.115, .16, .24, .3, .38])-.3, 3)
    x[:, 9] = np.tile([-.8, 0., .8], 5)
    x[:, 26:38] = np.random.default_rng(199001).uniform(-.5, .5, (15, 12))
    saved = x.copy(); r = reference(x)
    np.testing.assert_array_equal(x, saved)
    np.testing.assert_array_equal(reference(r), r)
    np.testing.assert_array_equal(r[:, 26:38], x[:, 26:38])
    assert np.isfinite(r).all() and not r[:, 38].any()
    bad = x.copy(); bad[0, 11] = -.3
    try: reference(bad)
    except ValueError: pass
    else: raise AssertionError('Invalid height accepted')
    return r
