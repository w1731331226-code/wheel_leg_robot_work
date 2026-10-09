"""Two fixed public-history feedback baselines; ordinary PD, not new theory."""
import numpy as np
from probe_route_feedback import actions as route_actions


def actions(history, variant, yaw_config):
    x = np.asarray(history, dtype=float)
    if x.ndim != 2 or x.shape[1] != 481 or not np.isfinite(x).all() or variant not in ('center', 'lower'):
        raise ValueError('Finite raw481 and center/lower feedback required')
    packet = x[:, 351:390]
    offset = np.clip(.30*packet[:, 0]+.12*packet[:, 3], -.035, .035)
    difference = offset/.035
    result = np.zeros((len(x), 3), np.float32)
    result[:, 1] = difference
    if variant == 'lower':
        result[:, 0] = -abs(difference)
    result[:, 2] = route_actions(packet, yaw_config['candidate'], yaw_config['arm'])[0][:, 2]
    return result


def unit(yaw_config):
    x = np.zeros((3, 481)); x[:, 351] = [.01, -.01, 0]; x[:, 354] = [.1, -.1, 0]
    a = actions(x, 'center', yaw_config); b = actions(x, 'lower', yaw_config)
    np.testing.assert_array_equal(a[:, 0], 0)
    np.testing.assert_allclose(a[:, 1], [.015/.035, -.015/.035, 0], rtol=0, atol=2e-8)
    np.testing.assert_array_equal(b[:, 0], -abs(b[:, 1]))
    np.testing.assert_array_equal(a[:, 1:], b[:, 1:])
    unused = x.copy(); unused[:, :351] = 99; unused[:, 390:] = -99
    np.testing.assert_array_equal(actions(unused, 'lower', yaw_config), b)
    mirrored = x.copy(); mirrored[:, [351, 353, 354, 356, 389]] *= -1
    c = actions(mirrored, 'lower', yaw_config)
    np.testing.assert_array_equal(c[:, 0], b[:, 0]); np.testing.assert_array_equal(c[:, 1:], -b[:, 1:])
    assert np.all(abs(b) <= 1)
    print('PASS public481 feedback,zero/mirror/bounds andunused-field independence;0physics', flush=True)
