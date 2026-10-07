"""Causal commanded-input/rotor-motion features, not a contact-force observer."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools'))
import hardware_profile as hw

SPIN_INERTIA = hw.MOTOR_INERTIA+hw.TIRE_MASS*hw.WHEEL_RADIUS**2+.5*hw.HUB_MASS*hw.WHEEL_HUB_RADIUS**2
SCALES = np.array([hw.HIP_PEAK_TORQUE]*4+[hw.MOTOR_PEAK_TORQUE]*2)


def features(previous_packet, current_packet, command_integral, sensor_elapsed_s):
    """Integral is final ctrl before actuator gain over the SAME sensor interval.

    Full Nom+residual commands are required; accepted residual/request alone is
    invalid. The caller must establish timestamps/delay and episode ownership.
    Rotor proxy omits carrier/base coupling and unknown actuator gains; it is
    a discrepancy feature, not true external torque, normal force or support.
    """
    old, new = np.asarray(previous_packet, float), np.asarray(current_packet, float)
    impulse, elapsed = np.asarray(command_integral, float), np.asarray(sensor_elapsed_s, float)
    if old.ndim != 2 or old.shape[1] != 39 or new.shape != old.shape:
        raise ValueError('Expected two finite raw39 packets')
    if impulse.shape != (len(old), 6) or elapsed.shape != (len(old),):
        raise ValueError('Command impulse6 and sensor elapsed must match worlds')
    if not all(np.isfinite(v).all() for v in (old, new, impulse, elapsed)) or np.any(elapsed < 0):
        raise ValueError('Nonfinite inputs or reversed sensor time')
    stopped = elapsed == 0
    if np.any(impulse[stopped] != 0) or np.any(new[stopped, 20:22] != old[stopped, 20:22]):
        raise ValueError('Zero sensor interval requires zero impulse and unchanged wheel speed')
    mean = np.divide(impulse, elapsed[:, None], out=np.zeros_like(impulse), where=elapsed[:, None]>0)
    acceleration = np.divide(new[:, 20:22]-old[:, 20:22], elapsed[:, None],
                             out=np.zeros((len(old), 2)), where=elapsed[:, None]>0)
    discrepancy = mean[:, 4:6]-SPIN_INERTIA*acceleration
    output = np.concatenate([mean/SCALES, discrepancy/hw.MOTOR_PEAK_TORQUE], axis=1).astype(np.float32)
    if not np.isfinite(output).all(): raise ValueError('Feature overflow')
    return output


def unit():
    old = np.zeros((3, 39)); new = old.copy(); impulse = np.zeros((3, 6)); dt = np.array([.02, .01, 0.])
    impulse[0, 4:6] = np.array([.2, -.3])*dt[0]
    new[0, 20:22] = impulse[0, 4:6]/SPIN_INERTIA
    result = features(old, new, impulse, dt)
    np.testing.assert_allclose(result[0, 6:], 0, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(result[2], 0)
    # Zero residual can coexist with nonzero total Nom command and input feature.
    assert result[0, 4] > 0 and result[0, 5] < 0
    new[1, 20:22] = [2., -2.]
    assert features(old, new, impulse, dt)[1, 6] < 0
    np.testing.assert_array_equal(old, np.zeros((3, 39)))
    assert np.count_nonzero(impulse) == 2
    unused = new.copy(); unused[:, [j for j in range(39) if j not in (20, 21)]] = 123
    np.testing.assert_array_equal(features(old, new, impulse, dt), features(old, unused, impulse, dt))
    for bad_dt, bad_impulse in [(np.array([-.02, .01, 0]), impulse), (dt, impulse+1), (dt, impulse*np.nan)]:
        try: features(old, new, bad_impulse, bad_dt)
        except ValueError: pass
        else: raise AssertionError('Invalid interval accepted')
    print('PASS228 impulse alignment preconditions, reset, sign, elapsed, input identity and invalid inputs;0physics')


if __name__ == '__main__': unit()
