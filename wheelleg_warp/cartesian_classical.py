"""Cart3 delayed-public endpoint-vector feedback with the original B1 route yaw.

d is metres; DeltaF is each side's signed +DeltaF/-DeltaF radial amplitude,
not the full left-right difference. Forces use the shared body/leg sagittal
coordinates before latent slew, Cart scaling and guard; no world rotation or
contact-force estimate. Public gyro is delayed and unfiltered here: this does
not reproduce Nom's private LPF or height-room clamp.
"""
import numpy as np
from probe_route_feedback import actions as route_actions
from state_estimation import leg_kinematics

F0 = 3.4335
GAMMAS = (0., .5, 1., 2.)


def actions(raw481, gamma, yaw_config):
    raw = np.asarray(raw481, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 481 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Finite nonempty raw481 packets required')
    if isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, (int, float, np.integer, np.floating)) or gamma not in GAMMAS:
        raise ValueError('Registered gamma0/.5/1/2 required')
    if not isinstance(yaw_config, dict) or not {'kp','kd','roll_gain','arm'} <= yaw_config.keys():
        raise ValueError('Explicit B1 kp/kd/roll_gain/arm required')
    gain = np.asarray([yaw_config[k] for k in ('kp','kd','roll_gain')], float)
    arm = yaw_config['arm']
    if not np.isfinite(gain).all() or np.any(gain < 0) or isinstance(arm, (bool, np.bool_)) or not isinstance(arm, (int, np.integer)) or arm not in (0,1):
        raise ValueError('Finite nonnegative B1 gains and arm0/1 required')
    packet = raw[:, 351:390]
    d = np.clip(.30 * packet[:, 0] + .12 * packet[:, 3], -.035, .035)
    delta = float(gamma) * F0 * d / .035
    try:
        geometry = np.array([[leg_kinematics(row[12+2*s:14+2*s], np.zeros(2))[0] for s in (0,1)] for row in packet])
    except (np.linalg.LinAlgError, ZeroDivisionError) as error:
        raise ValueError('Invalid delayed leg geometry') from error
    length = np.linalg.norm(geometry, axis=-1)
    if not np.isfinite(geometry).all() or np.any(length <= 1e-6):
        raise ValueError('Finite nondegenerate delayed legs required')
    unit = geometry / length[..., None]
    # Least-squares fit to the two desired sagittal vectors DeltaF*eL/eR.
    force_raw = delta[:, None] * unit.sum(axis=1) / 2
    shrink = 1 / np.maximum(1, abs(force_raw / F0).max(axis=1))
    yaw, _ = route_actions(packet, yaw_config, yaw_config['arm'])
    latent = np.column_stack((force_raw * shrink[:, None] / F0, yaw[:, 2])).astype(np.float32)
    latent = np.clip(latent, -1, 1)
    return latent, dict(d=d, DeltaF=delta, geometry=geometry, force_raw=force_raw,
        force_sagittal_N=F0 * latent[:, :2].astype(float), shrink=shrink, latent3=latent.copy())
