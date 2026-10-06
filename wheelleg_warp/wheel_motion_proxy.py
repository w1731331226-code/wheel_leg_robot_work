"""Public raw39 motion discrepancy and nominal capacity; not contact truth/headroom."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools'))
import hardware_profile as hw


def proxy(packet):
    x=np.asarray(packet)
    if x.ndim!=2 or x.shape[1]!=39 or not np.isfinite(x).all():
        raise ValueError('Expected finite unnormalized raw39 packet')
    omega=x[:,20:22].astype(np.float64);vx=x[:,6:7].astype(np.float64)
    rpm=np.abs(omega)*60/(2*np.pi)
    fraction=np.where(rpm<=hw.MOTOR_RATED_RPM,1.,np.clip((hw.MOTOR_NO_LOAD_RPM-rpm)/(hw.MOTOR_NO_LOAD_RPM-hw.MOTOR_RATED_RPM),0,1))
    return dict(wheel_body_discrepancy_m_s=hw.WHEEL_RADIUS*omega-vx,nominal_capacity_fraction=fraction,
                nominal_capacity_Nm=hw.MOTOR_PEAK_TORQUE*fraction)


def delivered(subset_history,decision_steps,delay_steps):
    if type(delay_steps) is not int or not 0<=delay_steps<=40:
        raise ValueError('Declared delay must be0..40 physical substeps')
    steps=np.asarray(decision_steps)
    if steps.ndim!=1 or not np.issubdtype(steps.dtype,np.integer) or np.any(steps<0) or np.any(steps>=len(subset_history)):
        raise ValueError('Decision endpoint outside saved first episode')
    indices=np.maximum(steps-delay_steps,0)
    return proxy(subset_history[indices]),indices


def unit():
    x=np.zeros((5,39),np.float32);x[:,6]=.7;x[:,20:22]=.7/hw.WHEEL_RADIUS
    assert np.max(np.abs(proxy(x)['wheel_body_discrepancy_m_s']))<1e-7
    np.testing.assert_array_equal(proxy(x)['nominal_capacity_fraction'],1)
    x[2,20]=hw.MOTOR_NO_LOAD_RPM*2*np.pi/60
    assert proxy(x)['nominal_capacity_fraction'][2,0]<1e-7
    y=x.copy();y[:,[i for i in range(39) if i not in (6,20,21)]]=123
    for key,value in proxy(x).items():np.testing.assert_array_equal(value,proxy(y)[key])
    _,indices=delivered(x,np.array([0,2,4]),2);np.testing.assert_array_equal(indices,[0,0,2])
    try:delivered(x,np.array([5]),0)
    except ValueError:pass
    else:raise AssertionError('Future packet was accepted')


if __name__=='__main__':unit();print('PASS raw39 proxy,unused-field independence,delay/reset/future checks')
