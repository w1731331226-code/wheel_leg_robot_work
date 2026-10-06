"""Offline analytic comparator proposal; no controller installation or safety claim."""
import numpy as np
from wheel_motion_proxy import proxy,hw

TAU=-.0005/np.log(1-.025)  # Existing Nom gyro pole;not fitted to development outcomes.
WHEEL_INERTIA=hw.MOTOR_INERTIA+hw.TIRE_MASS*hw.WHEEL_RADIUS**2+.5*hw.HUB_MASS*hw.WHEEL_HUB_RADIUS**2


def propose(packet,previous_gamma,dt=.02):
    p=proxy(packet);x=np.asarray(packet);gamma=np.asarray(previous_gamma,dtype=float)
    if gamma.shape!=(len(x),) or not np.isfinite(gamma).all() or np.any((gamma<0)|(gamma>1)) or not np.isfinite(dt) or dt<=0:
        raise ValueError('Need one finite gamma in[0,1]/world and positive dt')
    rated=hw.MOTOR_RATED_RPM*2*np.pi/60
    discrepancy=np.max(np.abs(p['wheel_body_discrepancy_m_s']),axis=1)/(hw.WHEEL_RADIUS*rated)
    capacity=p['nominal_capacity_fraction'].min(axis=1)
    denominator=capacity+discrepancy**2
    target=np.divide(capacity,denominator,out=np.ones_like(capacity),where=denominator>0)
    next_gamma=gamma+(1-np.exp(-dt/TAU))*(target-gamma)
    # This is a torque REQUEST only;existing Nom box/L1 slew/motor envelope must apply.
    omega=x[:,20:22].astype(float)
    damping=-WHEEL_INERTIA/TAU*np.sign(omega)*np.maximum(np.abs(omega)-rated,0)
    return dict(gamma_target=target,gamma=next_gamma,velocity_reference=next_gamma*x[:,9],
                wheel_damping_request_Nm=damping,nominal_capacity_fraction=capacity)


def ideal_task_map(left_coupling,right_coupling,half_track):
    """Illustrative longitudinal contact map,not the full closed-chain dynamics."""
    if min(left_coupling,right_coupling,half_track)<0 or not np.isfinite([left_coupling,right_coupling,half_track]).all():
        raise ValueError('Invalid illustrative contact coefficients')
    return np.array([[left_coupling,right_coupling],[-half_track*left_coupling,half_track*right_coupling]])


def unit():
    x=np.zeros((3,39),np.float32);x[:,6]=x[:,9]=.7;x[:,20:22]=.7/hw.WHEEL_RADIUS
    r=propose(x,np.ones(3));np.testing.assert_allclose(r['gamma'],1,atol=1e-14,rtol=0)
    np.testing.assert_array_equal(r['wheel_damping_request_Nm'],0)
    x[0,20]=72.;x[1,21]=-72.;x[1,6]=x[1,9]=-.7
    r=propose(x,np.ones(3));assert 0<r['gamma'][0]<1 and 0<r['gamma'][1]<1
    assert r['wheel_damping_request_Nm'][0,0]<0 and r['wheel_damping_request_Nm'][1,1]>0
    assert r['velocity_reference'][1]<0 and abs(r['velocity_reference'][1])<.7
    assert r['gamma'][2]>1-1e-14
    both=ideal_task_map(1,1,hw.TRACK_WIDTH/2);single=ideal_task_map(0,1,hw.TRACK_WIDTH/2)
    assert np.linalg.matrix_rank(both)==2 and np.linalg.matrix_rank(single)==1
    target=np.array([1.,0.]);np.testing.assert_allclose(both@np.linalg.solve(both,target),target)
    assert np.linalg.norm(single@np.linalg.lstsq(single,target,rcond=None)[0]-target)>0
    try:propose(x,np.ones(3)*1.1)
    except ValueError:pass
    else:raise AssertionError('Invalid reference memory accepted')


if __name__=='__main__':unit();print('PASS analytic comparator algebra/noop/sign/bounds and conditional task-rank examples')
