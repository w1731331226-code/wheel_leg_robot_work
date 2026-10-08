"""Public history decoder and approximate wheelaxis residuals; never normal-force."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools'))
from state_estimation import leg_kinematics
from execution_input_features import SPIN_INERTIA,SCALES,features


def decode(trace):
    x=np.asarray(trace)
    if x.ndim!=2 or x.shape[1]!=968 or not np.isfinite(x).all():
        raise ValueError('Finite saved968 Actor rows required')
    raw=x[:,:481];frames=raw[:,:390].reshape(-1,10,39)
    dt=raw[:,470].astype(float)*.02
    valid=np.all(raw[:,479:481]==1.,axis=1)&(dt>0)
    if np.any(dt<0) or np.any(dt>.020000001):raise ValueError('Sensor interval out of contract')
    mean=raw[:,454:460].astype(float)*SCALES
    return frames[:,-2].copy(),frames[:,-1].copy(),mean*dt[:,None],dt,valid


def carrier_rate(packet):
    x=np.asarray(packet,float)
    if x.ndim!=2 or x.shape[1]!=39 or not np.isfinite(x).all():raise ValueError('Finite raw39 required')
    return np.array([[leg_kinematics(p[12:14],p[16:18])[2],
                      leg_kinematics(p[14:16],p[18:20])[2]] for p in x])


def estimates(old,new,impulse,dt,damping=.005):
    old,new,impulse,dt=map(lambda a:np.asarray(a,float),(old,new,impulse,dt))
    relative=features(old,new,impulse,dt)[:,6:].astype(float)*SCALES[-2:]
    if np.any(dt<=0):raise ValueError('Only matched positive sensor intervals may be estimated')
    before=old[:,20:22]+old[:,4:5]+carrier_rate(old)
    after=new[:,20:22]+new[:,4:5]+carrier_rate(new)
    rotation_impulse=SPIN_INERTIA*(after-before)
    damping_impulse=damping*.5*(old[:,20:22]+new[:,20:22])*dt[:,None]
    corrected=(impulse[:,4:6]-rotation_impulse)/dt[:,None]
    lower=np.minimum(.95*impulse[:,4:6],1.05*impulse[:,4:6])-rotation_impulse-damping_impulse
    upper=np.maximum(.95*impulse[:,4:6],1.05*impulse[:,4:6])-rotation_impulse-damping_impulse
    return dict(relative_proxy_Nm=relative,carrier_proxy_Nm=corrected,
                corrected_proxy_Nm=(impulse[:,4:6]-rotation_impulse-damping_impulse)/dt[:,None],
                gain_interval_lower_Nm=lower/dt[:,None],gain_interval_upper_Nm=upper/dt[:,None])


def unit():
    old=np.zeros((2,39));new=old.copy();dt=np.array([.02,.02]);u=np.array([[.2,-.3],[-.2,.3]])
    impulse=np.zeros((2,6));impulse[:,4:6]=u*dt[:,None]
    # Valid standing activeq/v and bothcarrier rates0; relative-only free acceleration.
    out=estimates(old,new,impulse,dt)
    np.testing.assert_allclose(out['corrected_proxy_Nm'],u,rtol=0,atol=1e-15)
    assert np.all(out['gain_interval_lower_Nm']<=u) and np.all(out['gain_interval_upper_Nm']>=u)
    new[:,4]=1.;out=estimates(old,new,impulse,dt)
    np.testing.assert_allclose(out['carrier_proxy_Nm'],u-SPIN_INERTIA/.02,rtol=0,atol=1e-15)
    x=np.zeros((2,968));x[1,470]=1.;x[1,479:481]=1.;x[1,454:460]=1.
    a,b,U,d,valid=decode(x)
    np.testing.assert_array_equal(valid,[False,True])
    np.testing.assert_allclose(U[1],SCALES*.02,rtol=0,atol=1e-15)
    try:decode(np.zeros((2,967)))
    except ValueError:pass
    else:raise AssertionError('Malformed history accepted')
    print('PASS wheel information decoder/scales/sign/interval algebra;0physics',flush=True)


if __name__=='__main__':unit()
