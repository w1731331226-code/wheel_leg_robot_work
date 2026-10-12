"""CPU tests for units, Nom-sign convention, vector fit and same-chart B1 yaw."""
from unittest.mock import patch
import numpy as np
import cartesian_classical as classical
from probe_route_feedback import actions as route_actions


def fixture(n=4):
    raw = np.zeros((n,481), np.float32);packet = raw[:,351:390]
    packet[:,0]=[.01,-.01,.03,-.03][:n];packet[:,3]=[.02,-.02,.01,-.01][:n]
    packet[:,2]=[.01,-.02,.02,-.01][:n];packet[:,5]=[.02,-.01,.01,-.02][:n]
    packet[:,9]=.7;packet[:,38]=[.01,-.01,.02,-.02][:n]
    return raw


def run():
    config=dict(kp=.8,kd=.24,roll_gain=.5,arm=1);raw=fixture();packet=raw[:,351:390].astype(float)
    for gamma in classical.GAMMAS:
        before=raw.copy();got,diag=classical.actions(raw,gamma,config)
        d=np.clip(.30*packet[:,0]+.12*packet[:,3],-.035,.035)
        np.testing.assert_array_equal(diag['d'],d)
        np.testing.assert_allclose(diag['DeltaF'],gamma*classical.F0*d/.035,rtol=0,atol=1e-15)
        np.testing.assert_allclose(got[:,0],0,rtol=0,atol=1e-12)
        np.testing.assert_allclose(got[:,1],np.clip(-gamma*d/.035,-1,1),rtol=0,atol=3e-8)
        np.testing.assert_array_equal(got[:,2],route_actions(packet,config,1)[0][:,2])
        np.testing.assert_array_equal(diag['force_sagittal_N'],classical.F0*got[:,:2].astype(float))
        np.testing.assert_array_equal(before,raw);assert got.dtype==np.float32 and abs(got).max()<=1
    # Positive d means left +radial/right -radial; downward legs therefore az<0.
    saturated=raw.copy();saturated[:,351]=1;saturated[:,354]=0
    got,diag=classical.actions(saturated,2,config)
    np.testing.assert_array_equal(diag['d'],.035);np.testing.assert_array_equal(diag['shrink'],.5)
    np.testing.assert_array_equal(got[:,1],-1);np.testing.assert_array_equal(diag['DeltaF'],2*classical.F0)
    legs=np.array([[.12,-.16],[-.28,-.21]])
    def fake_kinematics(q,dq):return (legs[int(q[0])],None,None,None)
    asymmetric=raw[:1].copy();asymmetric[0,363]=0;asymmetric[0,365]=1
    with patch.object(classical,'leg_kinematics',fake_kinematics):
        got,diag=classical.actions(asymmetric,1,config)
        target=diag['DeltaF'][0]*(legs/np.linalg.norm(legs,axis=1)[:,None]);fit=diag['force_raw'][0]
        np.testing.assert_allclose(fit,target.mean(axis=0),rtol=0,atol=1e-15)
        np.testing.assert_allclose(4*fit-2*target.sum(axis=0),0,rtol=0,atol=1e-15)
        cost=lambda f:float(((f-target)**2).sum())
        assert cost(fit)<cost(fit+np.array([.01,-.02]))
    original=raw.copy();original[:,363:367]=[.03,-.02,-.04,.01]
    mirrored=original.copy();p=mirrored[:,351:390]
    p[:,[0,2,3,5,38]]*=-1;p[:,12:16]=p[:,[14,15,12,13]]
    first,diag=classical.actions(original,1,config);second,other=classical.actions(mirrored,1,config)
    np.testing.assert_array_equal(second,-first);np.testing.assert_array_equal(diag['shrink'],other['shrink'])
    used={351,353,354,356,360,363,364,365,366,389}
    changed=original.copy();indices=[i for i in range(481) if i not in used];changed[:,indices]=123
    np.testing.assert_array_equal(classical.actions(changed,1,config)[0],first)
    for arm in (0,1):
        candidate={**config,'arm':arm};got,_=classical.actions(original,.5,candidate)
        np.testing.assert_array_equal(got[:,2],route_actions(original[:,351:390],candidate,arm)[0][:,2])
    bad=raw.copy();bad[0,354]=np.nan
    invalid=[(raw,.25,config),(raw,np.inf,config),(raw,True,config),(bad,1,config),
        (raw[:,:480],1,config),(raw[:0],1,config),(raw,1,{**config,'arm':-1}),
        (raw,1,{**config,'arm':True}),(raw,1,{**config,'arm':1.0}),
        (raw,1,{**config,'kp':np.nan}),(raw,1,{'arm':1})]
    for obs,gamma,candidate in invalid:
        try:classical.actions(obs,gamma,candidate)
        except ValueError:pass
        else:raise AssertionError('Invalid input/config accepted')
    with patch.object(classical,'leg_kinematics',lambda *a:(np.zeros(2),None,None,None)):
        try:classical.actions(raw,1,config)
        except ValueError:pass
        else:raise AssertionError('Degenerate geometry accepted')
    print('PASS378 CPU: Cart sagittal units/sign/vector least squares/common2Box/mirror/public-field independence/exact B1 yaw/invalid rejection;no private Nom LPF/room equivalence,0physics/NN/GPU/FD')


if __name__=='__main__':run()
