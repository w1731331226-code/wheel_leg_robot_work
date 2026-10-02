"""Pose projection retains existing differential damping without extra gains."""
import numpy as np
from native.environment import NativeEnv
from probe_height_115_margin import cases
from state_estimation import leg_kinematics
from test_nominal_boundary import launch


def check():
    # A symmetric command fixture near the measured startup boundary, not a
    # physical rollout. Opposite angular rates leave the common rate unchanged.
    qleg=np.array([1.3819689750671387,-.7853073477745056])
    vx=.7036457061878022
    leg,_,_,jac=leg_kinematics(qleg,np.zeros(2))
    angular=np.linalg.solve(jac.T,[0.,.05])
    for common_boundary in (False,True):
        env=NativeEnv.height115_candidate(n=2,scenario=cases()[4:6],residual_scale=0,nominal_correction=common_boundary)
        try:
            values={}
            for projected in (False,True):
                for amplitude in (0.,1.,-1.):
                    env.reset();q=env.data.qpos.numpy();v=env.data.qvel.numpy();ids=env.ids.numpy()
                    q[:,ids[:4]]=np.tile(qleg,2);q[:,3:7]=[1.,0.,0.,0.]
                    v.fill(0);v[:,0]=vx;v[:,ids[4:6]]=amplitude*angular;v[:,ids[6:8]]=-amplitude*angular
                    env.data.qpos.assign(q);env.data.qvel.assign(v);env.data.sensordata.zero_();env.command.fill_(1.)
                    memory=env.k['state'].numpy();memory.fill(0);memory[:,0]=2.038;memory[:,1]=np.linalg.norm(leg)
                    memory[:,2]=np.arctan2(leg[1],leg[0])+np.pi/2;memory[:,7]=vx;env.k['state'].assign(memory)
                    reference=env.k['reference'].numpy();reference[:,5]=1.4 if projected else 0.;env.k['reference'].assign(reference)
                    launch(env);ctrl=env.data.ctrl.numpy();diag=env.diag.numpy()
                    assert np.all(diag[:,14]==0) and np.all(diag[:,6:12]==0)
                    if projected:assert np.all(diag[:,28]==2)
                    virtual=[np.linalg.solve(jac,ctrl[0,2*s:2*s+2]) for s in range(2)]
                    values[projected,amplitude]=((virtual[0][1]-virtual[1][1])/2,float(ctrl[0,4:6].mean()))
            for amplitude in (1.,-1.):
                unprojected=values[False,amplitude][0];projected=values[True,amplitude][0]
                assert abs(unprojected)>1e-4 and abs(projected-unprojected)<1e-5
                assert abs(values[True,amplitude][1]-values[True,0.][1])<1e-6
            assert abs(values[True,0.][0])<1e-6
        finally:env.close()
    print('PASS both common-boundary modes retain original differential damping under double projection; common wheel mean unchanged')


if __name__=='__main__':check()
