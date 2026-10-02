"""Same-state command-only check of damping retained through pose projection."""
from pathlib import Path
import sys,json
import numpy as np
import warp as wp
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from native.environment import NativeEnv
from native.controller import control_physical_nominal
from probe_height_115_margin import cases
from state_estimation import leg_kinematics
from test_nominal_boundary import launch
from probe_height_115_action_predict_loow import sha
import controller_candidate
OUT=Path(__file__).resolve().parent


def run():
    assert not (OUT/'fixture.json').exists()
    motion=np.load(OUT/'instrumented/motion.npz')['trace'];index=int(np.argmin(abs(motion[:,4,1]-2.038)))
    qleg=motion[index,4,20:22];vx=motion[index,4,3]
    leg,_,_,jac=leg_kinematics(qleg,np.zeros(2));angular=np.linalg.solve(jac.T,[0.,.05])
    env=NativeEnv.height115_candidate(n=2,scenario=cases()[4:6],residual_scale=0,nominal_correction=True)
    rows=[]
    try:
        for projected in (False,True):
            for amplitude in (0.,1.,-1.):
                env.reset();q=env.data.qpos.numpy();v=env.data.qvel.numpy();ids=env.ids.numpy()
                q[:,ids[:4]]=np.tile(qleg,2);q[:,3:7]=[1.,0.,0.,0.]
                v.fill(0);v[:,0]=vx
                v[:,ids[4:6]]=amplitude*angular;v[:,ids[6:8]]=-amplitude*angular
                env.data.qpos.assign(q);env.data.qvel.assign(v);env.data.sensordata.zero_();env.command.fill_(1.)
                memory=env.k['state'].numpy();memory.fill(0);memory[:,0]=2.038;memory[:,1]=np.linalg.norm(leg)
                memory[:,2]=np.arctan2(leg[1],leg[0])+np.pi/2;memory[:,7]=vx
                reference=env.k['reference'].numpy();reference[:,5]=1.4 if projected else 0.;env.k['reference'].assign(reference)
                outputs=[];diagnostics=[]
                for kernel in (control_physical_nominal,controller_candidate.control_physical_nominal):
                    env.k['state'].assign(memory);env.control_kernel=kernel;launch(env)
                    outputs.append(env.data.ctrl.numpy().copy());diagnostics.append(env.diag.numpy().copy())
                if not projected or amplitude==0:np.testing.assert_array_equal(*outputs)
                for diag in diagnostics:
                    assert np.all(diag[:,6:12]==0) and np.all(diag[:,14]==0)
                virtual=[np.stack([np.linalg.solve(jac,o[0,side*2:side*2+2]) for side in range(2)]) for o in outputs]
                rows.append(dict(projected=projected,amplitude=amplitude,old_projection_count=diagnostics[0][0,28],
                    old_differential_H=float((virtual[0][0,1]-virtual[0][1,1])/2),
                    new_differential_H=float((virtual[1][0,1]-virtual[1][1,1])/2),
                    motor_difference_max=float(abs(outputs[1]-outputs[0]).max()),
                    wheel_mean_difference=float((outputs[1][0,4:6]-outputs[0][0,4:6]).mean())))
        for amplitude in (1.,-1.):
            plain=next(r for r in rows if not r['projected'] and r['amplitude']==amplitude)
            clipped=next(r for r in rows if r['projected'] and r['amplitude']==amplitude)
            assert clipped['old_projection_count']==2
            assert abs(clipped['old_differential_H'])<1e-5
            assert abs(clipped['new_differential_H']-plain['old_differential_H'])<1e-5
            assert abs(clipped['new_differential_H'])>1e-4
            assert abs(clipped['wheel_mean_difference'])<1e-6
        result=dict(rows=rows,command_fixture_only=True,physical_success_claimed=False,
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),OUT/'controller_candidate.py',ROOT/'wheelleg_warp/native/controller.py')})
        (OUT/'fixture.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS original differential damping erased by double projection and restored; inactive/symmetric commands unchanged',rows,flush=True)
    finally:env.close()


if __name__=='__main__':run()
