"""Check per-world inertia and fail-closed shared options; optional one-step GPU parity."""
from pathlib import Path
import argparse,json,sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from native.models import batch,model
from ppo_env import Scenario


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    args=parser.parse_args();wp.init();wp.set_device(args.device)
    scenarios=[Scenario(mass=7.),Scenario(mass=8.)]
    cpu=[model(s) for s in scenarios];_,gm,gd,_=batch(cpu,scenarios)
    for models,cases in (([],[]),(cpu,scenarios[:1])):
        try:batch(models,cases)
        except ValueError:pass
        else:raise AssertionError('空批量或模型/场景数量不匹配未被拒绝')
    expected=np.array([m.stat.meaninertia for m in cpu])
    np.testing.assert_allclose(gm.stat.meaninertia.numpy(),expected,rtol=1e-7)
    assert abs(expected[0]-expected[1])>.1
    rejected=[]
    # Cover scalar solver count, array-backed tolerance, and topology-affecting
    # jacobian mode (not exposed as a Warp Option field).
    for name,value in (('iterations',50),('tolerance',1e-5),('jacobian',int(mujoco.mjtJacobian.mjJAC_SPARSE))):
        old=getattr(cpu[1].opt,name);assert not np.array_equal(old,value)
        setattr(cpu[1].opt,name,value)
        try:batch(cpu,scenarios)
        except ValueError as exc:
            assert f'opt.{name}' in str(exc);rejected.append(name)
        else:raise AssertionError(f'未拒绝异质 opt.{name}')
        finally:setattr(cpu[1].opt,name,old)
    report=dict(device=args.device,expected_meaninertia=expected.tolist(),actual_meaninertia=gm.stat.meaninertia.numpy().tolist(),rejected_options=rejected)
    if args.device!='cpu':
        ctrl=np.tile(np.array([.2,-.2,.3,-.3,.1,-.1],np.float32),(2,1));gd.ctrl.assign(ctrl)
        mjw.step(gm,gd);qpos=gd.qpos.numpy();qvel=gd.qvel.numpy();errors=[]
        for i,m in enumerate(cpu):
            d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,m.keyframe('stand').id);mujoco.mj_forward(m,d)
            sm=mjw.put_model(m);sd=mjw.put_data(m,d,nconmax=64,njmax=256);sd.ctrl.assign(ctrl[i:i+1]);mjw.step(sm,sd)
            errors.append(dict(qpos=float(abs(qpos[i]-sd.qpos.numpy()[0]).max()),qvel=float(abs(qvel[i]-sd.qvel.numpy()[0]).max())))
            np.testing.assert_allclose(qpos[i],sd.qpos.numpy()[0],atol=1e-6,rtol=0)
            np.testing.assert_allclose(qvel[i],sd.qvel.numpy()[0],atol=1e-5,rtol=0)
        report['one_step_single_world_errors']=errors
    print('PASS',json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
