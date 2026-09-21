"""统一拓扑物理核对、批量动力学核对及真实Gym接口/重置检查。"""
from pathlib import Path
import sys,json,time
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'wheelleg_ppo/tools')]
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from ppo_env import WheelLegEnv,Scenario
from gpu_env import WarpEnv
from native.models import model,bank
from native.environment import NativeEnv

report={}
for scenario in (Scenario(),Scenario(height_l=.012,mu_l=.7,mu_r=.9)):
    ref=WheelLegEnv(scenario=scenario,max_seconds=6.)
    isolated=WarpEnv(scenario=scenario,max_seconds=6.)
    isolated.source.mujoco=mujoco;isolated.source.build_model=model
    other=isolated.env
    ref.reset(seed=1609);other.reset(seed=1609)
    for _ in range(300):
        ref.step(np.zeros(3));other.step(np.zeros(3))
        np.testing.assert_allclose(ref.data.qpos,other.data.qpos,atol=1e-10,rtol=0)
        np.testing.assert_allclose(ref.data.qvel,other.data.qvel,atol=1e-9,rtol=0)
    ref.close();isolated.close()
report['padded_model_cpu_6s_equivalence']=True
wp.set_device('cuda:0')
cpu,gm,gd,scenarios=bank(4,stage=3)
ctrl=np.tile(np.array([.2,-.2,.3,-.3,.1,-.1],np.float32),(4,1));gd.ctrl.assign(ctrl)
mjw.step(gm,gd);positions=gd.qpos.numpy();velocities=gd.qvel.numpy();errors=[]
for i,s in enumerate(scenarios):
    m=model(s);d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,m.keyframe('stand').id);mujoco.mj_forward(m,d)
    single_m=mjw.put_model(m);single_d=mjw.put_data(m,d,nconmax=32,njmax=128);single_d.ctrl.assign(ctrl[i:i+1])
    mjw.step(single_m,single_d)
    errors.append(float(np.max(abs(positions[i]-single_d.qpos.numpy()[0]))))
    np.testing.assert_allclose(positions[i],single_d.qpos.numpy()[0],atol=1e-6,rtol=0)
    np.testing.assert_allclose(velocities[i],single_d.qvel.numpy()[0],atol=1e-5,rtol=0)
report['batched_model_step_max_qpos_error']=max(errors)
env=NativeEnv(8,stage=3);obs=env.reset();assert obs.shape==(8,32)
done_count=0;steps=0;start=time.perf_counter()
for i in range(600):
    obs,r,done,infos=env.step(np.tile([.1,-.1,.1],(8,1)).astype(np.float32))
    assert np.isfinite(obs).all() and np.isfinite(r).all()
    for j in np.flatnonzero(done):
        assert infos[j]['terminal_observation'].shape==(32,)
        assert 'episode' in infos[j] and infos[j]['duration_s']>0
    done_count+=int(done.sum());steps+=8
assert done_count>=8
for bad in (np.zeros((8,2)),np.full((8,3),np.nan),np.ones((8,3))*2):
    try:env.step_async(bad)
    except ValueError:pass
    else:raise AssertionError('非法动作未拒绝')
report.update(episodes_completed=done_count,policy_steps=steps,seconds=time.perf_counter()-start,finite_and_reset=True)
env.close()
out=Path('wheelleg_warp/results/native_gpu_20260921');out.mkdir(exist_ok=True)
(out/'engineering.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS',json.dumps(report,ensure_ascii=False),flush=True)
