"""覆盖越障接触变化，逐策略步核对打包与完整回读。"""
import json
from pathlib import Path
import time
import mujoco
import mujoco_warp as mjw
import numpy as np
from fast_physics import FastWarpEnv
from ppo_env import Scenario

env=FastWarpEnv(scenario=Scenario(height_l=.012,center=1.8,mu_l=.7,mu_r=.9),max_seconds=6.)
env.reset(seed=1609);reference=mujoco.MjData(env.env.model)
counts=set();started=time.perf_counter()
while not env.env.done:
    env.step(np.array([.1,-.1,.1]))
    d=env.env.data;m=env.env.model;mjw.get_data_into(reference,m,env.physics.data)
    for field in ('qpos','qvel','sensordata','ctrl','actuator_force','subtree_com'):
        np.testing.assert_array_equal(getattr(d,field),getattr(reference,field))
    assert d.ncon==reference.ncon;counts.add(d.ncon)
    np.testing.assert_array_equal(d.contact.geom,reference.contact.geom)
    np.testing.assert_array_equal(d.contact.friction,reference.contact.friction)
assert env.env.touched, '必须实际接触障碍，不能只测试未越障时段'
result=dict(pass_exact=True,physics_steps=env.env.steps,contact_counts=sorted(counts),
            touched=[env.env.model.geom(i).name for i in env.env.touched],seconds=time.perf_counter()-started,
            metrics=env.env.metrics())
Path('wheelleg_warp/results/packed_transfer_20260921/contact_check.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,ensure_ascii=False),flush=True)
env.close()
