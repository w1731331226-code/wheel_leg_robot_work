"""核对GPU同一状态的完整/打包回读逐值一致，再测真实闭环耗时。"""
import json
from pathlib import Path
import sys
import time
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from fast_physics import FastWarpEnv
from gpu_env import WarpEnv
from ppo_env import Scenario


def run():
    wp.init();wp.set_device('cuda:0')
    results=[]
    for scenario in (Scenario(),Scenario(height_l=.012,mu_l=.7,mu_r=.9)):
        env=FastWarpEnv(scenario=scenario,max_seconds=.4)
        env.reset(seed=1609)
        reference=mujoco.MjData(env.env.model)
        for i in range(20):
            env.step(np.array([.1,-.1,.1]))
            data=env.env.data;m=env.env.model
            mjw.get_data_into(reference,m,env.physics.data)
            for name in ('qpos','qvel','sensordata','ctrl','actuator_force','subtree_com'):
                np.testing.assert_array_equal(getattr(data,name),getattr(reference,name),err_msg=name)
            assert data.ncon==reference.ncon
            np.testing.assert_array_equal(data.contact.geom,reference.contact.geom)
            np.testing.assert_array_equal(data.contact.friction,reference.contact.friction)
            np.testing.assert_array_equal(data.contact.geom1,reference.contact.geom[:,0])
        env.close()
    # 各3轮，不把编译/模型初始化算进回读与闭环对比。
    for cls in (WarpEnv,FastWarpEnv):
        times=[]
        for repeat in range(3):
            env=cls(scenario=Scenario(),max_seconds=.42);env.reset(seed=1609)
            env.step(np.zeros(3))
            started=time.perf_counter()
            for i in range(20):env.step(np.array([.1,-.1,.1]))
            times.append(time.perf_counter()-started);env.close()
        results.append(dict(backend=cls.__name__,seconds=times,median=float(np.median(times)),policy_steps=20,physics_steps=800))
    result=dict(same_gpu_state_readback_exact=True,benchmarks=results,speedup=results[0]['median']/results[1]['median'],
                caveat='同机训练仍在运行；这是单环境真实闭环，不是完整PPO速度或独占硬件倍率')
    target=Path('wheelleg_warp/results/packed_transfer_20260921');target.mkdir(exist_ok=True)
    (target/'check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
