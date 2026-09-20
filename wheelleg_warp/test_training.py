"""训练适配的最小集成检查：原CPU未污染、GPU观测/奖励/时钟/重置。"""
from dataclasses import asdict
import math
import numpy as np
import mujoco
from gpu_env import WarpEnv
from ppo_env import WheelLegEnv, Scenario

original_step = mujoco.mj_step
cpu = WheelLegEnv(scenario=Scenario(), max_seconds=.08)
gpu = WarpEnv(scenario=Scenario(), max_seconds=.08)
for stage in (1, 2, 3):
    cpu.stage = gpu.env.stage = stage
    obs_cpu, _ = cpu.reset(seed=1609)
    obs_gpu, _ = gpu.reset(seed=1609)
    np.testing.assert_array_equal(obs_cpu, obs_gpu)
    for _ in range(4):
        action = np.array([.1, -.1, .1], dtype=np.float32)
        c = cpu.step(action)
        g = gpu.step(action)
        assert mujoco.mj_step is original_step
        assert g[0].shape == (32,) and np.isfinite(g[0]).all() and math.isfinite(g[1])
        assert g[2:4] == c[2:4]
        assert g[4]['physical_steps'] == c[4]['physical_steps']
        assert abs(g[4]['duration_s'] - g[4]['physical_steps']*.0005) < 1e-12
        np.testing.assert_allclose(cpu.data.qpos, gpu.env.data.qpos, atol=1e-4, rtol=0)
    assert gpu.env.done and gpu.env.reason == 'debug_time_limit'
    try:
        gpu.step(action)
    except RuntimeError:
        pass
    else:
        raise AssertionError('结束后的step必须拒绝')
cpu.close(); gpu.close()
print('PASS：CPU未污染，GPU非零动作/观测/奖励/整数时钟/结束与重置')
