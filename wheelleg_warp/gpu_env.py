"""独立加载冻结CPU环境，只替换该实例的物理步；不修改原CPU模块。"""
from dataclasses import asdict, is_dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import warp as wp
from baseline import WarpPhysics, ROOT


class WarpEnv(gym.Wrapper):
    def __init__(self, **kwargs):
        wp.config.quiet = True
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError('Warp训练必须有CUDA，禁止CPU回退')
        wp.set_device('cuda:0')
        self.module_name = '_warp_env_' + str(id(self))
        spec = importlib.util.spec_from_file_location(self.module_name, ROOT / 'wheelleg_ppo/tools/ppo_env.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module
        spec.loader.exec_module(module)
        self.source = module
        if is_dataclass(kwargs.get('scenario')):
            kwargs['scenario'] = module.Scenario(**asdict(kwargs['scenario']))
        super().__init__(module.WheelLegEnv(**kwargs))
        # 独立命名空间，不影响同进程CPU环境及控制器离线设计调用的mujoco。
        module.mujoco = SimpleNamespace(**{**mujoco.__dict__, 'mj_step': self._physics_step})
        self.physics = None

    def reset(self, **kwargs):
        if is_dataclass(self.env.fixed_scenario):
            self.env.fixed_scenario = self.source.Scenario(**asdict(self.env.fixed_scenario))
        if kwargs.get('options', {}).get('scenario') is not None:
            kwargs['options'] = dict(scenario=self.source.Scenario(**asdict(kwargs['options']['scenario'])))
        self.physics = None  # 先释放旧图/模型，避免reset时显存积累。
        result = self.env.reset(**kwargs)
        self.physics = WarpPhysics(self.env.model, self.env.data)
        return result

    def _physics_step(self, model, data):
        self.physics.step(model, data)
        # float32 GPU时钟累加会污染严格的episode时长核对；统一使用整数物理步时钟。
        data.time = (self.env.steps + 1) * self.env.dt

    def close(self):
        self.physics = None
        sys.modules.pop(self.module_name, None)
        super().close()
