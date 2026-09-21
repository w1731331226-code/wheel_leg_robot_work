"""复用真实状态记录器；GPU选择打包回读，续训保留全局采样步标记。"""
from ppo_env import WheelLegEnv
from dashboard.live_env import LiveRecorder


def make_env(backend,recording=None,environments=8,rollout_steps=250,start_steps=0,**kwargs):
    factory=WheelLegEnv
    if backend=='warp':
        from fast_physics import FastWarpEnv
        factory=FastWarpEnv
    env=factory(**kwargs)
    if not recording:return env
    recorder=LiveRecorder(env,recording,backend,environments,rollout_steps)
    recorder.total_steps=start_steps//environments
    return recorder
