"""Reuse current reference kernels/history and original episode-end curriculum."""
import numpy as np
import warp as wp
from stable_baselines3.common.vec_env import VecCheckNan,VecNormalize
import train_height_comparison as base
import joint_reference_adapter as adapter
from route_state import RouteState
from execution_history_env import ExecutionHistory


class ReferenceCurriculum(base.CurriculumEnv):
    def step_wait(self):
        result=super().step_wait()
        # Parent changes only completed worlds; actor/history sees the returned new reset packet.
        if result[2].any():
            b=self.venv._joint_buffers;b[-1].assign(result[2].astype(np.int32))
            wp.launch(adapter.clear,self.num_envs,[b[-1],b[0],b[1],b[2],b[3],b[4],b[5],b[7]])
        return result


def make_env(protocol,arm,seed,worlds=None):
    n=protocol['worlds'] if worlds is None else worlds
    if arm not in ('M_ref3','U_ref6') or not 1<=n<=100:
        raise ValueError('Registered reference learning arm and1…100 worlds required')
    factory=base.raw_env
    base.raw_env=lambda cases,mode:adapter.instrument(factory,cases,mode,dense=False)
    try:curriculum=ReferenceCurriculum(protocol,'virtual6',seed,n)
    finally:base.raw_env=factory
    raw=curriculum.venv;history=ExecutionHistory(RouteState(curriculum),raw,'H1')
    actions=adapter.JointReferenceActions(history,raw,'M3' if arm=='M_ref3' else 'U6')
    norm=VecNormalize(VecCheckNan(actions,raise_exception=True),**protocol['normalization'])
    return curriculum,raw,history,norm
