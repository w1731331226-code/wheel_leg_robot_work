"""真实训练环境0的只读状态采集；其他环境不录像，不增加物理步。"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np


def atomic_json(path, obj):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)


class LiveRecorder(gym.Wrapper):
    def __init__(self, env, directory, backend, environments=8, rollout_steps=250):
        super().__init__(env)
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.backend=backend;self.environments=environments;self.rollout_steps=rollout_steps
        self.total_steps=0;self.episode=0;self.rows=[];self.folder=None

    def reset(self, **kwargs):
        result=self.env.reset(**kwargs)
        self.episode+=1;self.rows=[];self.cumulative_reward=0.
        self.folder=self.directory/'episodes'/f'{self.episode:06d}'
        self.folder.mkdir(parents=True,exist_ok=False)
        base=self.env.unwrapped
        self.metadata=dict(kind='actual_training_frames',backend=self.backend,environment_index=0,
            environments=self.environments,episode=self.episode,scenario=asdict(base.scenario),
            initial_qpos=base.data.qpos.tolist(), initial_qvel=base.data.qvel.tolist(),
            reset_seed=kwargs.get('seed'),source_run=str(self.directory.parent),recorded_policy_hz=50,
            started=time.time(),status='recording',note='真实训练进程环境0，策略每20ms一步；不代表全部8环境逐物理步记录')
        atomic_json(self.folder/'metadata.json',self.metadata)
        self._snapshot(0.)
        return result

    def _snapshot(self,reward):
        data=self.env.unwrapped.data
        temporary=self.directory/'latest.tmp'
        with temporary.open('wb') as f:
            np.savez(f,qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,frame=self.total_steps,episode=self.episode)
        temporary.replace(self.directory/'latest.npz')
        atomic_json(self.directory/'latest.json',dict(**self.metadata,wall_time=time.time(),
            simulation_seconds=float(data.time),sample_steps=self.total_steps*self.environments,
            policy_update_steps=max(0,(self.total_steps-1)//self.rollout_steps)*self.rollout_steps*self.environments,
            cumulative_reward=self.cumulative_reward,episode_directory=str(self.folder),
            frame=self.total_steps,reward=float(reward)))

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        self.total_steps+=1;self.cumulative_reward+=reward;d=self.env.unwrapped.data
        self.rows.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),np.asarray(action).copy(),obs.copy(),float(reward)))
        self._snapshot(reward)
        if terminated or truncated:
            with (self.folder/'trajectory.tmp').open('wb') as f:
                np.savez_compressed(f,**{name:np.asarray([row[i] for row in self.rows]) for i,name in
                    enumerate(('time','qpos','qvel','ctrl','action','observation','reward'))})
            (self.folder/'trajectory.tmp').replace(self.folder/'trajectory.npz')
            self.metadata.update(status='completed',frames=len(self.rows),finished=time.time(),metrics=info)
            atomic_json(self.folder/'metadata.json',self.metadata)
        return obs,reward,terminated,truncated,info


def make_env(backend, recording=None, environments=8, rollout_steps=250, **kwargs):
    from ppo_env import WheelLegEnv
    factory=WheelLegEnv
    if backend=='warp':
        from gpu_env import WarpEnv
        factory=WarpEnv
    env=factory(**kwargs)
    return LiveRecorder(env,recording,backend,environments,rollout_steps) if recording else env
