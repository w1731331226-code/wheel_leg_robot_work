"""One actual39 RMS, with public reference cache before normalization."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools'))
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper
from nominal_packet_reference import reference


class RawReferenceCache(VecEnvWrapper):
    def __init__(self, venv):
        if venv.observation_space.shape != (39,):
            raise ValueError('Reference cache requires raw39 RouteState')
        super().__init__(venv)
        self.current_reference = None
        self.terminal_references = {}

    def reset(self):
        obs = self.venv.reset()
        self.current_reference = reference(obs)
        self.terminal_references.clear()
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, infos = self.venv.step_wait()
        self.current_reference = reference(obs)
        self.terminal_references = {w: reference(np.asarray(infos[w]['terminal_observation'])[None])[0]
                                    for w in np.flatnonzero(done)}
        return obs, reward, done, infos


class NormalizedReferencePair(VecEnvWrapper):
    def __init__(self, normalizer, cache):
        if normalizer.observation_space.shape != (39,) or cache.num_envs != normalizer.num_envs:
            raise ValueError('Pair requires one39 normalizer and matching cache')
        self.normalizer, self.cache = normalizer, cache
        super().__init__(normalizer, observation_space=Box(-np.inf, np.inf, (78,), dtype=np.float32))

    def _pair(self, obs):
        if self.cache.current_reference is None:
            raise RuntimeError('Reference pair requires reset')
        normalized_reference = self.normalizer.normalize_obs(self.cache.current_reference)
        return np.concatenate([obs, normalized_reference], axis=1).astype(np.float32)

    def reset(self):
        return self._pair(self.venv.reset())

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, infos = self.venv.step_wait()
        paired = self._pair(obs)
        for w in np.flatnonzero(done):
            info = dict(infos[w])
            terminal_ref = self.normalizer.normalize_obs(self.cache.terminal_references[w][None])[0]
            info['terminal_observation'] = np.r_[info['terminal_observation'], terminal_ref].astype(np.float32)
            infos[w] = info
        return paired, reward, done, infos
