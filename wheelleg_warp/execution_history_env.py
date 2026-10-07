"""Shared481D history; H0 masks only the54 total-command coordinates."""
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper
from execution_input_features import SCALES

COMMAND_INDICES = np.array([390+8*i+j for i in range(9) for j in range(6)])


class ExecutionHistory(VecEnvWrapper):
    def __init__(self, venv, collector, arm):
        if venv.observation_space.shape != (39,) or arm not in ('H0', 'H1'):
            raise ValueError('History requires raw39 route packets andH0/H1')
        if collector.num_envs != venv.num_envs or not hasattr(collector, '_execution_intervals'):
            raise ValueError('Matching instrumented command collector required')
        self.collector, self.arm = collector, arm
        super().__init__(venv, observation_space=Box(-np.inf, np.inf, (481,), dtype=np.float32))
        n = self.num_envs
        self.frames = np.zeros((n, 10, 39), np.float32)
        self.inputs = np.zeros((n, 9, 8), np.float32)
        self.elapsed = np.zeros((n, 9), np.float32)
        self.valid = np.zeros((n, 10), np.float32)
        self.ready = False

    def _reset_rows(self, rows, packet):
        self.frames[rows] = packet[rows, None, :]
        self.inputs[rows] = 0; self.elapsed[rows] = 0; self.valid[rows] = 0
        self.valid[rows, -1] = 1

    def encode(self, arm=None):
        arm = self.arm if arm is None else arm
        if arm not in ('H0', 'H1'): raise ValueError('Unknown history arm')
        n = self.num_envs
        value = np.concatenate([self.frames.reshape(n, 390), self.inputs.reshape(n, 72),
                                self.elapsed, self.valid], axis=1)
        if arm == 'H0': value[:, COMMAND_INDICES] = 0
        return value

    def reset(self):
        packet = self.venv.reset()
        self._reset_rows(np.ones(self.num_envs, bool), packet); self.ready = True
        return self.encode()

    def step_async(self, actions): self.venv.step_async(actions)

    def step_wait(self):
        if not self.ready: raise RuntimeError('Reset history before stepping')
        packet, reward, done, infos = self.venv.step_wait()
        data = self.collector._execution_intervals
        if data is None or not np.array_equal(data['terminal_worlds'], done):
            raise RuntimeError('Collector interval belongs to a different step')
        endpoints = packet.copy()
        for w in np.flatnonzero(done): endpoints[w] = infos[w]['terminal_observation']
        dt = np.asarray(data['sensor_elapsed_s']); impulse = np.asarray(data['command_integral'])
        if dt.shape != (self.num_envs,) or impulse.shape != (self.num_envs, 6) or not np.isfinite(impulse).all():
            raise ValueError('Malformed collector interval')
        if not np.isfinite(dt).all() or np.any(dt<0) or np.any(dt>.02):
            raise ValueError('Invalid delivered sensor duration')
        zero = dt == 0
        if np.any(impulse[zero] != 0) or np.any(endpoints[zero, 20:22] != self.frames[zero, -1, 20:22]):
            raise ValueError('Repeated sensor timestamp changed its rotor sample')
        mean = np.divide(impulse, dt[:, None], out=np.zeros_like(impulse), where=dt[:, None]>0)
        self.frames[:, :-1] = self.frames[:, 1:].copy(); self.frames[:, -1] = endpoints
        self.inputs[:, :-1] = self.inputs[:, 1:].copy(); self.inputs[:, -1] = 0
        self.inputs[:, -1, :6] = mean/SCALES  # Proxy slots6:8 remain zero in BOTH arms.
        self.elapsed[:, :-1] = self.elapsed[:, 1:].copy(); self.elapsed[:, -1] = dt/.02
        self.valid[:, :-1] = self.valid[:, 1:].copy(); self.valid[:, -1] = 1
        terminal = self.encode()
        for w in np.flatnonzero(done):
            infos[w] = dict(infos[w]); infos[w]['terminal_observation'] = terminal[w].copy()
        self._reset_rows(done, packet)
        obs = self.encode()
        if not np.isfinite(obs).all() or not np.isfinite(terminal).all(): raise ValueError('Nonfinite history')
        return obs, reward, done, infos
