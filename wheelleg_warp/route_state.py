"""Known-zero-origin causal route memory outside the frozen physical engine.

Both modes have 39 inputs; `zero` removes only the route feature. All learning
and classical comparisons must declare which mode they receive. Ideal body
velocities remain the original simulation assumption, not wheel odometry.
"""
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper
from probe_packet_odometry import PacketOdometry, lateral_velocity


class RouteState(VecEnvWrapper):
    def __init__(self, venv, mode='route'):
        if mode not in ('route', 'zero') or venv.observation_space.shape != (38,):
            raise ValueError('RouteState requires a 38D packet and route/zero mode')
        self.mode=mode
        self.spec=dict(version='packet_route_y_v1',dimension=39,base_packet_slice=[0,38],
            route_feature_index=38,mode=mode,actor_dt_s=.02,initial_y_m=0.,
            inputs='delayed packet yaw[2], body vx[6], body vy[7]; causal trapezoidal memory',
            partial_observability=True,unknown_initial_offset_observable=False,
            ground_truth_position=False,hidden_delay=False,future_terrain=False)
        space=venv.observation_space
        super().__init__(venv,observation_space=Box(np.r_[space.low,-np.inf],np.r_[space.high,np.inf],dtype=np.float32))
        self.clock=np.zeros(self.num_envs,np.float64)
        self.odometry=None

    def _append(self, packet, y):
        value=y if self.mode=='route' else np.zeros_like(y)
        return np.concatenate([packet,np.asarray(value,np.float32)[:,None]],axis=1)

    def reset(self):
        packet=self.venv.reset()
        self.clock.fill(0)
        self.odometry=PacketOdometry(packet)
        return self._append(packet,self.odometry.y)

    def step_wait(self):
        if self.odometry is None:raise RuntimeError('reset required before stepping')
        packet,reward,done,infos=self.venv.step_wait()
        endpoints=packet.copy();dt=np.full(self.num_envs,.02)
        for w in np.flatnonzero(done):
            endpoints[w]=infos[w]['terminal_observation']
            dt[w]=infos[w]['duration_s']-self.clock[w]
        terminal_y=self.odometry.update(endpoints,dt)
        self.clock+=dt
        for w in np.flatnonzero(done):
            info=dict(infos[w])
            info['terminal_observation']=self._append(endpoints[w:w+1],terminal_y[w:w+1])[0]
            infos[w]=info
            # The underlying environment already returned the new episode's
            # reset packet, possibly with a freshly switched curriculum bank.
            self.odometry.y[w]=0
            self.odometry.last_velocity[w]=lateral_velocity(packet[w:w+1])[0]
            self.clock[w]=0
        return self._append(packet,self.odometry.y),reward,done,infos
