"""Optional reward-only wrapper; original physics and observations stay unchanged."""
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper

class FailurePotential(VecEnvWrapper):
    def __init__(self,venv,gamma=.99,beta=10.):
        super().__init__(venv)
        if not (np.isfinite(gamma) and 0<gamma<1 and np.isfinite(beta) and beta>0):
            raise ValueError('Invalid potential discount/amplitude')
        self.raw=venv.unwrapped
        if self.raw.state.shape[1]<39 or self.observation_space.shape!=(38,):
            raise ValueError('Potential requires the frozen38-observation physical/design contract')
        self.gamma=float(gamma);self.beta=float(beta);self.potential=np.zeros(self.num_envs)

    def reset(self):
        self.potential.fill(0)
        return self.venv.reset()

    def step_wait(self):
        obs,reward,done,infos=self.venv.step_wait()
        state=self.raw.state.numpy();param=self.raw.param.numpy()
        peaks=np.where(param[:,7:8]>0,state[:,21:24],state[:,8:11])
        breach=(state[:,38]<0)|(np.minimum(state[:,31],state[:,32])<param[:,15])
        breach|=(state[:,33]<0)|(np.max(state[:,35:37],axis=1)>1e-6)
        breach|=(np.max(peaks,axis=1)>np.deg2rad(5))|(np.max(state[:,8:10],axis=1)>np.deg2rad(10))
        # The native evaluator keeps historical extrema; done rows already reset.
        # Both success/failure terminals must have zero potential for telescoping.
        following=np.where(breach&~done,-self.beta,0.)
        self.last_original_reward=reward.copy();self.last_before=self.potential.copy()
        self.last_after=following.copy();self.last_shaping=self.gamma*following-self.potential
        self.last_exact_reward=reward.astype(np.float64)+self.last_shaping
        shaped=self.last_exact_reward.astype(np.float32)
        if not np.isfinite(shaped).all():raise FloatingPointError('Nonfinite shaped reward')
        self.potential=following
        return obs,shaped,done,infos
