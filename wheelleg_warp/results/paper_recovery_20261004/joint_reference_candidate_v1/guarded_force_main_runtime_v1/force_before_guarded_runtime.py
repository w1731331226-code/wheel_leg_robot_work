"""Original differential virtual-wrench actions; nominal reference stays owned by control."""
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper,VecNormalize,VecCheckNan


def embed(actions,arm,n):
    a=np.asarray(actions,dtype=np.float32)
    if arm not in ('D3','V6') or a.shape!=(n,3 if arm=='D3' else 6) or not np.isfinite(a).all() or np.any(abs(a)>1):
        raise ValueError('Finite bounded D3/V6 force-space actions required')
    if arm=='V6':return a
    return np.column_stack((a[:,0],-a[:,0],a[:,1],-a[:,1],a[:,2],-a[:,2]))


class ForceActions(VecEnvWrapper):
    def __init__(self,venv,arm):
        if arm not in ('D3','V6') or venv.action_space.shape!=(6,):raise ValueError('Force wrapper needs virtual6 runtime')
        self.arm=arm
        super().__init__(venv,action_space=Box(-1,1,(3 if arm=='D3' else 6,),dtype=np.float32))
    def reset(self):return self.venv.reset()
    def step_async(self,actions):self.venv.step_async(embed(actions,self.arm,self.num_envs))
    def step_wait(self):return self.venv.step_wait()


def make_env(protocol,arm,seed):
    if arm not in ('D3','V6') or not 1<=protocol['worlds']<=100:raise ValueError('Registered force arm and1…100 worlds required')
    # Same fixed-reference construction as execution_input_study, before normalization.
    import execution_history_engineering as engine
    from execution_history_env import ExecutionHistory
    from route_state import RouteState
    factory=engine.base.raw_env
    engine.base.raw_env=lambda rows,mode:engine.collector.instrument(lambda a,b:engine.parking.instrument(factory,a,b),rows,mode)
    try:curriculum=engine.base.CurriculumEnv(protocol,'virtual6',seed,protocol['worlds'])
    finally:engine.base.raw_env=factory
    raw=curriculum.venv;history=ExecutionHistory(RouteState(curriculum),raw,'H1')
    norm=VecNormalize(VecCheckNan(ForceActions(history,arm),raise_exception=True),**protocol['normalization'])
    return curriculum,raw,history,norm


class PriorActor:
    """Fresh zero-head Gaussian prior for regression, not a trained control policy."""
    def __init__(self,model,arm):self.model=model;self.arm=arm;self.policy=model.policy
    @property
    def num_timesteps(self):return self.model.num_timesteps
    @property
    def _n_updates(self):return self.model._n_updates
    def predict(self,obs,deterministic=True):
        action,state=self.model.predict(obs,deterministic=False)
        return embed(action,self.arm,len(obs)),state


def self_check():
    from test_reference_learning_policy import NoStep
    a=np.random.default_rng(32731).uniform(-1,1,(128,3)).astype(np.float32);b=embed(a,'D3',len(a))
    np.testing.assert_array_equal(b[:,::2],a);np.testing.assert_array_equal(b[:,1::2],-a)
    np.testing.assert_array_equal(b[:,::2]+b[:,1::2],0)
    for value in (np.full((2,3),np.nan),np.full((2,3),1.1),np.zeros((2,6))):
        try:embed(value,'D3',2)
        except ValueError:pass
        else:raise AssertionError('Invalid force request admitted')
    class Capture(NoStep):
        def step_async(self,actions):self.last=actions.copy()
    raw=Capture(6);wrapped=ForceActions(raw,'D3');wrapped.step_async(a[:10])
    np.testing.assert_array_equal(raw.last,b[:10]);assert wrapped.action_space.shape==(3,)
    np.testing.assert_array_equal(embed(b,'V6',128),b)
    print('PASS327 original F/-F,H/-H,Y/-Y embed,virtualcommoninputzero,wrapper6delivery/3space/boundaries;0physics')


if __name__=='__main__':self_check()
