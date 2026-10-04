"""Four registered route-information/action-space arms on the frozen engine."""
import copy
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper,VecNormalize,VecCheckNan
from route_state import RouteState
from train_height_comparison import CurriculumEnv,raw_env,new_agent,summary
import wheelleg_sim as sim

ARMS={'M3-zero':dict(native='diff3',feature='zero',dimensions=3,initial='M3'),
      'M3-route':dict(native='diff3',feature='route',dimensions=3,initial='M3'),
      'V6-route':dict(native='virtual6',feature='route',dimensions=6,initial='B2-V'),
      'L2-route':dict(native='diff3',feature='route',dimensions=2,initial='L2')}


class LegOnly(VecEnvWrapper):
    """A genuinely 2D policy; wheel requests are absent, not masked old outputs."""
    def __init__(self,venv):
        if venv.action_space.shape!=(3,):raise ValueError('LegOnly needs native diff3')
        super().__init__(venv,action_space=Box(-1,1,(2,),dtype=np.float32))
    def reset(self):return self.venv.reset()
    def step_async(self,actions):
        a=np.asarray(actions,np.float32)
        if a.shape!=(self.num_envs,2) or not np.isfinite(a).all() or np.any(abs(a)>1):raise ValueError('Invalid two-leg request')
        self.venv.step_async(np.c_[a,np.zeros(self.num_envs,np.float32)])
    def step_wait(self):return self.venv.step_wait()


def self_check():
    from test_route_state import ScriptedEnv
    raw=ScriptedEnv();wrapped=LegOnly(RouteState(raw));assert wrapped.action_space.shape==(2,)
    assert wrapped.reset().shape==(2,39)
    a=np.array([[.2,-.3],[-.2,.3]],np.float32);wrapped.step(a)
    np.testing.assert_array_equal(raw.actions[:,:2],a);np.testing.assert_array_equal(raw.actions[:,2],0)
    wrapped.close()


def training_env(config,arm,seed,n):
    spec=ARMS[arm];raw=CurriculumEnv(config,spec['native'],seed,n)
    route=RouteState(raw,spec['feature']);action=LegOnly(route) if spec['dimensions']==2 else route
    env=VecNormalize(VecCheckNan(action,raise_exception=True),**config['normalization'])
    return raw,route,env


def agent(config,arm,seed,env):
    p=copy.deepcopy(config);p['initial_log_std']['L2']=p['initial_log_std']['M3'][:2]
    return new_agent(p,ARMS[arm]['initial'],seed,env)


def evaluate(cases,arm,model=None,normalization=None,classical=None):
    from probe_route_feedback import actions
    spec=ARMS[arm];raw=raw_env(cases,spec['native']);route=RouteState(raw,spec['feature'])
    env=LegOnly(route) if spec['dimensions']==2 else route
    if model:
        env=VecNormalize.load(str(normalization),VecCheckNan(env,raise_exception=True));env.training=False;env.norm_reward=False
        rms=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count)
    rows=[None]*len(cases)
    try:
        obs=env.reset();ids=raw.ids.numpy();deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
        for _ in range(deadline):
            a=model.predict(obs,deterministic=True)[0] if model else actions(obs,classical['candidate'],classical['arm'])[0]
            obs,_,done,infos=env.step(a);stopped=raw.stopped_q.numpy() if done.any() else None
            for w in np.flatnonzero(done):
                if rows[w] is None:
                    length=float(np.mean([sim.fk_joints(float(stopped[w,ids[2*s]]),float(stopped[w,ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    rows[w]=dict(**cases[w],**{k:v for k,v in infos[w].items() if k!='terminal_observation'},final_mean_fk_leg_m=length)
            if all(r is not None for r in rows):break
        assert all(r is not None for r in rows)
        if model:
            np.testing.assert_array_equal(rms[0],env.obs_rms.mean);np.testing.assert_array_equal(rms[1],env.obs_rms.var);assert rms[2]==env.obs_rms.count
        return dict(summary=summary(rows),physical=sum(r['physical_safety_passed'] for r in rows),design=sum(r['design_joint_passed'] for r in rows),runs=rows)
    finally:env.close()
