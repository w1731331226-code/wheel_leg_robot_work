"""Synthetic CPU contract checks; no gate/final simulation or learned probe choice."""
from pathlib import Path
from copy import deepcopy
import sys,json,math
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from train_height_comparison import stage_for,new_agent,summary,assess,METHODS
from ppo_env import Scenario
from dataclasses import asdict
HERE=Path(__file__).resolve().parent

def rejects(call):
    try:call()
    except ValueError:return
    raise AssertionError('invalid contract accepted')

def run():
    assert [stage_for(n,[20,100]) for n in (0,19,20,99,100)]==[1,1,2,2,3]
    p=json.loads((HERE/'protocol_gpu_v3/protocol.json').read_text())
    for method,mode in METHODS.items():
        dim=3 if mode=='diff3' else 6
        def make():
            e=gym.Env();e.observation_space=gym.spaces.Box(-np.inf,np.inf,(38,),dtype=np.float32);e.action_space=gym.spaces.Box(-1,1,(dim,),dtype=np.float32);return e
        env=DummyVecEnv([make for _ in range(10)])
        try:
            agent=new_agent(p,method,1609,env)
            np.testing.assert_array_equal(agent.policy.log_std.detach().cpu().numpy(),np.array(p['initial_log_std'][method],np.float32))
            assert agent.policy.log_std.requires_grad and not agent.policy.optimizer.state_dict()['state']
        finally:env.close()
    scenario={**asdict(Scenario()),'stand_height_m':.3,'terrain':'legacy','relative_attitude':False}
    p=deepcopy(p);p['gate']=[dict(seed=i,scenario=scenario) for i in range(64)]
    p['regression']=[dict(seed=str(i),scenario=scenario) for i in range(28)]
    p['original_regression']=[dict(success=True,velocity_rmse=.01,peak_deg=[1.,1.,1.]) for _ in range(28)]
    def row(case,score):
        goal=2.75;duration=8.;ref=3.5+1.5*goal
        return dict(**case,success=True,reason='completed',physical_steps=16000,duration_s=duration,arrival_s=6.,
            peak_deg=[1.,1.,1.],relative_peak_deg=[1.,1.,1.],rms_deg=[1.,1.,score*math.sqrt(ref/duration)],velocity_rmse=.01,
            stop_distance_m=.1,tail_speed_m_s=.01,task_goal_progress_m=goal,terrain_passed=True,terrain_exit_passed=True,
            terrain_evidence_passed=True,task_contract_version=2,physical_safety_passed=True,design_joint_passed=True,height_rmse_m=.001,final_mean_fk_leg_m=.3,target_leg_m=.3)
    gate={m:{s:[row(c,.8 if m=='M3' else 1.) for c in p['gate']] for s in (['fixed'] if m in ('B0','B1') else ['1609','1610','1611'])} for m in ['B0','B1',*METHODS]}
    regression={s:[row(c,.8) for c in p['regression']] for s in ['1609','1610','1611']}
    assert assess(p,gate,regression)['passed']
    bad=deepcopy(gate);bad['M3']['1609'][0].update(success=False);assert not assess(p,bad,regression)['passed']
    bad=deepcopy(gate);bad['M3']['1609'][0].update(reason='fall',success=False);assert not assess(p,bad,regression)['passed']
    bad=deepcopy(gate);bad['M3']['1609'][0]['velocity_rmse']=.1;assert not assess(p,bad,regression)['passed']
    bad=deepcopy(regression);bad['1609'][0]['peak_deg'][0]=1.2;assert not assess(p,gate,bad)['passed']
    bad=deepcopy(gate);bad.pop('B2');rejects(lambda:assess(p,bad,regression))
    bad=deepcopy(gate);bad['M3']['1609'][0]['scenario']={**scenario,'mass':7.5};rejects(lambda:assess(p,bad,regression))
    bad=deepcopy(gate);bad['M3']['1609'][0]['physical_safety_passed']=False;rejects(lambda:assess(p,bad,regression))
    for m,runs in gate.items():
        for s,rows in runs.items():
            for r in rows:r['rms_deg'][2]=0.
    assert not assess(p,gate,regression)['passed']
    (HERE/'contract_check.json').write_text(json.dumps(dict(passed=True,synthetic_only=True,stage_boundaries=True,
        initial_vectors_applied_and_trainable=True,effect_nonregression_identity_and_zero_reference=True,gate_simulated=False),indent=2)+'\n')
    print('PASS CPU contract: applied std vectors, stage boundaries, effect/paired/legacy/failure/identity/zero-reference guards; no gate simulation')

if __name__=='__main__':torch.set_num_threads(1);run()
