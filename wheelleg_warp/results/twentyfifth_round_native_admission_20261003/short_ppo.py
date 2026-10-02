"""Real standard PPO updates/optimizer-normalization restore/eval; probe only."""
from pathlib import Path
from dataclasses import replace
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback
from native.environment import NativeEnv
from native.terrain import sample_height_terrain_115
from probe_height_115_margin import cases
from training_contract import source_hashes,checkpoint_hashes
OUT=Path(__file__).resolve().parent


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


class Episodes(BaseCallback):
    def __init__(self):super().__init__();self.rows=[]
    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:self.rows.append({k:v for k,v in info.items() if k!='terminal_observation'})
        return True


def run(mode):
    output=OUT/f'ppo_{mode}';output.mkdir()
    scenes=[cases()[4],replace(cases()[5],mass=7.5,mu_l=.6,mu_r=1.,drive_difference=.03,delay_ms=10.),
        replace(cases()[4],stand_height_m=.38),sample_height_terrain_115(1154043,3,'development')]
    def make():return NativeEnv.height115_candidate(n=4,scenario=scenes,residual_mode=mode,shared_reference=True)
    raw=make();vec=VecNormalize(VecCheckNan(raw,raise_exception=True),norm_reward=False)
    restored=None;callback=Episodes()
    try:
        torch.set_num_threads(1)
        agent=PPO('MlpPolicy',vec,n_steps=50,batch_size=100,n_epochs=1,seed=1155000,
            policy_kwargs=dict(net_arch=[32,32]),device='cpu',verbose=0)
        before={k:v.clone() for k,v in agent.policy.state_dict().items()}
        protocol=dict(kind='engineering_short_update_not_formal_training',mode=mode,observation=raw.observation_spec,
            baseline_version=raw.baseline_version,design_contract='active-1p4-v1',
            total_policy_steps=2000,n_steps=50,n_epochs=1,optimizer='standard SB3 PPO/Adam',
            final_holdout_opened=False,probe_checkpoint_promoted=False,source_sha256=source_hashes(__file__))
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        agent.learn(total_timesteps=2000,callback=callback)
        assert agent.num_timesteps==2000 and agent._n_updates==10
        assert any(not torch.equal(before[k],v) for k,v in agent.policy.state_dict().items())
        assert agent.policy.optimizer.state_dict()['state']
        assert callback.rows, 'No actual task termination exercised'
        for info in callback.rows:
            assert info['shared_reference_contract']=='public-region-v2-state-phase-vmc'
            assert info['design_joint_contract']=='active-1p4-v1'
            assert not info['TimeLimit.truncated']
            assert not info['success'] or (info['physical_safety_passed'] and info['design_joint_passed'])
        prefix=output/'probe';agent.save(prefix);vec.save(str(prefix)+'.pkl')
        saved_optimizer=agent.policy.optimizer.state_dict();saved_weights=agent.policy.state_dict()
        raw_new=make();restored=VecNormalize.load(str(prefix)+'.pkl',VecCheckNan(raw_new,raise_exception=True))
        loaded=PPO.load(str(prefix)+'.zip',env=restored,device='cpu')
        equal(saved_weights,loaded.policy.state_dict());equal(saved_optimizer,loaded.policy.optimizer.state_dict())
        equal(vec.obs_rms.mean,restored.obs_rms.mean);equal(vec.obs_rms.var,restored.obs_rms.var);equal(vec.obs_rms.count,restored.obs_rms.count)
        assert loaded.num_timesteps==2000 and loaded._n_updates==10
        fixture=vec.normalize_obs(raw.obs0.numpy() if raw.obs0.shape[1]==38 else np.c_[raw.obs0.numpy(),np.zeros((4,6),np.float32)])
        np.testing.assert_array_equal(agent.predict(fixture,deterministic=True)[0],loaded.predict(fixture,deterministic=True)[0])
        loaded.learn(total_timesteps=200,reset_num_timesteps=False)
        assert loaded.num_timesteps==2200 and loaded._n_updates==11
        restored.training=False;restored.norm_reward=False
        initial=restored.obs_rms.mean.copy();obs=restored.reset();evaluation=[None]*4
        for _ in range(1000):
            action,_=loaded.predict(obs,deterministic=True);obs,_,done,infos=restored.step(action)
            for w in np.flatnonzero(done):
                if evaluation[w] is None:evaluation[w]={k:v for k,v in infos[w].items() if k!='terminal_observation'}
            if all(e is not None for e in evaluation):break
        assert all(e is not None for e in evaluation)
        np.testing.assert_array_equal(initial,restored.obs_rms.mean)
        result=dict(passed=True,mode=mode,policy_steps_before_restore=2000,updates_before_restore=10,
            resumed_policy_steps=2200,resumed_updates=11,weights_changed=True,
            weights_optimizer_normalization_restored_exactly=True,deterministic_prediction_roundtrip=True,
            physical_environment_state_restored=False,normalization_frozen_for_evaluation=True,
            collection_episodes=callback.rows,evaluation=evaluation,
            learning_performed=True,formal_training=False,checkpoint_promoted=False,
            checkpoint_sha256=checkpoint_hashes(prefix),source_sha256=source_hashes(__file__))
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PASS REAL PPO',mode,'2000steps10updates, restore exact, resume2200steps11updates; evaluation success',sum(e['success'] for e in evaluation),'/4',flush=True)
    finally:
        vec.close()
        if restored is not None:restored.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('diff3','virtual6'));run(p.parse_args().mode)
