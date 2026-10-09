"""Fresh source-gated8k GPU lifecycle engineering; not scientific training."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
import joint_reference_adapter as adapter
from route_state import RouteState
from execution_history_env import ExecutionHistory
from train_height_comparison import raw_env,equal
from smoke_reward_training import weight_digest,check_agent
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/reference_learning_engineering_v1'


def make_env(config,arm,directory):
    raw=adapter.instrument(raw_env,config['cases'],'virtual6',directory)
    history=ExecutionHistory(RouteState(raw),raw,'H1')
    actions=adapter.JointReferenceActions(history,raw,'M3' if arm=='M_ref3' else 'U6')
    norm=VecNormalize(VecCheckNan(actions,raise_exception=True),**config['normalization'])
    return raw,history,norm


def agent(config,norm,shared=None):
    kwargs=dict(config['ppo']);policy=kwargs.pop('policy_kwargs')
    model=PPO('MlpPolicy',norm,seed=config['engineering_seed'],device='cuda',policy_kwargs=policy,**kwargs)
    with torch.no_grad():
        if shared is not None:
            model.policy.mlp_extractor.load_state_dict(shared['features'])
            model.policy.value_net.load_state_dict(shared['value'])
        model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
        model.policy.log_std.fill_(config['reference_log_std'])
    shared={name:{k:v.detach().cpu().clone() for k,v in module.state_dict().items()} for name,module in
            [('features',model.policy.mlp_extractor),('value',model.policy.value_net)]}
    assert model.observation_space.shape==(481,) and model.device.type=='cuda'
    return model,shared


class Ledger(BaseCallback):
    def __init__(self,norm,directory):
        super().__init__();self.norm=norm;self.directory=directory;self.saved=[];self.episodes=[]
    def save(self):
        if self.model.num_timesteps in self.saved:return
        step=self.model.num_timesteps;prefix=self.directory/f'step_{step}'
        self.model.save(str(prefix));self.norm.save(str(prefix)+'.pkl')
        metrics={name:float(value) for name,value in self.model.logger.name_to_value.items() if name.startswith('train/')}
        assert all(np.isfinite(value) for value in metrics.values())
        atomic_json(self.directory/f'step_{step}.json',dict(policy_samples=step,epochs=self.model._n_updates,
            weights_sha256=weight_digest(self.model),model_sha256=sha(prefix.with_suffix('.zip')),
            normalization_sha256=sha(prefix.with_suffix('.pkl')),training_metrics=metrics,engineering_only=True))
        self.saved.append(step)
    def _on_rollout_start(self):
        if self.model.num_timesteps:self.save()
    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episodes.append({k:v for k,v in info.items() if k!='terminal_observation'})
                assert info['physical_safety_passed'] and info['design_joint_passed'], 'Engineering physical/design violation'
        return True
    def _on_training_end(self):
        self.save()


def verify():
    config=json.loads((OUT/'runtime_config.json').read_text());admit=json.loads((OUT/'source_admission.json').read_text())
    assert admit['verified'] and admit['runtime_config_sha256']==sha(OUT/'runtime_config.json')
    assert all(sha(ROOT/n)==h for n,h in admit['source_sha256'].items())
    assert config['policy_samples_per_arm']==4000 and config['expected_adam_steps']==160
    return config


def run():
    assert not any((OUT/n).exists() for n in ('started.json','completion.json','failure.json'))
    config=verify();torch.set_num_threads(1);reports={};shared=None;world=None;fd_calls=[];fd=mujoco.mjd_transitionFD
    def counted(*args,**kwargs):
        fd_calls.append(float(args[2]));return fd(*args,**kwargs)
    mujoco.mjd_transitionFD=counted
    atomic_json(OUT/'started.json',dict(source_admission_sha256=sha(OUT/'source_admission.json'),total_engineering_policy_samples=8000,formal_PPO_admitted=False))
    try:
        for arm in ('M_ref3','U_ref6'):
            verify();directory=OUT/arm;directory.mkdir(exist_ok=False)
            raw,history,norm=make_env(config,arm,directory);model,shared=agent(config,norm,shared)
            current=[b.numpy().copy() for b in (raw.q0,raw.param,raw.k['gains'],raw.k['reference'])]
            if world is None:world=current
            else:equal(world,current)
            initial=weight_digest(model);np.savez_compressed(directory/'initial_world.npz',q=current[0],param=current[1],gains=current[2],reference=current[3])
            torch.save(dict(policy=model.policy.state_dict(),optimizer=model.policy.optimizer.state_dict(),RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),directory/'initial_policy.pt')
            ledger=Ledger(norm,directory);start=time.perf_counter()
            try:
                model.learn(total_timesteps=4000,callback=ledger)
                assert model.num_timesteps==4000 and model._n_updates==80 and weight_digest(model)!=initial
                assert ledger.saved==list(range(500,4001,500));check_agent(model,160)
                loaded=PPO.load(str(directory/'step_4000.zip'),device='cuda');equal(model.policy.state_dict(),loaded.policy.state_dict())
                equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict());check_agent(loaded,160)
                restored=VecNormalize.load(str(directory/'step_4000.pkl'),norm.venv);restored.training=False;restored.norm_reward=False
                for name in ('obs_rms','ret_rms'):
                    for field in ('mean','var','count'):equal(getattr(getattr(norm,name),field),getattr(getattr(restored,name),field))
                obs=model._last_obs.copy();equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
                np.testing.assert_allclose(norm.obs_rms.count,4010.0001,rtol=0,atol=1e-7)
                np.savez_compressed(directory/'final_history.npz',frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)
                atomic_json(directory/'episodes.json',dict(episodes=ledger.episodes))
                report=dict(verified=True,arm=arm,policy_samples=4000,epochs=80,Adam_steps=160,
                    completed_episodes=len(ledger.episodes),checkpoints=ledger.saved,actual_policy_parameters=sum(p.numel() for p in model.policy.parameters()),
                    learn_checkpoint_wall_seconds=time.perf_counter()-start,initial_weights=initial,reload_model_Adam_RMS_prediction_exact=True,
                    policy_device=str(model.device),physics_device=str(raw.data.qpos.device),engineering_only=True,formal_PPO_admitted=False)
                atomic_json(directory/'verification.json',report);reports[arm]=report
            except BaseException as error:
                model.save(str(directory/'interrupted_model.zip'));norm.save(str(directory/'interrupted_RMS.pkl'))
                atomic_json(directory/'failure.json',dict(error=repr(error),policy_samples=model.num_timesteps,epochs=model._n_updates,implicit_retry=False))
                raise
            finally:norm.close()
        atomic_json(OUT/'completion.json',dict(verified=True,reports=reports,total_policy_samples=8000,baseline_constructor_FD_calls=fd_calls,scientific_evaluations=0,formal_PPO_admitted=False))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),completed_arms=list(reports),FD_calls=fd_calls,implicit_retry=False))
        raise
    finally:mujoco.mjd_transitionFD=fd


if __name__=='__main__':run()
