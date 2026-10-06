"""Source-qualified continuous engineer; main qualification is a separate admission."""
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
import light_phase_reference as light
from nominal_reference_env import RawReferenceCache, NormalizedReferencePair
from nominal_mean_policy import NominalMeanPolicy
from route_state import RouteState
from route_pilot import RouteLedger
from train_height_comparison import equal
from smoke_reward_training import weight_digest, check_agent
from dashboard.live_env import atomic_json
from review_yaw_sector import ROOT, sha
import stable_baselines3.common.policies as policies
import stable_baselines3.common.distributions as distributions
import stable_baselines3.common.torch_layers as layers

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_mean_matched_learning_v1'


def make_env(p, seed, n, milestones):
    raw = light.curriculum(p, 'virtual6', seed, n, milestones)
    route = RouteState(raw, 'route'); cache = RawReferenceCache(route)
    norm = VecNormalize(VecCheckNan(cache, raise_exception=True), **p['normalization'])
    return raw, route, cache, norm, NormalizedReferencePair(norm, cache)


def new_agent(p, arm, seed, env):
    kwargs = dict(p['ppo']); policy_kwargs = dict(kwargs.pop('policy_kwargs'))
    policy_kwargs.update(eta=p['arms'][arm]['eta'], net_arch=[64, 64])
    model = PPO(NominalMeanPolicy, env, seed=seed, device='cuda', policy_kwargs=policy_kwargs, **kwargs)
    with torch.no_grad():
        model.policy.log_std.copy_(torch.tensor(p['initial_log_std'], device=model.device))
    assert model.observation_space.shape == (78,) and model.action_space.shape == (6,)
    assert model.policy.action_net.bias is None and not model.policy.action_net.weight.any()
    return model


class MatchedLedger(RouteLedger):
    def __init__(self, raw, route, cache, norm, directory, interval):
        super().__init__(raw, route, directory, interval)
        self.cache, self.norm = cache, norm

    def save(self):
        before = (self.cache.current_reference.copy(),
                  {w:r.copy() for w,r in self.cache.terminal_references.items()},
                  self.norm.obs_rms.mean.copy(), self.norm.obs_rms.var.copy(), self.norm.obs_rms.count,
                  [a.numpy().copy() for a in self.raw.venv._light_phase_buffers],
                  self.raw.stages.copy(), self.model._last_obs.copy())
        super().save()
        after = (self.cache.current_reference, self.cache.terminal_references,
                 self.norm.obs_rms.mean, self.norm.obs_rms.var, self.norm.obs_rms.count,
                 [a.numpy() for a in self.raw.venv._light_phase_buffers], self.raw.stages, self.model._last_obs)
        equal(before, after)
        record = self.saved[-1]
        record.update(paired_reference_RMS_smallbuffers_stages_lastobs_unchanged_by_save=True,
                      normalization_dimensions=39, stored_observation_dimensions=78, policy_eta=self.model.policy.eta)
        atomic_json(self.directory/f'step_{self.model.num_timesteps}.json', record)


def verify_engineering():
    p = json.loads((OUT/'proposal.json').read_text()); c = json.loads((OUT/'engineering_contract.json').read_text())
    source = json.loads((OUT/'policy_source_contract.json').read_text())
    assert source['verified'] and source['static_policy_stage_admitted'] and source['separate_engineering_budget_admitted'] == 24000
    assert c['proposal_sha256'] == source['proposal_sha256'] == sha(OUT/'proposal.json')
    assert c['policy_source_contract_sha256'] == sha(OUT/'policy_source_contract.json')
    assert c['source_sha256'] == sha(__file__)
    assert all(sha(ROOT/n) == s for n,s in c['input_sha256'].items())
    assert all(sha(ROOT/n) == s for n,s in source['source_sha256'].items())
    for name,module in [('policies',policies),('distributions',distributions),('torch_layers',layers)]:
        assert source['installed_SB3_sha256'][name] == sha(Path(module.__file__))
    assert c['total_policy_steps'] == p['engineering']['total_policy_steps'] == 24000
    assert c['arms'] == list(p['arms']) and c['policy_steps_each'] == 12000 and not c['main_admitted']
    return p, c


def freeze_engineering():
    assert not (OUT/'engineering_contract.json').exists()
    p = json.loads((OUT/'proposal.json').read_text())
    inputs = {str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in
        ('proposal.json','environment_source_contract.json','round201_environment_unit.json',
         'policy_source_contract.json','round202_policy_unit.json')}
    for file in ('route_pilot.py','reward_pilot_train.py','smoke_reward_training.py','train_height_comparison.py'):
        inputs['wheelleg_warp/'+file] = sha(ROOT/'wheelleg_warp'/file)
    atomic_json(OUT/'engineering_contract.json', dict(source_sha256=sha(__file__), input_sha256=inputs,
        proposal_sha256=sha(OUT/'proposal.json'), policy_source_contract_sha256=sha(OUT/'policy_source_contract.json'),
        arms=list(p['arms']), total_policy_steps=24000, policy_steps_each=12000, worlds=10, seed=1991,
        worker='wheelleg-nominal-mean-engineering-v1.service; exclusive project-write.lock',
        main_admitted=False, scope='Two fresh continuousengineering runs; no scientificscore/modelselection orwarmstart into1.2M main. No implicit resume/extra budget.'))
    verify_engineering(); print('FROZEN24k engineering only;0 training steps consumed;main not admitted', flush=True)


def run_engineering_arm(arm):
    p,c = verify_engineering(); directory = OUT/'engineering'/arm; directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    seed = c['seed']; raw,route,cache,norm,env = make_env(p, seed, c['worlds'], p['engineering']['milestones'])
    model = new_agent(p, arm, seed, env); initial = weight_digest(model)
    native = raw.venv; world = hashlib.sha256()
    for a in (native.q0, native.param, native.k['gains'], native.k['reference']): world.update(a.numpy().tobytes())
    for peer in c['arms']:
        file = OUT/'engineering'/peer/'initialization.json'
        if file.exists():
            other = json.loads(file.read_text()); assert initial == other['initial_weight_sha256'] and world.hexdigest() == other['initial_world_sha256']
    atomic_json(directory/'initialization.json', dict(arm=arm, seed=seed, initial_weight_sha256=initial,
        initial_world_sha256=world.hexdigest(), actor_storage=78, effective_current_reference=39, critic=39,
        initial_mean_zero=True, eta=p['arms'][arm]['eta'], engineering_only=True))
    cb = MatchedLedger(raw,route,cache,norm,directory,p['engineering']['checkpoints_every'])
    try:
        start = time.perf_counter(); model.learn(total_timesteps=12000, callback=cb); cb.save(); wall = time.perf_counter()-start
        assert model.num_timesteps == 12000 and model._n_updates == 240 and weight_digest(model) != initial
        adam = check_agent(model,480)
        assert [r['policy_steps'] for r in cb.saved] == list(range(2000,12001,2000))
        assert cb.rows and {r['stage'] for r in raw.transition_log} == {2,3} and all(r['actual_episode_end'] for r in raw.transition_log)
        assert np.any(raw.stages == 3) and native.data.qpos.device.is_cuda
        assert norm.obs_rms.mean.shape == (39,) and np.isfinite(norm.obs_rms.mean).all() and np.isfinite(norm.obs_rms.var).all()
        np.testing.assert_allclose(norm.obs_rms.count,12000+c['worlds']+1e-4,atol=1e-7,rtol=0)
        prefix = directory/'step_12000'; loaded = PPO.load(prefix.with_suffix('.zip'),device='cuda')
        equal(model.policy.state_dict(),loaded.policy.state_dict()); equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict());check_agent(loaded,480)
        assert loaded.policy.eta == model.policy.eta
        with prefix.with_suffix('.pkl').open('rb') as f: restored = pickle.load(f)
        equal(norm.obs_rms.mean,restored.obs_rms.mean);equal(norm.obs_rms.var,restored.obs_rms.var);equal(norm.obs_rms.count,restored.obs_rms.count)
        equal(norm.ret_rms.mean,restored.ret_rms.mean);equal(norm.ret_rms.var,restored.ret_rms.var);equal(norm.ret_rms.count,restored.ret_rms.count)
        atomic_json(directory/'episodes.json',dict(episodes=cb.rows,curriculum_transitions=raw.transition_log))
        atomic_json(directory/'verification.json',dict(verified=True,arm=arm,policy_steps=12000,ppo_epochs=240,adam_updates=480,
            adam=adam,completed_episodes=len(cb.rows),checkpoints=[r['policy_steps'] for r in cb.saved],
            stage_counts={str(stage):int((raw.stages==stage).sum()) for stage in (1,2,3)},transitions=raw.transition_log,
            weights_Adam_eta_39RMS_reload_exact=True,RMS_count_actual_samples_only=True,
            save_preserves_world_route_pair_reference_RMS_RNG=True,continuous_learn_and_checkpoint_wall_s=wall,
            initial_weight_sha256=initial,engineering_only=True,physics_trajectory_restored=False,
            no_scientific_evaluation_or_promotion=True))
        atomic_json(directory/'progress.json',dict(status='complete',sampled_steps=12000,trained_steps=12000))
        print('PASS ENGINEERING',arm,'12k/240epochs/480Adam',len(cb.rows),'episodes',wall,'seconds',flush=True)
    except BaseException as error:
        prefix = directory/'interrupted_state'; model.save(prefix); norm.save(prefix.with_suffix('.pkl'))
        np.savez_compressed(directory/'interrupted_world_reference.npz',q=native.data.qpos.numpy(),v=native.data.qvel.numpy(),
            native_state=native.state.numpy(),controller_state=native.k['state'].numpy(),base_reference=native.k['reference'].numpy(),
            current_reference=cache.current_reference,route_y=route.odometry.y,route_clock=route.clock,stages=raw.stages)
        atomic_json(directory/'interruption.json',dict(error=repr(error),sampled_steps=model.num_timesteps,
            confirmed_trained_steps=cb.trained,last_saved_steps=cb.saved[-1]['policy_steps'] if cb.saved else 0,
            checkpoints=cb.saved,silently_resumable=False))
        raise
    finally: env.close()


def engineering():
    p,c = verify_engineering(); completed=[]
    for arm in c['arms']:
        atomic_json(OUT/'engineering_progress.json',dict(status='running',completed_arms=completed,pending_arm=arm))
        run_engineering_arm(arm); completed.append(arm)
    records={arm:json.loads((OUT/'engineering'/arm/'verification.json').read_text()) for arm in c['arms']}
    assert all(r['verified'] and r['policy_steps']==12000 for r in records.values())
    atomic_json(OUT/'engineering_verification.json',dict(verified=True,records=records,total_training_steps=24000,
        engineering_contract_sha256=sha(OUT/'engineering_contract.json'),source_sha256=sha(__file__),main_admitted=False,
        scope='Separate engineeringmodels never mainwarmstart orscientific results; actual gradients/checkpoint/lifecycle only.'))
    atomic_json(OUT/'engineering_progress.json',dict(status='complete',completed_arms=completed,trained_policy_steps=24000))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze_engineering','engineering'])
    globals()[parser.parse_args().command]()
