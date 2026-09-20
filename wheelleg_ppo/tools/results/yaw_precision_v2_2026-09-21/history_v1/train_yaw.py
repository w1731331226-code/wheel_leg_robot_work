"""PPO entry point: --smoke validates the frozen configuration, no long run by default."""
import argparse
from functools import partial
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecCheckNan
from ppo_env import WheelLegEnv, Scenario
from pretrain_yaw import ROOT, FROZEN, verify_sources, write, summarize, selection_key


def evaluate(model, normalization, mode, cases):
    raw = SubprocVecEnv([partial(WheelLegEnv, mode=mode) for _ in range(4)], start_method='spawn')
    env = VecNormalize.load(normalization, raw)
    env.training = False
    env.norm_reward = False
    stats = env.obs_rms.mean.copy()
    rows = []
    try:
        for start in range(0, len(cases), 4):
            batch = cases[start:start+4]
            for i in range(4):
                raw.set_attr('fixed_scenario', Scenario(**batch[i % len(batch)]['scenario']), indices=i)
            obs = env.reset()
            pending = set(range(len(batch)))
            batch_rows = {}
            while pending:
                obs, _, done, infos = env.step(model.predict(obs, deterministic=True)[0])
                for i in list(pending):
                    if done[i]:
                        batch_rows[i] = dict(seed=batch[i]['seed'], scenario=batch[i]['scenario'], **infos[i])
                        # SB3 adds arrays that are not part of the scientific metrics.
                        batch_rows[i].pop('terminal_observation', None)
                        pending.remove(i)
            rows.extend(batch_rows[i] for i in range(len(batch)))
        np.testing.assert_array_equal(stats, env.obs_rms.mean)
    finally:
        env.close()
    return rows


class SelectionCallback(BaseCallback):
    def __init__(self, config, manifest, output, smoke):
        super().__init__()
        self.config, self.manifest, self.output, self.smoke = config, manifest, output, smoke
        self.records = []
        self.last_saved = -1

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # This hook is after the previous rollout's PPO update.
        if not self.smoke and self.num_timesteps and self.num_timesteps % self.config['evaluation_interval'] == 0:
            self.save_and_evaluate()
        stage = max(x['stage'] for x in self.config['curriculum'] if self.num_timesteps >= x['start'])
        self.training_env.set_attr('stage', stage)

    def save_and_evaluate(self):
        step = self.model.num_timesteps
        if self.last_saved == step:
            return
        self.last_saved = step
        prefix = self.output / f'step_{step}'
        self.model.save(prefix)
        self.training_env.save(str(prefix)+'.pkl')
        loaded = PPO.load(str(prefix)+'.zip', device=self.config['device'])
        probe = np.zeros((2, 32), dtype=np.float32)
        np.testing.assert_array_equal(self.model.predict(probe, deterministic=True)[0],
                                      loaded.predict(probe, deterministic=True)[0])
        cases = self.manifest['sets']['selection']
        if self.smoke:
            old = json.loads((ROOT/'tools/results/development_extension_2026-09-20/preregistered_scenarios.json').read_text())
            cases = [dict(seed=seed, scenario=case['scenario']) for seed, case in zip(old['seeds'][:4], old['cases'][:4])]
        rows = evaluate(loaded, str(prefix)+'.pkl', self.model.get_env().get_attr('mode')[0], cases)
        summary = summarize(rows, [c['seed'] for c in cases])
        write(Path(str(prefix)+'.json'), dict(steps=step, smoke=self.smoke, summary=summary, runs=rows))
        self.records.append(dict(steps=step, path=str(prefix), **summary))
        valid = [r for r in self.records if r['complete']]
        best = min(valid, key=lambda r: selection_key(r, r['steps'])) if valid else None
        write(self.output/'selection.json', dict(smoke=self.smoke, checkpoints=self.records, best=best,
                                                gate_evaluated=False, test_set_evaluated=False))


def run(method, seed, output, smoke):
    manifest = verify_sources()
    config_path = FROZEN/'training_config.json'
    config = json.loads(config_path.read_text())
    for package, expected in config['versions'].items():
        if version(package) != expected:
            raise RuntimeError('Frozen package version changed: '+package)
    if seed not in config['training_seeds']:
        raise ValueError('Seed is not preregistered for pilot training')
    if not smoke:
        gate = ROOT/'tools/results/pretrain_yaw_2026-09-21/readiness.json'
        if not gate.exists() or not json.loads(gate.read_text())['engineering_ready']:
            raise RuntimeError('Pre-training readiness is not complete')
        readiness = json.loads(gate.read_text())
        for name, digest in readiness['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Readiness evidence is stale: '+name)
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT/'tools/pretrain_yaw.py', config_path]
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    write(output/'run_config.json', dict(method=method, seed=seed, smoke=smoke, config=config,
                                       source_sha256=source_hashes, frozen_source_sha256=manifest['source_sha256']))
    torch.set_num_threads(1)
    mode = config['methods'][method]
    raw = SubprocVecEnv([partial(WheelLegEnv, mode=mode, stage=1) for _ in range(config['environments'])], start_method='spawn')
    env = VecNormalize(VecCheckNan(raw, raise_exception=True), **config['normalization'])
    env.seed(seed)
    model = PPO('MlpPolicy', env, device=config['device'], seed=seed, **config['ppo'])
    initial = {k:v.clone() for k,v in model.policy.state_dict().items()}
    callback = SelectionCallback(config, manifest, output, smoke)
    steps = 4000 if smoke else config['policy_steps']
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=steps, callback=callback)
        assert model.num_timesteps == steps
        assert all(torch.isfinite(x).all() for x in model.policy.state_dict().values())
        assert any(not torch.equal(initial[k], v) for k,v in model.policy.state_dict().items())
        losses = {k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}
        assert all(np.isfinite(v) for v in losses.values())
        train_seconds = time.perf_counter()-start
        callback.save_and_evaluate()
        # Exercise scheduled stage updates/reset without extending the training budget.
        if smoke:
            for stage in (2, 3):
                env.set_attr('stage', stage)
                env.reset()
                assert env.get_attr('stage') == [stage]*config['environments']
        verify_sources()
        assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in source_hashes.items())
        write(output/'completed.json',dict(method=method, smoke=smoke, passed=True, policy_steps=steps,
              train_seconds=train_seconds, total_seconds=time.perf_counter()-start, losses=losses,
              save_reload_pass=True, evaluation_normalization_frozen=True,
              curriculum_reset_pass=smoke, formal_training_started=not smoke))
    finally:
        env.close()
    print(method, 'PASS', steps, train_seconds, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['M','B2-V','B2'], required=True)
    parser.add_argument('--seed', type=int, default=1609)
    parser.add_argument('--output', type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--smoke', action='store_true')
    choice.add_argument('--train', action='store_true')
    args = parser.parse_args()
    run(args.method, args.seed, args.output, args.smoke)
