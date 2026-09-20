"""Resume a saved PPO update into a new segment; simulator/RNG state is reset and disclosed."""
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan, VecNormalize
from ppo_env import WheelLegEnv
from pretrain_yaw import ROOT, verify_sources, write
from train_yaw import SelectionCallback

FROZEN = ROOT/'tools/results/yaw_precision_v2_2026-09-21'


def resume(checkpoint, output, smoke):
    checkpoint = checkpoint.resolve().with_suffix('')
    manifest = verify_sources(FROZEN)
    config = json.loads((FROZEN/'training_config.json').read_text())
    ready = json.loads((FROZEN/'readiness.json').read_text())
    if not ready['engineering_ready'] or not ready['research_ready']:
        raise RuntimeError('v2 readiness is not complete')
    for name, digest in ready['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    for package, expected in config['versions'].items():
        assert version(package) == expected, package
    source = json.loads((checkpoint.parent/'run_config.json').read_text())
    if source['smoke'] != smoke or source['config'] != config:
        raise ValueError('Checkpoint run type/config differs; smoke cannot initialize a formal run')
    for name, digest in source['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    method, seed = source['method'], source['seed']
    torch.set_num_threads(1)
    model = PPO.load(str(checkpoint)+'.zip', device=config['device'])
    start_step = model.num_timesteps
    interval = config['ppo']['n_steps'] * config['environments']
    if checkpoint.name != f'step_{start_step}' or start_step <= 0 or start_step % interval:
        raise ValueError('Not a complete PPO rollout checkpoint')
    target = start_step + interval if smoke else config['policy_steps']
    if target <= start_step or (not smoke and start_step % config['evaluation_interval']):
        raise ValueError('No remaining budget or not a scheduled formal checkpoint')
    initial = {k:v.clone() for k,v in model.policy.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in initial.values())
    optimizer = model.policy.optimizer.state_dict()['state']
    assert optimizer and all(torch.isfinite(v).all() for r in optimizer.values() for v in r.values() if torch.is_tensor(v))
    selection = json.loads((checkpoint.parent/'selection.json').read_text())
    records = selection['checkpoints']
    if selection['smoke'] != smoke or len({r['steps'] for r in records}) != len(records) or any(r['steps']>start_step for r in records):
        raise ValueError('Invalid checkpoint selection history')
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(str(checkpoint)+'.zip'), Path(str(checkpoint)+'.pkl'), checkpoint.parent/'run_config.json',
             checkpoint.parent/'selection.json', Path(__file__)]
    hashes = {str(p.resolve().relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write(output/'run_config.json', dict(method=method,seed=seed,smoke=smoke,config=config,
          source_sha256=hashes,frozen_source_sha256=manifest['source_sha256'],
          resume_from=str(checkpoint),start_policy_steps=start_step,target_policy_steps=target,
          reset_simulator_and_rng=True,exact_trajectory_resume=False,
          note='Policy, optimizer and VecNormalize restored; environment state and random streams were not saved. Count this as a disclosed segmented run.'))
    stage = max(item['stage'] for item in config['curriculum'] if start_step>=item['start'])
    raw = SubprocVecEnv([partial(WheelLegEnv,mode=config['methods'][method],stage=stage)
                        for _ in range(config['environments'])],start_method='spawn')
    env = VecNormalize.load(str(checkpoint)+'.pkl', VecCheckNan(raw,raise_exception=True))
    env.training, env.norm_reward = True, False
    env.seed(seed)
    model.set_env(env, force_reset=True)
    callback = SelectionCallback(config,manifest,output,smoke)
    callback.records = records.copy()
    callback.init_callback(model)
    callback.num_timesteps = start_step
    if any(r['steps']==start_step for r in records):
        callback.last_saved = start_step
    started = time.perf_counter()
    try:
        # Complete a checkpoint evaluation interrupted after policy/normalization save.
        callback.save_and_evaluate()
        print('RESUMED',method,seed,start_step,'target',target,'stage',stage,flush=True)
        model.learn(total_timesteps=target-start_step,reset_num_timesteps=False,callback=callback)
        assert model.num_timesteps == target
        assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
        assert any(not torch.equal(initial[k],v) for k,v in model.policy.state_dict().items())
        losses={k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}
        assert all(np.isfinite(v) for v in losses.values())
        callback.save_and_evaluate()
        verify_sources(FROZEN)
        assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
        write(output/'completed.json',dict(protocol=config['protocol'],method=method,smoke=smoke,passed=True,
              policy_steps=target,additional_policy_steps=target-start_step,start_policy_steps=start_step,
              seconds=time.perf_counter()-started,losses=losses,segmented_run=True,exact_trajectory_resume=False,
              formal_training_started=not smoke,gate_evaluated=False,test_set_evaluated=False))
    finally:
        env.close()
    print('PASS',method,'resumed',start_step,'to',target,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--smoke',action='store_true')
    choice.add_argument('--train',action='store_true')
    args=parser.parse_args()
    resume(args.checkpoint,args.output,args.smoke)
