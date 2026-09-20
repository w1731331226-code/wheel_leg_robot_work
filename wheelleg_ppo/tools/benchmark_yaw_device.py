"""Paired CPU/CUDA timing from one checkpoint; diagnostic updates never rejoin training."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecCheckNan
from ppo_env import WheelLegEnv
from pretrain_yaw import ROOT, verify_sources, write


def trial(checkpoint, config, mode, device, seed):
    raw = SubprocVecEnv([partial(WheelLegEnv,mode=mode,stage=3)
                        for _ in range(config['environments'])],start_method='spawn')
    env = VecNormalize.load(str(checkpoint)+'.pkl',VecCheckNan(raw,raise_exception=True))
    env.training, env.norm_reward = True, False
    model = PPO.load(str(checkpoint)+'.zip',env=env,device=device,force_reset=True)
    model.set_random_seed(seed)
    assert model.device.type==device and next(model.policy.parameters()).device.type==device
    start_steps = model.num_timesteps
    update_times=[]
    train=model.train
    def synchronize():
        if device=='cuda': torch.cuda.synchronize()
    def timed_train():
        synchronize(); start=time.perf_counter()
        train()
        synchronize(); update_times.append(time.perf_counter()-start)
    model.train=timed_train
    try:
        obs=env.reset()  # Prime nominal model caches outside the timing window.
        for _ in range(5): model.predict(obs,deterministic=True)
        synchronize()
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        model.learn(total_timesteps=2000,reset_num_timesteps=False)
        synchronize(); elapsed=time.perf_counter()-started
        assert model.num_timesteps-start_steps==2000 and len(update_times)==1
        assert all(torch.isfinite(x).all() for x in model.policy.state_dict().values())
        losses={k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}
        assert all(np.isfinite(x) for x in losses.values())
        return dict(device=device,seed=seed,policy_steps=2000,seconds=elapsed,update_seconds=sum(update_times),
                    rollout_and_overhead_seconds=elapsed-sum(update_times),policy_steps_per_second=2000/elapsed,
                    finite=True,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
    finally:
        model.train=train
        env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    checkpoint=args.checkpoint.resolve().with_suffix('')
    torch.set_num_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    frozen=ROOT/'tools/results/yaw_precision_v2_2026-09-21'
    verify_sources(frozen)
    run=json.loads((checkpoint.parent/'run_config.json').read_text())
    config=run['config']; mode=config['methods'][run['method']]
    assert config['environments']==8 and config['ppo']['n_steps']==250 and config['ppo']['batch_size']==250
    paths=[Path(__file__),Path(str(checkpoint)+'.zip'),Path(str(checkpoint)+'.pkl'),checkpoint.parent/'run_config.json']
    hashes={str(p.resolve().relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'design.json',dict(checkpoint=str(checkpoint),order=[['cpu','cuda'],['cuda','cpu'],['cpu','cuda']],
          seeds=[7100,7101,7102],stage=3,policy_steps_per_trial=2000,total_diagnostic_policy_steps=12000,
          switch_rule='median total CPU/CUDA speedup >= 1.10 and every pair > 1; all CUDA finite',
          source_sha256=hashes,config=config,
          note='Same starting weights/optimizer/normalization and distribution; CPU/CUDA random streams and trajectories can differ. No validation/test sets.'))
    rows=[]
    for i,order in enumerate([('cpu','cuda'),('cuda','cpu'),('cpu','cuda')]):
        for device in order:
            result=trial(checkpoint,config,mode,device,7100+i)
            rows.append(result);write(args.output/'trials.json',rows)
            print(json.dumps(result),flush=True)
    totals={device:median(r['seconds'] for r in rows if r['device']==device) for device in ('cpu','cuda')}
    pairs=[next(r['seconds'] for r in rows if r['seed']==seed and r['device']=='cpu')/
           next(r['seconds'] for r in rows if r['seed']==seed and r['device']=='cuda') for seed in (7100,7101,7102)]
    speedup=totals['cpu']/totals['cuda']
    verify_sources(frozen)
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    result=dict(cpu_median_seconds=totals['cpu'],cuda_median_seconds=totals['cuda'],cuda_speedup=speedup,
                paired_speedups=pairs,switch_to_cuda=speedup>=1.10 and min(pairs)>1,
                rows=rows,source_sha256=hashes,cuda_name=torch.cuda.get_device_name(),
                torch_version=torch.__version__,gate_evaluated=False,test_set_evaluated=False,
                diagnostic_models_discarded=True)
    write(args.output/'summary.json',result)
    print('SUMMARY',json.dumps({k:result[k] for k in ('cpu_median_seconds','cuda_median_seconds','cuda_speedup','switch_to_cuda')}),flush=True)
