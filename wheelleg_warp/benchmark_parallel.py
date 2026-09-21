"""独占机器并行配置实测：相同训练场景库、250步轨迹与原PPO超参。"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_ppo/tools'), str(ROOT / 'wheelleg_warp')]
# 在导入数值库前限制每个环境的BLAS线程，避免进程数×线程数过量。
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecCheckNan
from ppo_env import WheelLegEnv, sample_scenario


def make_env(backend, index):
    cls = WheelLegEnv
    if backend == 'warp':
        from fast_physics import FastWarpEnv
        cls = FastWarpEnv
    return cls(mode='diff3', stage=3, scenario=sample_scenario('train', 730000 + index, 3))


class TimedPPO(PPO):
    def collect_rollouts(self, *args, **kwargs):
        start = time.perf_counter()
        result = super().collect_rollouts(*args, **kwargs)
        self.sample_seconds = time.perf_counter() - start
        return result

    def train(self):
        start = time.perf_counter()
        super().train()
        update = time.perf_counter() - start
        self.timings.append(dict(sample_seconds=self.sample_seconds, update_seconds=update,
                                 seconds=self.sample_seconds + update))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'warp', 'native'], required=True)
    p.add_argument('--envs', type=int, required=True)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args()
    if not 1 <= a.envs <= 1024 or a.repeats < 1 or a.output.exists():
        p.error('环境数须为1～1024，重复数必须为正，结果不得覆盖')
    start = time.perf_counter()
    torch.set_num_threads(1)
    config = json.loads((ROOT / 'wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21/training_config.json').read_text())
    sources = [Path(__file__), ROOT / 'wheelleg_warp/fast_physics.py', ROOT / 'wheelleg_ppo/tools/ppo_env.py']
    sources += list((ROOT / 'wheelleg_warp/native').glob('*.py')) if a.backend == 'native' else []
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    samples = []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                line = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu', '--format=csv,noheader,nounits'], text=True)
                samples.append([float(x) for x in line.strip().split(',')])
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    raw = None
    try:
        if a.backend == 'native':
            from native.environment import NativeEnv
            raw = NativeEnv(a.envs, stage=3, seed=730000)
        else:
            raw = SubprocVecEnv([partial(make_env, a.backend, i) for i in range(a.envs)], start_method='spawn')
        env = VecNormalize(VecCheckNan(raw, raise_exception=True), **config['normalization'])
        model = TimedPPO('MlpPolicy', env, seed=1609, device=a.device, **config['ppo'])
        model.timings = []
        initial = {k: v.clone() for k, v in model.policy.state_dict().items()}
        setup = time.perf_counter() - start
        # 首轮单独记录，随后各轮含真实采样、自动重置、GAE、十轮梯度更新。
        model.learn(total_timesteps=a.envs * 250 * (a.repeats + 1))
        assert model.num_timesteps == a.envs * 250 * (a.repeats + 1)
        assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
        assert any(not torch.equal(initial[k], v) for k, v in model.policy.state_dict().items())
        losses = {k: float(v) for k, v in model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}
        assert losses and all(np.isfinite(v) for v in losses.values())
        unchanged = all(hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h for f, h in hashes.items())
        assert unchanged, '测试期间源码发生变化，结果无效'
        rounds = model.timings[1:]
        result = dict(backend=a.backend, environments=a.envs, device=a.device, setup_seconds=setup,
            warmup=model.timings[0], rounds=rounds, policy_steps_per_round=a.envs*250,
            median_policy_sps=a.envs*250/float(np.median([r['seconds'] for r in rounds])),
            median_sampling_sps=a.envs*250/float(np.median([r['sample_seconds'] for r in rounds])),
            total_seconds=time.perf_counter()-start, total_policy_steps=model.num_timesteps,
            gpu_mean=np.mean(samples, axis=0).tolist() if samples else None,
            gpu_peak=np.max(samples, axis=0).tolist() if samples else None,
            gpu_sample_columns=['util_percent','memory_MiB','power_W','temperature_C'],
            ppo=config['ppo'], losses=losses, source_sha256=hashes, passed=True,
            scope='固定训练场景库，stage3；无录像/周期评估；包含真实PPO更新；不证明学习效果或原生GPU等价')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if raw is not None:
            raw.close()
        stop.set()
        thread.join(timeout=2)


if __name__ == '__main__':
    main()
