"""冻结双物理后端正式训练协议，运行两条队列，完成后自动在CPU共同终评。"""
import argparse
from dataclasses import asdict
from datetime import datetime
from functools import partial
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'wheelleg_ppo/tools'))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan, VecNormalize
from ppo_env import WheelLegEnv, sample_scenario
from pretrain_yaw import verify_sources, summarize
from train_yaw import SelectionCallback, evaluate

FROZEN = ROOT / 'wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21'


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_protocol(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('正式协议源码已变化：' + name)
    for name, expected in protocol['versions'].items():
        if version(name) != expected:
            raise RuntimeError('正式协议依赖已变化：' + name)
    verify_sources(FROZEN)
    return protocol


def initialize(output):
    manifest = verify_sources(FROZEN)
    original = json.loads((FROZEN / 'training_config.json').read_text())
    # 用户此次明确授权为独立后端比较，不把尚未对齐的GPU冒称原CPU等价替代。
    reference = json.loads((ROOT / 'wheelleg_warp/CPU_REFERENCE.json').read_text())
    for path, digest in reference['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    paths = [ROOT / 'wheelleg_warp' / name for name in
             ('train_compare.py', 'gpu_env.py', 'baseline.py', 'requirements.txt', 'CPU_REFERENCE.json')]
    paths += [FROZEN / 'training_config.json', FROZEN / 'protocol_manifest.json']
    hashes = dict(reference['source_sha256'])
    hashes.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    # 不打开原研究门控/最终集；新场景在任何训练和选模前固定，仅最终报告使用。
    final_cases = [dict(seed=seed, scenario=asdict(sample_scenario('test_iid', seed, 3)))
                   for seed in range(91000, 91064)]
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(name='cpu-warp-formal-backend-comparison-v1', created=datetime.now().astimezone().isoformat(),
        method='M3', backends=['cpu', 'warp'], seeds=original['training_seeds'], config=original,
        policy_device='cpu', physics_devices={'cpu':'cpu', 'warp':'cuda:0'},
        total_policy_steps=2 * len(original['training_seeds']) * original['policy_steps'],
        selection_cases=manifest['sets']['selection'], final_cases=final_cases,
        checkpoint_selection='共同CPU选择集上按成功数降序、Jpsi升序、步数升序选最优；同时报告末次检查点',
        primary_effect='共同CPU终评64例成功数与完整轨迹Jpsi；3配对种子逐一及均值/标准差，不预定优胜方',
        secondary_effect=['速度RMSE','停车距离','尾速','姿态峰值','残差/电机饱和率','各终止原因'],
        timing='采样与更新总耗时（含周期选择评估）及端到端耗时；并发资源竞争下不作独占硬件速度结论',
        known_limitations=['GPU非对称接触全状态门未通过，属于不同仿真后端的效果比较',
                          'GPU物理逐步回传CPU控制器，当前不是GPU原生批量闭环，不承诺提速',
                          '现有CPU分段运行不纳入配对，保留为历史实验；新两组从相同种子重新初始化'],
        original_research_holdouts_used=False, source_sha256=hashes,
        versions={**original['versions'], 'mujoco-warp':version('mujoco-warp'), 'warp-lang':version('warp-lang')})
    write(output / 'protocol.json', protocol)
    print('协议已冻结：', output, flush=True)


class ProgressCallback(SelectionCallback):
    def _on_rollout_start(self):
        super()._on_rollout_start()
        write(self.output / 'progress.json', dict(policy_steps=self.num_timesteps, target=self.config['policy_steps'],
              updated=datetime.now().astimezone().isoformat(), stage=self.training_env.get_attr('stage')[0]))


def run_one(output, backend, seed, smoke=False):
    protocol = load_protocol(output)
    config = protocol['config']
    if backend not in protocol['backends'] or seed not in protocol['seeds']:
        raise ValueError('未注册的后端/种子')
    run = output / (('smoke_' if smoke else '') + f'{backend}_{seed}')
    if (run / 'completed.json').exists():
        if json.loads((run / 'completed.json').read_text())['passed']:
            return
    run.mkdir(exist_ok=False)  # 失败/中断必须显式处理，不能自动覆盖或伪装连续恢复。
    factory = WheelLegEnv
    if backend == 'warp':
        from gpu_env import WarpEnv
        factory = WarpEnv
    torch.set_num_threads(1)
    raw = SubprocVecEnv([partial(factory, mode='diff3', stage=1) for _ in range(config['environments'])], start_method='spawn')
    env = VecNormalize(VecCheckNan(raw, raise_exception=True), **config['normalization'])
    env.seed(seed)
    model = PPO('MlpPolicy', env, device='cpu', seed=seed, **config['ppo'])
    initial = {k:v.clone() for k,v in model.policy.state_dict().items()}
    initial_hash = hashlib.sha256(b''.join(t.cpu().numpy().tobytes() for t in initial.values())).hexdigest()
    write(run / 'run_config.json', dict(backend=backend, seed=seed, smoke=smoke, config=config,
          physical_device=protocol['physics_devices'][backend], policy_device='cpu', initial_policy_sha256=initial_hash,
          protocol_sha256=hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest(),
          known_limitations=protocol['known_limitations']))
    callback = ProgressCallback(config, dict(sets=dict(selection=protocol['selection_cases'])), run, smoke)
    budget = 4000 if smoke else config['policy_steps']
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=budget, callback=callback)
        train_seconds = time.perf_counter() - start
        assert model.num_timesteps == budget
        assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
        assert any(not torch.equal(initial[k],v) for k,v in model.policy.state_dict().items())
        losses = {k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}
        assert losses and all(np.isfinite(v) for v in losses.values())
        callback.save_and_evaluate()  # 使用原CPU evaluate；两个后端共享同一选择标准。
        if smoke:
            for stage in (2, 3):
                env.set_attr('stage', stage)
                obs = env.reset()
                obs, rewards, _, _ = env.step(np.tile([.1, -.1, .1], (config['environments'], 1)))
                assert np.isfinite(obs).all() and np.isfinite(rewards).all()
        load_protocol(output)
        write(run / 'completed.json', dict(passed=True, backend=backend, seed=seed, smoke=smoke,
            policy_steps=budget, train_seconds=train_seconds, total_seconds=time.perf_counter()-start,
            losses=losses, initial_policy_sha256=initial_hash, formal_training_started=not smoke,
            policy_parameter_count=sum(v.numel() for v in model.policy.parameters()),
            evaluation_backend='cpu', final_set_evaluated=False))
        print('完成', backend, seed, budget, train_seconds, flush=True)
    except Exception as exc:
        write(run / 'failed.json', dict(error=repr(exc), policy_steps=model.num_timesteps))
        raise
    finally:
        env.close()


def compare(output):
    protocol = load_protocol(output)
    cases = protocol['final_cases']
    rows = []
    for seed in protocol['seeds']:
        for backend in protocol['backends']:
            run = output / f'{backend}_{seed}'
            done = json.loads((run / 'completed.json').read_text())
            assert done['passed'] and not done['smoke'] and done['policy_steps'] == protocol['config']['policy_steps']
            selection = json.loads((run / 'selection.json').read_text())
            last = run / f"step_{done['policy_steps']}"
            chosen = Path(selection['best']['path']) if selection['best'] else last
            # 无完整选择轨迹时明确回退末次；所有失败保留，不过滤成更好成绩。
            for kind, prefix in [('selected', chosen), ('last', last)]:
                result_path = run / f'final_{kind}.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    model = PPO.load(str(prefix) + '.zip', device='cpu')
                    runs = evaluate(model, str(prefix)+'.pkl', 'diff3', cases)
                    result = dict(backend=backend, seed=seed, checkpoint_kind=kind, checkpoint=str(prefix),
                                  selection_fallback=selection['best'] is None,
                                  summary=summarize(runs, [r['seed'] for r in cases]), runs=runs)
                    write(result_path, result)
                rows.append(dict(backend=backend, seed=seed, kind=kind, **result['summary'],
                                 train_seconds=done['train_seconds'], total_seconds=done['total_seconds'],
                                 initial_policy_sha256=done['initial_policy_sha256']))
    for seed in protocol['seeds']:
        pair = [r for r in rows if r['seed'] == seed and r['kind'] == 'selected']
        assert pair[0]['initial_policy_sha256'] == pair[1]['initial_policy_sha256']
    aggregates = []
    for backend in protocol['backends']:
        for kind in ('selected', 'last'):
            group = [r for r in rows if r['backend'] == backend and r['kind'] == kind]
            scores = [r['mean_yaw_score_deg'] for r in group]
            aggregates.append(dict(backend=backend, kind=kind,
                success_mean=float(np.mean([r['success_count'] for r in group])),
                success_std=float(np.std([r['success_count'] for r in group], ddof=1)),
                yaw_mean=float(np.mean(scores)) if all(s is not None for s in scores) else None,
                yaw_std=float(np.std(scores, ddof=1)) if all(s is not None for s in scores) else None,
                total_seconds=sum(r['total_seconds'] for r in group)))
    report = dict(protocol=protocol['name'], evaluation_backend='cpu', cases=64, rows=rows,
                  aggregates=aggregates, limitations=protocol['known_limitations'], completed=True)
    write(output / 'comparison.json', report)
    lines = ['# CPU与Warp GPU正式训练效果比较', '', '两组均从同种子初始化，M3，每种子200万策略步，3个配对种子；共同CPU终评64例。', '',
             '| 后端 | 检查点 | 成功数均值±标准差 /64 | Jψ均值±标准差(°) | 三种子总耗时(s) |',
             '|---|---|---|---|---|']
    for row in aggregates:
        yaw = f"{row['yaw_mean']:.6f} ± {row['yaw_std']:.6f}" if row['yaw_mean'] is not None else '有未完成轨迹，不计算均值'
        lines.append(f"| {row['backend']} | {row['kind']} | {row['success_mean']:.2f} ± {row['success_std']:.2f} | {yaw} | {row['total_seconds']:.1f} |")
    lines += ['', '选模检查点和末次检查点均报告，不根据终评重新选模。详细单场景指标与失败原因保存在各final JSON。', '',
              'GPU训练保留CPU控制器并逐步回读，接触全状态等价门未通过。并发运行有资源竞争，耗时不能作为独占硬件倍率。此比较不验证差模论文相对其他方法的优势。']
    (output / 'COMPARISON.md').write_text('\n'.join(lines) + '\n')


def orchestrate(output):
    protocol = load_protocol(output)
    for backend in protocol['backends']:
        check = json.loads((output / f'smoke_{backend}_1609/completed.json').read_text())
        assert check['passed'] and check['smoke']
    processes = []
    logs = []
    try:
        for backend in protocol['backends']:
            log = (output / f'{backend}.log').open('a')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__)), 'queue',
                              '--backend', backend, '--output', str(output)], stdout=log, stderr=subprocess.STDOUT))
        write(output / 'status.json', dict(status='training', pids=dict(zip(protocol['backends'], [p.pid for p in processes]))))
        # ponytail: 单机双队列；跨机器调度有实际需求时再替换，退出码失败保留各自产物。
        codes = [p.wait() for p in processes]
        if any(codes):
            write(output / 'status.json', dict(status='failed', returncodes=dict(zip(protocol['backends'], codes))))
            raise RuntimeError('训练队列失败，禁止生成完整比较结论')
        write(output / 'status.json', dict(status='evaluating'))
        compare(output)
        write(output / 'status.json', dict(status='completed', comparison=str(output / 'COMPARISON.md')))
    finally:
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['init', 'smoke', 'queue', 'orchestrate', 'compare'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['cpu', 'warp'])
    args = p.parse_args()
    output = args.output.resolve()
    if args.command == 'init':
        initialize(output)
    elif args.command == 'smoke':
        run_one(output, args.backend, 1609, smoke=True)
    elif args.command == 'queue':
        for seed in load_protocol(output)['seeds']:
            run_one(output, args.backend, seed)
    elif args.command == 'orchestrate':
        orchestrate(output)
    else:
        compare(output)


if __name__ == '__main__':
    main()
