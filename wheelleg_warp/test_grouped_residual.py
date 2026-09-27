"""Small GPU check for the optional leg/wheel residual split."""
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from trace_failure_chain import COL, RecordedEnv
from dashboard.live_env import atomic_json as write

PANEL = ROOT / 'wheelleg_warp/results/contract_v2_baseline_checked_20260923/protocol.json'


def check(output):
    for kwargs in ({'grouped_residual': 1}, {'grouped_residual': True, 'project_clipped_base': True},
                   {'grouped_residual': True, 'residual_mode': 'virtual6'}):
        try: TerrainEnv(1, **kwargs)
        except ValueError: pass
        else: raise AssertionError('Invalid grouped allocation accepted')
    plain = TerrainEnv(1, scenario=TerrainScenario())
    grouped = TerrainEnv(1, scenario=TerrainScenario(), grouped_residual=True)
    try:
        x = plain.reset(); y = grouped.reset()
        np.testing.assert_allclose(x, y, atol=1e-6, rtol=0)
        for _ in range(50):
            action = np.array([[.1, -.1, .1]], dtype=np.float32)
            x = plain.step(action); y = grouped.step(action)
            np.testing.assert_allclose(x[0], y[0], atol=1e-6, rtol=0)
            np.testing.assert_allclose(x[1], y[1], atol=1e-6, rtol=0)
    finally: plain.close(); grouped.close()
    panel = json.loads(PANEL.read_text())
    case = TerrainScenario(**next(row['scenario'] for row in panel['panels'] if row['scenario']['terrain_seed'] == 700200))
    checkpoint = panel['checkpoints']['terrain_v3']['path']
    raw = RecordedEnv(1, scenario=case, grouped_residual=True)
    env = VecNormalize.load(checkpoint + '.pkl', raw); env.training = False; env.norm_reward = False
    torch.set_num_threads(1); model = PPO.load(checkpoint + '.zip', device='cpu')
    obs = env.reset(); chunks = []
    try:
        for _ in range(1000):
            obs, _, done, infos = env.step(model.predict(obs, deterministic=True)[0])
            chunk = raw.trace.numpy()[:, 0, :]
            chunks.append(chunk[chunk[:, 0] > 0].copy())
            if done[0]: break
        else: raise AssertionError('Episode did not terminate')
    finally: env.close()
    trace = np.concatenate(chunks); col = lambda name: trace[:, COL.index(name)]
    base = np.column_stack([col(f'base_{j}') for j in range(6)])
    requested = np.column_stack([col(f'requested_{j}') for j in range(6)])
    executed = np.column_stack([col(f'executed_{j}') for j in range(6)])
    bound = np.column_stack([col(f'bound_{j}') for j in range(6)])
    assert np.isfinite(trace).all() and len(trace) == infos[0]['physical_steps']
    assert np.max(np.abs(base + executed) - bound) < 1e-7
    for part in (slice(0, 4), slice(4, 6)):
        ratio = np.divide(executed[:, part], requested[:, part],
            out=np.full_like(executed[:, part], np.nan), where=np.abs(requested[:, part]) > 1e-9)
        assert np.nanmin(ratio) >= -1e-8 and np.nanmax(ratio) <= 1. + 1e-8
        valid = np.isfinite(ratio).sum(axis=1) >= 2
        assert np.max(np.nanmax(ratio[valid], axis=1) - np.nanmin(ratio[valid], axis=1)) < 1e-7
    released = (col('lambda') < 1e-9) & (np.linalg.norm(executed[:, :4], axis=1) > 1e-9)
    assert released.any()
    np.testing.assert_allclose(col('reward_actual').sum(), infos[0]['episode']['r'], atol=1e-7)
    result = dict(passed=True, nonbinding_seconds=1., trace_steps=len(trace),
        released_leg_substeps=int(released.sum()), max_nominal_bound_excess=float(np.max(np.abs(base + executed) - bound)),
        note='Engineering check only; single-world trajectory is not a capability result or hardware torque validation.')
    output.parent.mkdir(parents=True, exist_ok=True); write(output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    if a.output.exists(): p.error('Refusing to overwrite evidence')
    check(a.output)
