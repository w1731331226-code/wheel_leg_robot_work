"""Paired public-panel ablation of the optional leg/wheel residual allocation."""
from pathlib import Path
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from terrain_eval import validate_terrain_rows
from training_contract import TASK_CONTRACT_VERSION, source_hashes
from dashboard.live_env import atomic_json as write

PANEL = ROOT / 'wheelleg_warp/results/contract_v2_baseline_checked_20260923/protocol.json'
ORDER = (False, True, True, False)
SELECTED = (700200, 700202, 700300, 700302)


def run(output):
    frozen = json.loads(PANEL.read_text())
    assert frozen['task_contract_version'] == TASK_CONTRACT_VERSION and len(frozen['panels']) == 160
    for item in frozen['checkpoints'].values():
        for suffix, key in (('.zip', 'checkpoint_sha256'), ('.pkl', 'normalization_sha256')):
            assert hashlib.sha256(Path(item['path'] + suffix).read_bytes()).hexdigest() == item[key]
    for path, digest in frozen['source_panel_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    scenarios = [TerrainScenario(**row['scenario']) for row in frozen['panels']]
    assert len({case.terrain_seed for case in scenarios}) == len(scenarios)
    checkpoint = frozen['checkpoints']['terrain_v3']['path']
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol.json', dict(task_contract_version=TASK_CONTRACT_VERSION,
        panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(), source_sha256=source_hashes(__file__),
        checkpoint=frozen['checkpoints']['terrain_v3'], order=ORDER, worlds=160, selected=SELECTED,
        changed_factor='grouped residual projection only', training=False, holdout_evaluated=False,
        note='All policies, normalization, scenarios and task thresholds frozen; no model update.'))
    torch.set_num_threads(1); results = []
    for number, grouped in enumerate(ORDER, 1):
        raw = TerrainEnv(len(scenarios), scenario=scenarios, grouped_residual=grouped)
        env = VecNormalize.load(checkpoint + '.pkl', raw); env.training = False; env.norm_reward = False
        model = PPO.load(checkpoint + '.zip', device='cpu')
        original_stats = (env.obs_rms.mean.copy(), env.obs_rms.var.copy(), env.obs_rms.count)
        observations = env.reset(); pending = set(range(len(scenarios))); rows = [None] * len(scenarios)
        try:
            for _ in range(1000):
                if not pending: break
                observations, _, done, infos = env.step(model.predict(observations, deterministic=True)[0])
                for i in list(pending):
                    if done[i]:
                        rows[i] = {key: value for key, value in infos[i].items() if key != 'terminal_observation'}
                        pending.remove(i)
            if pending: raise RuntimeError('Public panel did not terminate within 20 s')
            validate_terrain_rows(rows)
            np.testing.assert_array_equal(env.obs_rms.mean, original_stats[0])
            np.testing.assert_array_equal(env.obs_rms.var, original_stats[1])
            assert env.obs_rms.count == original_stats[2]
            outcomes = [dict(seed=case.terrain_seed, group=frozen['panels'][i]['group'], info=rows[i])
                        for i, case in enumerate(scenarios)]
            result = dict(run=number, grouped_residual=grouped, outcomes=outcomes,
                hard_success=sum(x['info']['success'] for x in outcomes if x['group'] == 'difficult/step'),
                hard_complete=sum(x['info']['reason'] == 'completed' for x in outcomes if x['group'] == 'difficult/step'),
                all_success=sum(row['success'] for row in rows),
                selected={str(seed): rows[next(i for i, case in enumerate(scenarios) if case.terrain_seed == seed)]
                          for seed in SELECTED})
            write(output / f'run_{number:02d}.json', result); results.append(result)
            print('run', number, 'grouped', grouped, 'hard', result['hard_success'],
                  'selected', [result['selected'][str(seed)]['success'] for seed in SELECTED], flush=True)
        finally: env.close()
    write(output / 'summary.json', dict(runs=[dict(run=r['run'], grouped_residual=r['grouped_residual'],
        hard_success=r['hard_success'], hard_complete=r['hard_complete'], all_success=r['all_success'],
        selected_success={seed: row['success'] for seed, row in r['selected'].items()}) for r in results],
        training=False, promoted=False, holdout_evaluated=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.output)
