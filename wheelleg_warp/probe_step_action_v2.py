"""Bounded, position-gated M3 action probe on public 24/27 mm steps."""
from pathlib import Path
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from baseline_contract_v2 import verify
from native.terrain import TerrainScenario
from trace_failure_chain import RecordedEnv, COL, analyze
from terrain_eval import validate_terrain_rows
from training_contract import TASK_CONTRACT_VERSION, source_hashes
from dashboard.live_env import atomic_json as write

PANEL = ROOT / 'wheelleg_warp/results/contract_v2_baseline_checked_20260923/protocol.json'
SEEDS = (700200, 700202, 700300, 700302)
# M3: differential support, differential hub torque, differential wheel torque.
MODES = ((None, 0.), (1, .5), (1, -.5), (2, .5), (2, -.5))
FOLLOWUP_MODES = (
    (None, 0., (-.60, -.25)),
    (0, .5, (-.60, -.25)), (0, -.5, (-.60, -.25)),
    (0, .5, (-.95, -.60)), (0, -.5, (-.95, -.60)),
    (0, .5, (-.95, -.25)), (0, -.5, (-.95, -.25)),
    (None, 0., (-.60, -.25)),
)


def gated_action(action, progress, touched, component, delta, window=(-.60, -.25)):
    """Apply a clipped offset only in the frozen pre-contact window."""
    if component is None or touched or not window[0] <= progress <= window[1]:
        return action, False
    changed = action.copy()
    changed[component] = np.clip(changed[component] + delta, -1., 1.)
    return changed, bool(changed[component] != action[component])


def run(output, followup=False):
    source = json.loads(PANEL.read_text()); verify(source)
    cases = [TerrainScenario(**row['scenario']) for row in source['panels']]
    selected = {seed: next(i for i, case in enumerate(cases) if case.terrain_seed == seed) for seed in SEEDS}
    assert len(cases) == 160 and len(set(selected.values())) == len(SEEDS)
    checkpoint = source['checkpoints']['terrain_v3']['path']
    modes = FOLLOWUP_MODES if followup else tuple((component, delta, (-.60, -.25)) for component, delta in MODES)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'protocol.json', dict(task_contract_version=TASK_CONTRACT_VERSION,
        panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(), source_sha256=source_hashes(__file__),
        checkpoint=source['checkpoints']['terrain_v3'], worlds=160, seeds=SEEDS, modes=modes,
        training=False, holdout_evaluated=False,
        privileged_position_gate=True,
        note='Terrain-v3 deterministic M3 policy; only four public worlds receive clipped pre-contact action offsets. '
             'Known step position and simulator contact bits are privileged. Other 156 worlds are unchanged.'))
    torch.set_num_threads(1)
    results = []
    for run_id, (component, delta, window) in enumerate(modes, 1):
        raw = RecordedEnv(len(cases), scenario=cases)
        env = VecNormalize.load(checkpoint + '.pkl', raw)
        env.training = False; env.norm_reward = False
        model = PPO.load(checkpoint + '.zip', device='cpu')
        observations = env.reset(); pending = set(range(len(cases)))
        rows = [None] * len(cases); chunks = {i: [] for i in selected.values()}
        used = {i: 0 for i in selected.values()}; first_progress = {i: None for i in selected.values()}
        policy = {i: [] for i in selected.values()}
        try:
            for step in range(1, 1001):
                if not pending: break
                action = model.predict(observations, deterministic=True)[0]
                state = raw.state.numpy()
                for i in selected.values():
                    if i not in pending: continue
                    progress = float(min(state[i, 24:26]))
                    original = action[i].copy()
                    action[i], applied = gated_action(action[i], progress, int(state[i, 14]) & 12,
                                                       component, delta, window)
                    policy[i].append((float(state[i, 0] * .0005), state[i, 24:26].copy(),
                                      original, action[i].copy()))
                    if applied:
                        used[i] += 1
                        if first_progress[i] is None: first_progress[i] = progress
                observations, _, done, infos = env.step(action)
                trace = raw.trace.numpy()
                for i in list(chunks):
                    if i in pending: chunks[i].append(trace[trace[:, i, 0] > 0, i, :].copy())
                for i in list(pending):
                    if done[i]:
                        rows[i] = {key: value for key, value in infos[i].items() if key != 'terminal_observation'}
                        pending.remove(i)
            if pending: raise RuntimeError('Public panel did not terminate within 20 s')
            validate_terrain_rows(rows); verify(source)
            cases_out = {}
            for seed, i in selected.items():
                trace = np.concatenate(chunks[i]); info = rows[i]
                assert len(trace) == info['physical_steps'] and np.isfinite(trace).all()
                diagnosis = analyze(trace)
                np.testing.assert_allclose(diagnosis['reward'], info['episode']['r'], atol=1e-7)
                timeline = policy[i]
                with (output / f'run_{run_id:02d}_{seed}.npz').open('xb') as stream:
                    np.savez_compressed(stream, trace=trace, columns=np.asarray(COL),
                        policy_time=np.asarray([item[0] for item in timeline]),
                        wheel_progress=np.asarray([item[1] for item in timeline]),
                        original_action=np.asarray([item[2] for item in timeline]),
                        used_action=np.asarray([item[3] for item in timeline]))
                cases_out[str(seed)] = dict(info=info, first_intervention_progress_m=first_progress[i],
                    intervention_policy_steps=used[i], **diagnosis)
            result = dict(run=run_id, component=component, delta=delta, window=window,
                all_success=sum(row['success'] for row in rows),
                hard_success=sum(row['success'] for j, row in enumerate(rows) if source['panels'][j]['group'] == 'difficult/step'),
                selected=cases_out)
            if followup:
                result['outcomes'] = [dict(seed=case.terrain_seed, group=source['panels'][i]['group'],
                    info=rows[i]) for i, case in enumerate(cases)]
            write(output / f'run_{run_id:02d}.json', result); results.append(result)
            print('run', run_id, component, delta, window,
                  [(s, cases_out[str(s)]['info']['success']) for s in SEEDS], flush=True)
        finally: env.close()
    write(output / 'summary.json', dict(runs=[dict(run=r['run'], component=r['component'], delta=r['delta'],
        window=r['window'],
        all_success=r['all_success'], hard_success=r['hard_success'],
        selected_success={s: v['info']['success'] for s, v in r['selected'].items()}) for r in results],
        training=False, weights_promoted=False, holdout_evaluated=False,
        note='A small privileged action screen; even a positive case needs paired repeats and regression before claiming capability.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    parser.add_argument('--followup', action='store_true'); parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        a = np.array([.2, .8, -.8], dtype=np.float32)
        assert gated_action(a, -.4, 0, 0, .5)[0][0] > a[0]
        assert not gated_action(a, -.4, 12, 0, .5)[1]
        assert not gated_action(a, -.6, 0, 0, .5, (-.95, -.61))[1]
        assert gated_action(a, -.4, 0, 1, .5)[0][1] == 1.
        assert np.array_equal(a, np.array([.2, .8, -.8], dtype=np.float32))
        print('gate check passed')
    elif args.output is None: parser.error('--output is required unless --check is used')
    else: run(args.output, args.followup)
