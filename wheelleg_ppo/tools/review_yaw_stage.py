"""Read-only 200k pilot review. Does not evaluate a policy or change training."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
import torch

from pretrain_yaw import ROOT, summarize, selection_key, write


def review(run):
    run = run.resolve()
    stage = 200000
    output = run/'stage_200000_review.json'
    if output.exists() or (run/'stage_200000_curve.png').exists():
        raise FileExistsError('Stage review already exists; keep the existing evidence')
    if not (run/f'step_{stage}.json').exists():
        print('PENDING: complete 200000-step evaluation is not available')
        return
    manifest = json.loads((ROOT/'tools/results/yaw_precision_v2_2026-09-21/protocol_manifest.json').read_text())
    config = json.loads((run/'run_config.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    cases = manifest['sets']['selection']
    records = [r for r in selection['checkpoints'] if r['steps'] <= stage]
    assert [r['steps'] for r in records] == list(range(20000,stage+1,20000))
    sources = [run/'run_config.json', Path(__file__)]
    summaries = []
    for record in records:
        path = Path(record['path']+'.json')
        if not path.is_absolute():
            path = ROOT.parent/path
        sources.append(path)
        data = json.loads(path.read_text())
        assert not data['smoke'] and data['steps']==record['steps']
        rows = data['runs']
        assert [r['scenario'] for r in rows] == [r['scenario'] for r in cases]
        summary = summarize(rows, [r['seed'] for r in cases])
        assert summary == data['summary']
        assert all(summary[k] == record[k] for k in summary)
        summaries.append(dict(steps=record['steps'], **summary))
    checkpoint = run/f'step_{stage}.zip'
    sources.append(checkpoint)
    model = PPO.load(checkpoint, device='cpu')
    assert model.num_timesteps == stage
    assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
    states = model.policy.optimizer.state_dict()['state']
    assert states and all(torch.isfinite(v).all() for row in states.values() for v in row.values() if torch.is_tensor(v))
    expected_updates = stage // config['config']['ppo']['batch_size'] * config['config']['ppo']['n_epochs']
    assert {int(row['step']) for row in states.values()} == {expected_updates}
    normalization = run/f'step_{stage}.pkl'
    sources.append(normalization)
    with normalization.open('rb') as stream:
        norm = pickle.load(stream)
    assert np.isfinite(norm.obs_rms.mean).all() and np.isfinite(norm.obs_rms.var).all()
    assert (norm.obs_rms.var >= 0).all() and np.isfinite(norm.obs_rms.count)
    assert norm.obs_rms.count >= stage
    baseline_path = ROOT/'tools/results/pretrain_yaw_2026-09-21/summary.json'
    sources.append(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    b0, b1 = baseline['candidates'][0], baseline['selected_b1']
    valid = [r for r in summaries if r['complete']]
    best = min(valid, key=lambda r:selection_key(r,r['steps'])) if valid else None
    best_constraints = None
    if best:
        record = next(r for r in records if r['steps']==best['steps'])
        best_path = Path(record['path']+'.json')
        if not best_path.is_absolute():
            best_path = ROOT.parent/best_path
        rows = json.loads(best_path.read_text())['runs']
        b0_path = ROOT/'tools/results/pretrain_yaw_2026-09-21/B0.json'
        sources.append(b0_path)
        base_rows = json.loads(b0_path.read_text())['runs']
        assert [r['seed'] for r in base_rows] == [r['seed'] for r in rows]
        best_constraints = dict(
            lost_B0_success=[r['seed'] for r,b in zip(rows,base_rows) if b['success'] and not r['success']],
            velocity_violations=[r['seed'] for r,b in zip(rows,base_rows) if r['velocity_rmse']>1.05*b['velocity_rmse']+.005],
            arrival_violations=[r['seed'] for r,b in zip(rows,base_rows) if r['arrival_s']>1.05*b['arrival_s']+.05])
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    result = dict(protocol='yaw-precision-v2',method=config['method'],seed=config['seed'],stage_steps=stage,
                  stage_engineering_pass=True,optimizer_minibatch_updates=expected_updates,
                  normalization_count=float(norm.obs_rms.count),normalization_finite=True,
                  all_selection_tasks_completed=all(r['complete'] for r in summaries),
                  latest=summaries[-1],best_within_stage=best,history=summaries,
                  best_selection_constraints=best_constraints,nominal_28_case_regression_evaluated=False,
                  B0=dict(success_count=b0['success_count'],mean_yaw_score_deg=b0['mean_yaw_score_deg']),
                  B1=dict(success_count=b1['success_count'],mean_yaw_score_deg=b1['mean_yaw_score_deg']),
                  segmented_run=bool(config.get('resume_from')),formal_gate_passed=False,
                  gate_evaluated=False,test_set_evaluated=False,
                  source_sha256={str(p.resolve().relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    x = [r['steps']/1000 for r in summaries]
    fig, axes = plt.subplots(2,1,figsize=(8,6),sharex=True)
    axes[0].plot(x,[r['success_count'] for r in summaries],'o-',label=config['method'])
    axes[0].axhline(b0['success_count'],color='gray',linestyle='--',label='B0 / B1')
    axes[0].set(ylabel='Successes / 32',ylim=(0,33))
    axes[1].plot(x,[r['mean_yaw_score_deg'] if r['complete'] else np.nan for r in summaries],'o-',label=config['method'])
    axes[1].axhline(b0['mean_yaw_score_deg'],color='gray',linestyle='--',label='B0')
    axes[1].axhline(b1['mean_yaw_score_deg'],color='green',linestyle='--',label='B1')
    axes[1].set(ylabel='Normalized heading error (deg)',xlabel='Policy steps (thousands)')
    for axis in axes:
        if config.get('resume_from'):
            axis.axvline(config['start_policy_steps']/1000,color='red',alpha=.5,linestyle=':',label='Segment restart')
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.suptitle(f"{config['method']} seed {config['seed']}: selection-set pilot review")
    fig.tight_layout()
    fig.savefig(run/'stage_200000_curve.png',dpi=150)
    plt.close(fig)
    write(output,result)
    print(json.dumps({k:result[k] for k in ('stage_engineering_pass','latest','best_within_stage','B0','B1')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    review(args.run)
