"""Descriptive last-window variability under identical registered development budget."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from training_contract import digest,verify_checkpoint
from pretrain_yaw import selection_key
HERE=Path(__file__).resolve().parent;P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=1300000);a=parser.parse_args();p=json.loads((P/'protocol.json').read_text());interval=p['evaluation_interval'];assert a.steps%interval==0
out=HERE/f'paired_stability_1609_{a.steps}.json';assert not out.exists();reports={}
for method in ('M3','B2-V'):
 rows=[]
 for n in range(interval,a.steps+1,interval):
  r=json.loads((P/'runs'/method/'1609'/f'step_{n}.json').read_text());verify_checkpoint(r['path'],r)
  assert [v['scenario'] for v in r['runs']]==[v['scenario'] for v in p['selection']]
  rows.append(r)
 best=min((r for r in rows if r['summary']['complete']),key=lambda r:selection_key(r['summary'],r['policy_steps']));window=rows[-10:]
 scores=[r['summary']['mean_yaw_score_deg'] for r in window];success=[r['summary']['success_count'] for r in window]
 assert all(r['summary']['complete'] for r in window)
 reports[method]=dict(best_steps=best['policy_steps'],best=best['summary'],last_window_policy_steps=[r['policy_steps'] for r in window],
  last_window_success_mean=float(np.mean(success)),last_window_success_range=[min(success),max(success)],
  last_window_yaw_mean=float(np.mean(scores)),last_window_yaw_std=float(np.std(scores)),last_window_yaw_range=[min(scores),max(scores)],
  every_window_point_matches_selected_best=all(r['summary']==best['summary'] for r in window))
result=dict(seed=1609,equal_budget=a.steps,window_evaluations=10,methods=reports,
 inference='Descriptive correlated development checkpoints, not independent samples/confidence intervals or proof of convergence. Original best-selection rule retained.',
 extra_task_evaluations=0,gate_or_final_simulated=False,protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__))
out.write_text(json.dumps(result,indent=2)+'\n');print('STABILITY',reports)
