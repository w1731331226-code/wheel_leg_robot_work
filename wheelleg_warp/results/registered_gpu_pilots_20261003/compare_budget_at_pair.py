"""Same-seed same-budget development comparison without new evaluations."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from training_contract import digest,verify_checkpoint
from pretrain_yaw import selection_key
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=200000);parser.add_argument('--seed',type=int,default=1609);a=parser.parse_args()
p=json.loads((P/'protocol.json').read_text());assert a.steps%p['evaluation_interval']==0
stem=f'paired_M3_B2V_{a.seed}_{a.steps}';out=HERE/(stem+'.json');assert not out.exists()
methods={}
for method in ('M3','B2-V'):
    rows=[]
    for n in range(p['evaluation_interval'],a.steps+1,p['evaluation_interval']):
        path=P/'runs'/method/str(a.seed)/f'step_{n}.json';r=json.loads(path.read_text());verify_checkpoint(r['path'],r)
        assert [x['seed'] for x in r['runs']]==[x['seed'] for x in p['selection']]
        assert [x['scenario'] for x in r['runs']]==[x['scenario'] for x in p['selection']]
        rows.append(r)
    best=min((r for r in rows if r['summary']['complete']),key=lambda r:selection_key(r['summary'],r['policy_steps']))
    methods[method]=dict(best_steps=best['policy_steps'],best=best['summary'],last=rows[-1]['summary'],series=[dict(policy_steps=r['policy_steps'],**r['summary']) for r in rows])
fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
for method,d in methods.items():
    x=[r['policy_steps']/1000 for r in d['series']]
    axes[0].plot(x,[r['success_count'] for r in d['series']],'o-',ms=3,label=method)
    axes[1].plot(x,[r['mean_yaw_score_deg'] for r in d['series']],'o-',ms=3,label=method)
b1=json.loads((P/'classical_selection.json').read_text())['selected']['summary']
axes[0].axhline(b1['success_count'],color='gray',linestyle='--');axes[1].axhline(b1['mean_yaw_score_deg'],color='gray',linestyle='--')
axes[0].set_ylabel('Success /32');axes[1].set_ylabel('Mean yaw score(deg)');axes[1].set_xlabel('Consumed policy steps(thousands)')
for axis in axes:axis.legend();axis.grid(alpha=.2)
fig.tight_layout();fig.savefig(HERE/(stem+'.png'),dpi=150);plt.close(fig)
report=dict(seed=a.seed,equal_budget=a.steps,methods=methods,strong_classical=b1,selection_cases=32,configuration_or_budget_changed=False,
 inference='Early single-seed development comparison only. Full-budget3seed gate remains required; dimension/covariance/reachable-set differences retained.',
 extra_task_evaluations=0,gate_or_final_simulated=False,protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__))
out.write_text(json.dumps(report,indent=2)+'\n');print('PAIRED',a.steps,{m:(d['best_steps'],d['best']) for m,d in methods.items()})
