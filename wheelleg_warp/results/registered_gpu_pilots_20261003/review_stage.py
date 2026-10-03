"""Read-only registered200k development review; never evaluates gate/final cases."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from training_contract import digest,verify_checkpoint
from pretrain_yaw import selection_key
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
p=json.loads((P/'protocol.json').read_text());run=P/'runs/M3/1609'
rows=[]
for steps in range(20000,200001,20000):
 path=run/f'step_{steps}.json';r=json.loads(path.read_text());assert r['policy_steps']==steps
 assert [x['seed'] for x in r['runs']]==[x['seed'] for x in p['selection']]
 assert [x['scenario'] for x in r['runs']]==[x['scenario'] for x in p['selection']]
 verify_checkpoint(r['path'],r);rows.append(r)
classical=json.loads((P/'classical_selection.json').read_text());b1=classical['selected']['summary']
best=min((r for r in rows if r['summary']['complete']),key=lambda r:selection_key(r['summary'],r['policy_steps']))
last=rows[-1]
fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
x=[r['policy_steps']/1000 for r in rows]
axes[0].plot(x,[r['summary']['success_count'] for r in rows],'o-',label='M3/1609 selection')
axes[0].axhline(b1['success_count'],color='gray',linestyle='--',label='Selected B1');axes[0].set_ylabel('Successful cases /32');axes[0].legend()
axes[1].plot(x,[r['summary']['mean_yaw_score_deg'] for r in rows],'o-')
axes[1].axhline(b1['mean_yaw_score_deg'],color='gray',linestyle='--');axes[1].set_ylabel('Mean yaw score (deg)');axes[1].set_xlabel('Consumed policy steps (thousands)')
for a in axes:
 a.axvline(100,color='orange',alpha=.5,linestyle=':');a.grid(alpha=.2)
fig.tight_layout();fig.savefig(HERE/'M3_1609_stage_200000.png',dpi=160);plt.close(fig)
report=dict(stage_policy_steps=200000,method='M3',seed=1609,protocol_sha256=digest(P/'protocol.json'),
 records=[dict(policy_steps=r['policy_steps'],summary=r['summary'],metrics_sha256=digest(run/f"step_{r['policy_steps']}.json")) for r in rows],
 best=best['summary'],best_policy_steps=best['policy_steps'],last=last['summary'],selected_b1=b1,
 engineering_current_source='initial_runtime_check.json',no_current_method_advantage=best['summary']['success_count']<b1['success_count'] or best['summary']['mean_yaw_score_deg']>=b1['mean_yaw_score_deg'],
 decision='Continue unchanged registered2M limit and full paired comparison;200k is a development stage, not a new budget or gate opening',
 gate_or_final_simulated=False,verifier_sha256=digest(__file__))
(HERE/'M3_1609_stage_200000.json').write_text(json.dumps(report,indent=2)+'\n')
print('REVIEW200k: best',best['policy_steps'],best['summary'],'B1',b1,'no advantage; continue unchanged registered limit')
