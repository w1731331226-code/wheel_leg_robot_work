"""Pre-existing design-node intervals, same-budget selected policies; no new scoring."""
from pathlib import Path
from collections import Counter
import json,sys,math
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from training_contract import digest,verify_checkpoint
from pretrain_yaw import selection_key
HERE=Path(__file__).resolve().parent;P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3';p=json.loads((P/'protocol.json').read_text())
limits=(.115,.16,.25,.30,.38);reports={}
for method in ('M3','B2-V','B2'):
    rows=[json.loads((P/'runs'/method/'1609'/f'step_{n}.json').read_text()) for n in range(20000,1000001,20000)]
    best=min((r for r in rows if r['summary']['complete']),key=lambda r:selection_key(r['summary'],r['policy_steps']));verify_checkpoint(best['path'],best)
    assert [r['scenario'] for r in best['runs']]==[r['scenario'] for r in p['selection']]
    groups=[]
    for i,(low,high) in enumerate(zip(limits[:-1],limits[1:])):
        selected=[r for r in best['runs'] if low<=r['target_leg_m'] and (r['target_leg_m']<high or i==3 and r['target_leg_m']<=high)]
        failures=Counter()
        for r in selected:
            if not r['physical_safety_passed']:failures['physical']+=1
            if not r['design_joint_passed']:failures['design']+=1
            if not r['terrain_evidence_passed']:failures['terrain']+=1
            if r['touched_contact_mask']&r['required_contact_mask']!=r['required_contact_mask']:failures['original_contact']+=1
            if max(r['relative_peak_deg'] if r['scenario'].get('relative_attitude') else r['peak_deg'])>5 or r['peak_deg'][2]>5:failures['attitude']+=1
        groups.append(dict(interval_m=[low,high],cases=len(selected),success=sum(r['success'] for r in selected),nonexclusive_failures=dict(failures),case_ids=[r['seed'] for r in selected]))
    assert sum(g['cases'] for g in groups)==32
    reports[method]=dict(selected_steps=best['policy_steps'],summary=best['summary'],groups=groups)
result=dict(seed=1609,equal_budget=1000000,group_boundaries='Existing115/160/250/300/380mm design nodes; no outcome-fitted cuts',methods=reports,
 inference='Small correlated development subsets with different terrain/parameters; no significance or height-causality claim, no exclusion of failed cases',
 extra_task_evaluations=0,gate_or_final_simulated=False,protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__))
(HERE/'height_group_audit_1609_1m.json').write_text(json.dumps(result,indent=2)+'\n')
for method,r in reports.items():print(method,[(g['interval_m'],g['success'],g['cases'],g['nonexclusive_failures']) for g in r['groups']])
