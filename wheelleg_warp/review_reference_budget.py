"""Read-only original-rule and per-transition property audit of GPU pilot."""
import json,math
import numpy as np
from review_yaw_sector import ROOT,sha,success
from pilot_reference_budget import OUT,verify
from dashboard.live_env import atomic_json


def height_bin(h):
    if h<.16:return 0
    if h<.24:return 1
    if h<.30:return 2
    if h<.38:return 3
    return 4


def run():
    contract=verify();p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'completion.json').read_text());ledger=json.loads((OUT/'completed_jobs.json').read_text())
    assert c['completed_episodes']==ledger['completed_episodes']==1840 and c['training_updates']==0 and c['runtime_contract_sha256']==sha(OUT/'runtime_contract.json') and c['records']==ledger['records']
    results={};counts={}
    for job in c['records']:
        assert sha(OUT/job['path'])==job['sha256'];d=json.loads((OUT/job['path']).read_text());assert len(d['runs'])==job['episodes']==184
        for row,case in zip(d['runs'],p['cases']):
            assert row['seed']==case['seed'] and row['scenario']==case['scenario'] and row['success']==success(row)
            physical=min(row['min_actual_A_leg_m'],row['min_actual_B_leg_m'])>=row['geometric_limit_m'] and row['min_eight_joint_margin_rad']>=0 and max(row['max_actual_torque_excess_Nm'],row['max_command_torque_excess_Nm'])<=1e-6
            assert row['physical_safety_passed']==physical and row['design_joint_passed']==(row['min_active_design_margin_rad']>=0)
            assert row['physical_evidence_steps']==row['physical_steps']>0 and math.isclose(row['duration_s'],row['physical_steps']*.0005,abs_tol=1e-8)
            a=np.array(row['reference_budget_stats']);assert a.shape==(18,) and np.isfinite(a).all()
            assert a[0]==row['physical_steps'] and a[13]+a[9]==a[0] and 0<=a[3]<=a[0] and a[6]==a[8]==0
            assert 0<=a[2]<=a[10]<=1 and 0<=a[1]<=a[13]+1e-8 and 0<=a[5]<=a[4]+1e-8 and 0<=a[15]<=a[13]+1e-8
            if job['label']=='B1':assert a[3]==0 and a[4]==a[5]==0 and a[11]==a[13]
            elif job['label'].endswith('_original'):assert a[3]==0 and a[4]==a[5] and a[1]==a[13]
            elif job['label'].endswith('_zero_leg_after_filter'):assert a[5]==a[1]==0
        assert d['summary']['success_count']==sum(r['success'] for r in d['runs']) and d['physical']==sum(r['physical_safety_passed'] for r in d['runs']) and d['design']==sum(r['design_joint_passed'] for r in d['runs'])
        assert d['summary']==job['summary'] and d['physical']==job['physical'] and d['design']==job['design']
        results[job['label']]=d;counts[job['label']]=dict(success=d['summary']['success_count'],physical=d['physical'],design=d['design'])
    required=['B1']+[str(m['seed'])+'_'+name for m in p['models'] for name in ['original','reference_budget','zero_leg_after_filter']];assert list(results)==required
    gates={};authority={};paired={}
    for m in p['models']:
        seed=str(m['seed']);o=results[seed+'_original'];t=results[seed+'_reference_budget'];z=results[seed+'_zero_leg_after_filter'];bins={}
        for k in range(5):
            rows=[r for r in t['runs'] if height_bin(r['target_leg_m'])==k];old=sum(r['reference_budget_stats'][4] for r in rows);new=sum(r['reference_budget_stats'][5] for r in rows)
            bins[str(k)]=dict(episodes=len(rows),same_state_command_rms_ratio=math.sqrt(new/old) if old>0 else None)
        authority[seed]=bins
        valid=all(r['reference_budget_stats'][9]==0 for r in t['runs']);nondegenerate=all(b['same_state_command_rms_ratio'] is not None and b['same_state_command_rms_ratio']>=.10 for b in bins.values())
        gates[seed]=dict(physical_clean=t['physical']==184,design_clean=t['design']==184,task_nonregression=t['summary']['success_count']>=o['summary']['success_count'],valid_operator=valid,nondegenerate=nondegenerate)
        paired[seed]=dict(candidate_minus_original=t['summary']['success_count']-o['summary']['success_count'],candidate_minus_zero=t['summary']['success_count']-z['summary']['success_count'],candidate_minus_B1=t['summary']['success_count']-results['B1']['summary']['success_count'],lost_original_success_cases=[r['seed'] for r,b in zip(t['runs'],o['runs']) if b['success'] and not r['success']])
    gate=all(all(v.values()) for v in gates.values())
    atomic_json(OUT/'review.json',dict(verified=True,completed_episodes=1840,unique_development_cases=184,training_updates=0,counts=counts,authority=authority,gates=gates,paired=paired,qualification_gate=gate,
        completion_sha256=sha(OUT/'completion.json'),runtime_contract_sha256=sha(OUT/'runtime_contract.json'),reviewer_sha256=sha(__file__),
        limits='Original task/physical/design checks reconstructed; per-step cumulative GPU checks verified through frozen kernel/unit. No full independent raw-trajectory reconstruction, learned-method/strong-reference superiority, state-invariance or new independent-test claim.',
        disposition='Candidate qualification only if all prescribed gates; failures stop fixed candidate without normalization/threshold sweep. Never automatically launch PPO.'))
    print('PASS1840 episodes audit;qualification_gate',gate,counts,flush=True)


if __name__=='__main__':run()
