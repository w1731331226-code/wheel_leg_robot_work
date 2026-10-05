"""Existing-data gate component/authority/pair diagnostic; no new rollouts."""
import json,math
from collections import Counter
import numpy as np
from analyze_reward_failures import flags
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reference_budget_learning_v1'


def run():
    gate=json.loads((OUT/'study_gate_review.json').read_text());assert gate['verified'] and not gate['formal_expansion_gate'];p=json.loads((OUT/'proposal.json').read_text());sources={};results={};rows={}
    paths=[(f'{a}/{s}/{panel}',OUT/'runs'/a/str(s)/(panel+'_final.json')) for s in p['seeds'] for a in p['arms'] for panel in ['regular','controlled']]
    paths +=[(f'{r}/{panel}',OUT/'classical'/f'{r}_{panel}.json') for r in p['classical_reference_labels'] for panel in ['regular','controlled']]
    paths +=[(f'zero/{s}/{panel}',OUT/'zero_controls'/f'{s}_{panel}.json') for s in p['seeds'] for panel in ['regular','controlled']]
    total=0
    for key,path in paths:
        d=json.loads(path.read_text());panel=key.split('/')[-1];decoded=[flags(r) for r in d['runs']];assert len(d['runs'])==len(p[panel])
        for row,case,f in zip(d['runs'],p[panel],decoded):assert row['seed']==case['seed'] and row['scenario']==case['scenario'] and row['success']==(not any(f.values()))
        counts={name:sum(f[name] for f in decoded) for name in decoded[0]};bad=[dict(case=r['seed'],terrain=r['scenario']['terrain'],height_m=r['target_leg_m'],flags=[k for k,v in f.items() if v],yaw_peak_deg=r['peak_deg'][2]) for r,f in zip(d['runs'],decoded) if any(f.values())]
        stats=[r.get('allocation_stats',r.get('reference_budget_stats')) for r in d['runs']]
        authority=None
        if all(x is not None for x in stats):
            old=sum(x[4] for x in stats);new=sum(x[5] for x in stats);samples=sum(x[13] for x in stats)
            authority=dict(same_state_leg_command_RMS_ratio=math.sqrt(new/old) if old else None,mean_applied_factor=sum(x[1] for x in stats)/samples if samples else None,changed_physical_steps=sum(x[3] for x in stats),valid_samples=samples,invalid_samples=sum(x[9] for x in stats))
        results[key]=dict(total=len(decoded),success=d['summary']['success_count'],overlapping_failure_counts=counts,failures=bad,authority=authority);rows[key]={r['seed']:(r,f) for r,f in zip(d['runs'],decoded)};sources[str(path.relative_to(OUT))]=sha(path);total+=len(decoded)
    assert total==2448
    pairs={}
    for s in p['seeds']:
        for panel in ['regular','controlled']:
            candidate=rows[f'L2-room/{s}/{panel}']
            for ref in [f'L2-plain/{s}/{panel}',f'L2-constant/{s}/{panel}',f'zero/{s}/{panel}',f'B0/{panel}',f'B1/{panel}',f'B1-route/{panel}']:
                reference=rows[ref];lost=[];gained=[]
                for case,(a,af) in candidate.items():
                    b,bf=reference[case]
                    if b['success'] and not a['success']:lost.append(dict(case=case,candidate_failures=[k for k,v in af.items() if v],reference_failures=[]))
                    if not b['success'] and a['success']:gained.append(dict(case=case,removed_reference_flags=[k for k,v in bf.items() if v]))
                pairs[f'L2-room/{s}/{panel}--{ref}']=dict(lost=lost,gained=gained,success_difference=len(gained)-len(lost))
    candidate_keys=[f'L2-room/{s}/{panel}' for s in p['seeds'] for panel in ['regular','controlled']]
    allbad=[r for key in candidate_keys for r in results[key]['failures']]
    all_counts=Counter(f for r in allbad for f in r['flags'])
    output=dict(verified=True,original_gate_reconstructed_evaluations=2448,unique_development_cases=136,candidate_failed_trajectories=len(allbad),candidate_unique_failure_cases=len({r['case'] for r in allbad}),candidate_overlapping_failure_counts=dict(all_counts),results=results,pairs=pairs,input_sha256=sources,gate_review_sha256=sha(OUT/'study_gate_review.json'),auditor_sha256=sha(__file__),new_physics_steps=0,training_updates=0,
        interpretation='Terminal flags overlap, pair/authority/height/terrain distributions are descriptive, not unique causal explanations. Source shows L2 excludes learnedwheel requests whileB1/route add fixedwheel feedback; V6 has6D authority yet underperforms, so missingwheel freedom alone is not demonstrated cause. No criticMC calibration or reward intervention performed.',
        next='150 assess whether to test common strongestNom/controlledwheel allocation or other specific supported explanation; no automaticnewtraining or difficulty/metric change. Currentclosedlearner remains failed.')
    atomic_json(OUT/'failure_component_audit.json',output);print('PASS2448 existing-row flag/authority/pair reconstruction;candidate failures',len(allbad),dict(all_counts),flush=True)


if __name__=='__main__':run()
