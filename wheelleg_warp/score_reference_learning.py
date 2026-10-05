"""Predeclared finite-study gates; refuses partial runs or missing zero controls."""
import json,math
import numpy as np
from reference_budget_train import OUT
from review_yaw_sector import ROOT,sha,success
from dashboard.live_env import atomic_json

BASE_PROTOCOL=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json'


def yaw_gate(values,reference):
    return all(v is not None and v<reference for v in values) and float(np.mean(values))<=min(.85*reference,reference-.05)


def self_check():
    assert yaw_gate([.20,.21,.22],.30)
    assert not yaw_gate([.20,.21,.31],.30)
    assert not yaw_gate([.26,.26,.26],.30)
    assert not yaw_gate([None,.20,.20],.30)
    print('PASS fixed all-seed direction/15percent+0.05deg/missing-score checks')


def load_rows(path,cases):
    r=json.loads(path.read_text());assert len(r['runs'])==len(cases)
    for row,case in zip(r['runs'],cases):
        assert row['seed']==case['seed'] and row['scenario']==case['scenario'] and row['success']==success(row)
        physical=min(row['min_actual_A_leg_m'],row['min_actual_B_leg_m'])>=row['geometric_limit_m'] and row['min_eight_joint_margin_rad']>=0 and max(row['max_actual_torque_excess_Nm'],row['max_command_torque_excess_Nm'])<=1e-6
        assert row['physical_safety_passed']==physical and row['design_joint_passed']==(row['min_active_design_margin_rad']>=0)
    assert r['summary']['success_count']==sum(x['success'] for x in r['runs']) and r['physical']==sum(x['physical_safety_passed'] for x in r['runs']) and r['design']==sum(x['design_joint_passed'] for x in r['runs'])
    complete=all(x['reason']=='completed' for x in r['runs']);assert r['summary']['complete']==complete
    if complete:
        score=[x['rms_deg'][2]*math.sqrt(x['duration_s']/(3.5+1.5*x['task_goal_progress_m']/abs(x['scenario']['speed']))) for x in r['runs']]
        assert math.isclose(r['summary']['mean_yaw_score_deg'],sum(score)/len(score),abs_tol=1e-12)
    else:assert r['summary']['mean_yaw_score_deg'] is None
    return r


def run():
    self_check();p=json.loads((OUT/'proposal.json').read_text());progress=json.loads((OUT/'main_progress.json').read_text());zero=json.loads((OUT/'zero_control_completion.json').read_text());closed=json.loads((OUT/'closed_runs_review_final.json').read_text());contract=json.loads((OUT/'trainer_contract.json').read_text())
    assert progress['status']=='complete' and progress['trained_policy_steps']==2400000 and len(progress['completed_runs'])==12
    assert zero['episodes']==408 and zero['training_updates']==0 and closed['verified'] and len(closed['records'])==12 and closed['trainer_contract_sha256']==sha(OUT/'trainer_contract.json')
    assert all(sha(ROOT/n)==v for n,v in contract['source_sha256'].items());old=json.loads(BASE_PROTOCOL.read_text())['nondegradation']
    zero_map={}
    for rec in zero['records']:
        assert sha(OUT/rec['path'])==rec['sha256'];zero_map[(rec['seed'],rec['panel'])]=load_rows(OUT/rec['path'],p[rec['panel']])
    refs={};models={};inputs={}
    for panel in ['regular','controlled']:
        for label in p['classical_reference_labels']:
            f=OUT/'classical'/f'{label}_{panel}.json';refs[(label,panel)]=load_rows(f,p[panel]);inputs[str(f.relative_to(OUT))]=sha(f)
        for seed in p['seeds']:
            for arm in p['arms']:
                f=OUT/'runs'/arm/str(seed)/(panel+'_final.json');models[(arm,seed,panel)]=load_rows(f,p[panel]);inputs[str(f.relative_to(OUT))]=sha(f)
    mechanism={};advantage={};counts={}
    for panel in ['regular','controlled']:
        a=[models[('L2-room',seed,panel)] for seed in p['seeds']];n=len(p[panel]);comp={}
        for arm in ['L2-plain','L2-constant']:
            b=[models[(arm,seed,panel)] for seed in p['seeds']];pairs=[dict(seed=s,success_gain=x['summary']['success_count']-y['summary']['success_count'],lost_paired_success_cases=[u['seed'] for u,v in zip(x['runs'],y['runs']) if v['success'] and not u['success']],yaw_reduction=None if x['summary']['mean_yaw_score_deg'] is None or y['summary']['mean_yaw_score_deg'] is None else y['summary']['mean_yaw_score_deg']-x['summary']['mean_yaw_score_deg']) for s,x,y in zip(p['seeds'],a,b)]
            complete=all(x['yaw_reduction'] is not None for x in pairs);comp[arm]=dict(pairs=pairs,success_nondegradation=all(x['success_gain']>=0 for x in pairs),yaw_positive_at_least2=complete and sum(x['yaw_reduction']>0 for x in pairs)>=2,yaw_mean_positive=complete and float(np.mean([x['yaw_reduction'] for x in pairs]))>0)
        clean=all(x['physical']==x['design']==n for x in a)
        mechanism[panel]=dict(candidate_clean=clean,comparisons=comp,passed=clean and all(c['success_nondegradation'] and c['yaw_positive_at_least2'] and c['yaw_mean_positive'] for c in comp.values()))
        references=[(name,refs[(name,panel)]) for name in p['classical_reference_labels']]+[('zero_'+str(seed),zero_map[(seed,panel)]) for seed in p['seeds']]
        assert all(r['summary']['complete'] for _,r in references);bestyaw=min(r['summary']['mean_yaw_score_deg'] for _,r in references);bestcount=max(r['summary']['success_count'] for _,r in references);details=[]
        for seed,candidate in zip(p['seeds'],a):
            lost=[];velocity=[];arrival=[];attitude=[]
            for label,reference in references:
                for x,y in zip(candidate['runs'],reference['runs']):
                    if not y['success']:continue
                    if not x['success']:lost.append(dict(reference=label,case=x['seed']))
                    if x['velocity_rmse'] is None or x['velocity_rmse']>old['velocity_multiplier']*y['velocity_rmse']+old['velocity_add_m_s']:velocity.append(dict(reference=label,case=x['seed']))
                    if x['arrival_s'] is None or x['arrival_s']>old['arrival_multiplier']*y['arrival_s']+old['arrival_add_s']:arrival.append(dict(reference=label,case=x['seed']))
                    if x['scenario']['terrain']=='legacy' and any(x['peak_deg'][j]>y['peak_deg'][j]+old['legacy_roll_pitch_add_deg'] for j in [0,1]):attitude.append(dict(reference=label,case=x['seed']))
            details.append(dict(seed=seed,success=candidate['summary']['success_count'],success_count_pass=candidate['summary']['success_count']>=bestcount,lost_reference_success=lost,velocity_failures=velocity,arrival_failures=arrival,legacy_attitude_failures=attitude))
        yaw=yaw_gate([x['summary']['mean_yaw_score_deg'] for x in a],bestyaw)
        passed=clean and yaw and all(d['success_count_pass'] and not(d['lost_reference_success'] or d['velocity_failures'] or d['arrival_failures'] or d['legacy_attitude_failures']) for d in details)
        advantage[panel]=dict(best_reference_success=bestcount,best_reference_Jpsi_deg=bestyaw,candidate_required_mean_upper_deg=min(.85*bestyaw,bestyaw-.05),clean=clean,yaw_gate=yaw,details=details,passed=passed)
        counts[panel]={arm:[dict(seed=s,success=models[(arm,s,panel)]['summary']['success_count'],physical=models[(arm,s,panel)]['physical'],design=models[(arm,s,panel)]['design'],Jpsi=models[(arm,s,panel)]['summary']['mean_yaw_score_deg']) for s in p['seeds']] for arm in p['arms']}
    passed=all(x['passed'] for x in mechanism.values()) and all(x['passed'] for x in advantage.values())
    atomic_json(OUT/'study_gate_review.json',dict(verified=True,main_training_steps=2400000,main_evaluation_episodes=1632,classical_episodes=408,zero_control_episodes=408,total_development_evaluation_episodes=2448,mechanism=mechanism,advantage=advantage,counts=counts,formal_expansion_gate=passed,
        source_inputs_sha256=inputs,proposal_sha256=sha(OUT/'proposal.json'),trainer_contract_sha256=sha(OUT/'trainer_contract.json'),closed_runs_review_sha256=sha(OUT/'closed_runs_review_final.json'),zero_completion_sha256=sha(OUT/'zero_control_completion.json'),reviewer_sha256=sha(__file__),
        disposition='Only fullpredeclared gates admit expansion; failedcurrent learner/benefit branch stops with allnegative data preserved. No bestcheckpoint/arm/seed selection or independent-test claim.'))
    print('PASS complete study audit;formal expansion',passed,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');args=p.parse_args();self_check() if args.self_check else run()
