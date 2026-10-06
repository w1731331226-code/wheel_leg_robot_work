"""Stored return/task alignment and discount-timescale audit, not PPO attribution."""
import json,math
from pathlib import Path
import numpy as np
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reference_budget_learning_v1'


def compare(left,right):
    assert len(left)==len(right);opposed=[];ties=0
    for a,b in zip(left,right):
        assert a['seed']==b['seed'];gain=a['episode']['r']-b['episode']['r'];j_a=a['rms_deg'][2]*math.sqrt(a['duration_s']/(3.5+1.5*a['task_goal_progress_m']/abs(a['scenario']['speed'])));j_b=b['rms_deg'][2]*math.sqrt(b['duration_s']/(3.5+1.5*b['task_goal_progress_m']/abs(b['scenario']['speed'])))
        if gain>0 and j_a>j_b:opposed.append(dict(case=a['seed'],return_gain=gain,Jpsi_worsening=j_a-j_b,same_success_status=a['success']==b['success']))
        if gain==0:ties+=1
    return dict(rows=len(left),higher_return_but_worse_Jpsi_cases=opposed,return_ties=ties,
        scope='Undiscounted stored episode return differs from discounted PPO objective; state duration, tracking/otherattitude andcommand/smoothness penalties also differ. Not a pureyawreward treatment or trainedcritic error.')


def self_check():
    a=dict(seed=1,episode=dict(r=2.),rms_deg=[0,0,2.],duration_s=5.,task_goal_progress_m=2.,scenario=dict(speed=.7),success=True)
    b={**a,'episode':dict(r=1.),'rms_deg':[0,0,1.]};assert len(compare([a],[b])['higher_return_but_worse_Jpsi_cases'])==1
    assert not compare([b],[a])['higher_return_but_worse_Jpsi_cases'];assert compare([a],[a])['return_ties']==1


def run():
    self_check();proposal=json.loads((OUT/'proposal.json').read_text());gate=json.loads((OUT/'study_gate_review.json').read_text());assert gate['verified'];gamma=proposal['ppo']['gamma'];dt=.02;assert gamma==.99
    sources={};sets={};times=[]
    for seed in proposal['seeds']:
        for panel in ['regular','controlled']:
            for arm in proposal['arms']:
                path=OUT/'runs'/arm/str(seed)/(panel+'_final.json');d=json.loads(path.read_text());sets[(arm,seed,panel)]=d['runs'];sources[str(path.relative_to(OUT))]=sha(path)
                for row in d['runs']:
                    assert row['reason']=='completed';length=row['episode']['l'];assert length==math.ceil(row['physical_steps']/40)
                    bonus=10. if row['success'] else -10.;discount=gamma**(length-1)
                    times.append(dict(arm=arm,seed=seed,panel=panel,case=row['seed'],duration_s=row['duration_s'],policy_transitions=length,success=row['success'],undiscounted_return=row['episode']['r'],undiscounted_dense_component=row['episode']['r']-bonus,terminal_bonus=bonus,initial_state_discounted_terminal_contribution=discount*bonus,terminal_discount_weight=discount))
            path=OUT/'zero_controls'/f'{seed}_{panel}.json';sets[('zero',seed,panel)]=json.loads(path.read_text())['runs'];sources[str(path.relative_to(OUT))]=sha(path)
    comparisons={}
    for seed in proposal['seeds']:
        for panel in ['regular','controlled']:
            for ref in ['L2-plain','L2-constant','zero']:
                comparisons[f'{seed}/{panel}/room--{ref}']=compare(sets[('L2-room',seed,panel)],sets[(ref,seed,panel)])
    assert len(times)==1632
    weight=[r['terminal_discount_weight'] for r in times];lengths=[r['policy_transitions'] for r in times]
    result=dict(verified=True,main_evaluations=1632,zero_comparison_evaluations=408,new_physics_steps=0,training_updates=0,gamma_per_policy_transition=gamma,policy_dt_s=dt,discount_e_folding_time_s=-dt/math.log(gamma),
        terminal_discount=dict(min=min(weight),median=float(np.median(weight)),max=max(weight),min_policy_transitions=min(lengths),max_policy_transitions=max(lengths)),comparisons=comparisons,terminal_terms=times,input_sha256=sources,source_sha256=sha(Path(__file__)),gate_review_sha256=sha(OUT/'study_gate_review.json'),
        reward_source_sha256=sha(ROOT/'wheelleg_warp/native/environment.py'),
        source_interpretation='Dense reward integrates speedtracking, roll/pitch/yaw squared, acceptedresidual magnitude andsmoothness. Originalterminal±10 usesfulltask includingdesign andcontacts/stop/height. These known terms do not make storedtotalreturn identicaltoJpsi orsuccess.',
        limits='Computed terminal discount at initial state only; actual PPOGAE/value/bootstrap can propagate outcomes, not evidence feedbackvanishes. No MonteCarlo critic calibration, trainingadvantage reconstruction or rewardcausal intervention; cannot blame reward alone or authorize a change.',
        next='154 check existing controller-context/capability and candidate researchclaim against necessity;155 deepreview/cleanup before anynewstudy. Currentlearner gates stayfailed.')
    atomic_json(OUT/'objective_alignment_audit.json',result);print('PASS existing1632 returns/408zero pair andterminal-weight audit',result['terminal_discount'],flush=True)


if __name__=='__main__':run()
