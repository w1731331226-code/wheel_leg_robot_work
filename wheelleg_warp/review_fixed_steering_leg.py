"""Original-rule and paired2x2 fixed-context audit, never promotes a learner."""
import json,math
import numpy as np
from fixed_steering_leg import OUT,verify,CONDITIONS
from review_yaw_sector import sha,success
from dashboard.live_env import atomic_json


def run():
    verify();p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'completion.json').read_text());ledger=json.loads((OUT/'completed_jobs.json').read_text());assert c['completed_episodes']==ledger['completed_episodes']==1632 and c['records']==ledger['records'] and c['training_updates']==0 and c['source_contract_sha256']==sha(OUT/'source_contract.json')
    results={};summary={}
    for job in c['records']:
        assert sha(OUT/job['path'])==job['sha256'];d=json.loads((OUT/job['path']).read_text());cases=p['cases'][job['panel']];assert len(d['runs'])==job['episodes']==len(cases);scores=[]
        leg,assist=CONDITIONS[job['condition']]
        for r,case in zip(d['runs'],cases):
            assert r['seed']==case['seed'] and r['scenario']==case['scenario'] and r['success']==success(r)
            physical=min(r['min_actual_A_leg_m'],r['min_actual_B_leg_m'])>=r['geometric_limit_m'] and r['min_eight_joint_margin_rad']>=0 and max(r['max_actual_torque_excess_Nm'],r['max_command_torque_excess_Nm'])<=1e-6
            assert r['physical_safety_passed']==physical and r['design_joint_passed']==(r['min_active_design_margin_rad']>=0) and r['physical_evidence_steps']==r['physical_steps']>0
            a=np.array(r['leg_room_stats']);w=np.array(r['steering_stats']);assert a.shape==(18,) and w.shape==(8,) and np.isfinite(a).all() and np.isfinite(w).all()
            assert a[0]==a[13]==w[0]==r['physical_steps'] and a[6]==a[8]==a[9]==0
            assert abs(a[15]/a[0]-r['mean_residual_lambda'])<1e-12 and abs(w[6]-a[15])<1e-8 and w[7]<1e-9
            if not leg:assert a[3]==a[4]==a[5]==0 and a[11]==a[13]
            if not assist:assert np.max(np.abs(w[1:6]))==0
            scores.append(r['rms_deg'][2]*math.sqrt(r['duration_s']/(3.5+1.5*r['task_goal_progress_m']/abs(r['scenario']['speed']))) if r['reason']=='completed' else None)
        assert d['summary']['success_count']==sum(r['success'] for r in d['runs']) and d['physical']==sum(r['physical_safety_passed'] for r in d['runs']) and d['design']==sum(r['design_joint_passed'] for r in d['runs']) and d['summary']['complete']==all(s is not None for s in scores)
        if all(s is not None for s in scores):assert math.isclose(d['summary']['mean_yaw_score_deg'],sum(scores)/len(scores),abs_tol=1e-12)
        key=(job['seed'],job['condition'],job['panel']);assert key not in results;results[key]=d
        w=np.sum([r['steering_stats'] for r in d['runs']],axis=0)
        summary['/'.join(map(str,key))]=dict(success=d['summary']['success_count'],physical=d['physical'],design=d['design'],Jpsi=d['summary']['mean_yaw_score_deg'],filtered_vs_raw_steering_RMS=math.sqrt(w[2]/w[1]) if w[1]>0 else None,diag_accepted_vs_filtered_RMS=math.sqrt(w[3]/w[2]) if w[2]>0 else None,executed_left_vs_diag_RMS=math.sqrt(w[4]/w[3]) if w[3]>0 else None)
    expected={(m['seed'],condition,panel) for m in p['models'] for condition in CONDITIONS for panel in ['regular','controlled']};assert set(results)==expected
    paired={}
    for panel in ['regular','controlled']:
        pairs=[]
        for m in p['models']:
            seed=m['seed'];zero=results[(seed,'zero_leg_no_assist',panel)];plain=results[(seed,'learned_room_leg_no_assist',panel)];fixed=results[(seed,'zero_leg_fixed_assist',panel)];hybrid=results[(seed,'learned_room_leg_fixed_assist',panel)]
            def comparison(a,b):
                aj=a['summary']['mean_yaw_score_deg'];bj=b['summary']['mean_yaw_score_deg']
                return dict(success_gain=a['summary']['success_count']-b['summary']['success_count'],Jpsi_reduction=bj-aj if aj is not None and bj is not None else None,lost_success_cases=[r['seed'] for r,s in zip(a['runs'],b['runs']) if s['success'] and not r['success']],gained_success_cases=[r['seed'] for r,s in zip(a['runs'],b['runs']) if not s['success'] and r['success']])
            pairs.append(dict(seed=seed,fixed_steering_alone=comparison(fixed,zero),leg_without_assist=comparison(plain,zero),leg_with_assist=comparison(hybrid,fixed),assist_to_learned=comparison(hybrid,plain)))
        paired[panel]=pairs
    consistent_increment={panel:all(r['leg_with_assist']['success_gain']>=0 and not r['leg_with_assist']['lost_success_cases'] and r['leg_with_assist']['Jpsi_reduction'] is not None and r['leg_with_assist']['Jpsi_reduction']>0 for r in pairs) for panel,pairs in paired.items()}
    atomic_json(OUT/'review.json',dict(verified=True,completed_episodes=1632,unique_development_cases=136,training_updates=0,summary=summary,paired=paired,consistent_leg_increment_by_panel=consistent_increment,
        completion_sha256=sha(OUT/'completion.json'),source_contract_sha256=sha(OUT/'source_contract.json'),reviewer_sha256=sha(__file__),
        limits='Fixedpolicycontext factorial, rawpacket/oldscope/actions/source/unit andcumulativephysicschecks; not3newtrainingseedmethod comparison, externalfullphysicsreconstruction or independentgeneralization. Global-lambda affectsacceptedsteering; requestidentity not acceptedtorqueidentity. Oldstronggate remainsfailed.',
        disposition='No consistentincrement beyondpurefixedsteering closes thiscontext hypothesis as directlearningroute; positive onlyjustifies newindependentqualification, notoldmodelpromotion or automaticretraining.'))
    print('PASS1632 originalrule/factor/steeringacceptance audit',consistent_increment,flush=True)


if __name__=='__main__':run()
