"""Read-only complete route pilot audit; no physics reruns or policy selection."""
from pathlib import Path
import argparse,json,pickle,math,warnings,csv,gc
import numpy as np
from stable_baselines3 import PPO
from review_yaw_sector import sha,success,ROOT
from smoke_reward_training import check_agent,weight_digest
warnings.filterwarnings('ignore',message='You are trying to run PPO on the GPU.*',category=UserWarning)


def panel(path,cases):
    d=json.loads(path.read_text());rows=d['runs'];assert len(rows)==len(cases);scores=[]
    patterns=dict(contact=0,yaw=0,design=0,noncompleted=0)
    for r,c in zip(rows,cases):
        assert r['seed']==c['seed'] and r['scenario']==c['scenario'] and type(r['success']) is bool and r['success']==success(r)
        assert r['physical_evidence_steps']==r['physical_steps']>0
        physical=min(r['min_actual_A_leg_m'],r['min_actual_B_leg_m'])>=r['geometric_limit_m'] and r['min_eight_joint_margin_rad']>=0 and max(r['max_actual_torque_excess_Nm'],r['max_command_torque_excess_Nm'])<=1e-6
        assert r['physical_safety_passed']==physical and r['design_joint_passed']==(r['min_active_design_margin_rad']>=0)
        assert r['terrain_evidence_passed']==bool(r['terrain_passed'] and r['terrain_exit_passed'])
        duration=r['physical_steps']*.0005;assert math.isclose(duration,r['duration_s'],abs_tol=1e-8)
        if r['reason']=='completed':
            assert r['arrival_s'] is not None and duration>=r['arrival_s']+2-1e-8 and math.isfinite(r['rms_deg'][2])
            scores.append(r['rms_deg'][2]*math.sqrt(duration/(3.5+1.5*r['task_goal_progress_m']/abs(r['scenario']['speed']))))
        else:scores.append(None)
        patterns['contact']+=int(not r['terrain_passed'] or (r['touched_contact_mask']&r['required_contact_mask'])!=r['required_contact_mask'])
        patterns['yaw']+=int(r['peak_deg'][2]>5);patterns['design']+=int(not r['design_joint_passed']);patterns['noncompleted']+=int(r['reason']!='completed')
    complete=all(v is not None for v in scores);assert d['summary']['complete']==complete
    assert d['summary']['total']==len(rows) and d['summary']['success_count']==sum(r['success'] for r in rows)
    if complete:assert math.isclose(sum(scores)/len(scores),d['summary']['mean_yaw_score_deg'],abs_tol=1e-12)
    else:assert d['summary']['mean_yaw_score_deg'] is None
    assert d['physical']==sum(r['physical_safety_passed'] for r in rows) and d['design']==sum(r['design_joint_passed'] for r in rows)
    return d,patterns


def run(out):
    proposal=json.loads((out/'proposal.json').read_text());contract=json.loads((out/'trainer_contract.json').read_text());queue=json.loads((out/'queue_progress.json').read_text())
    assert contract['proposal_sha256']==sha(out/'proposal.json') and contract['source_sha256']==proposal['source_sha256']
    assert all(sha(ROOT/n)==v for n,v in proposal['source_sha256'].items())
    for name,key in [('engineering_verification.json','engineering_sha256'),('initial_effort_audit.json','scale_sha256'),('classical_feasibility.json','classical_sha256')]:assert sha(out/name)==contract[key]
    assert queue['status']=='complete' and queue['training_steps']==proposal['total_training_budget']==2400000
    jobs=[dict(arm=a,seed=s) for s in proposal['training_seeds'] for a in proposal['arms']];assert queue['completed_runs']==jobs
    base=json.loads((ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json').read_text())
    stats={};rows={};patterns={};hashes={};inits={};checkpoints=0;training_episodes=0;table=[]
    for label in proposal['classical']:
        for p in ['regular','controlled']:
            path=out/'classical'/(label+'_'+p+'.json');d,f=panel(path,proposal[p]);key=label+'/'+p
            stats[key]={k:v for k,v in d.items() if k!='runs'};rows[key]=d['runs'];patterns[key]=f;hashes[str(path.relative_to(out))]=sha(path)
    for job in jobs:
        arm,seed=job['arm'],job['seed'];folder=out/'runs'/arm/str(seed);label=f'{arm}/{seed}'
        v=json.loads((folder/'verification.json').read_text());init=json.loads((folder/'initialization.json').read_text());inits[label]=init
        assert v['verified'] and v['policy_steps']==200000 and v['ppo_epochs']==400 and v['adam_updates']==8000
        assert v['source_sha256']==proposal['source_sha256'] and v['observations']==39 and v['dimensions']==proposal['arms'][arm]['dimensions']
        assert not v['engineering_only'] and not v['weights_promoted'] and not v['old_gate_or_final_used'] and v['forced_physical_or_route_resets_midrun']==0
        assert v['initial_weight_sha256']==init['weight_sha256'] and v['checkpoints']==list(range(20000,200001,20000))
        episodes=json.loads((folder/'episodes.json').read_text());assert len(episodes['episodes'])==v['completed_episodes'];training_episodes+=len(episodes['episodes'])
        banks={int(stage):{r['seed']:r['scenario'] for r in cases} for stage,cases in base['training_banks'][str(seed)].items()}
        for r in episodes['episodes']:
            assert r['scenario']==banks[r['curriculum_stage']][r['seed']]
            assert r['physical_evidence_steps']==r['physical_steps']>0 and math.isclose(r['duration_s'],r['physical_steps']*.0005,abs_tol=1e-8)
        stages=np.ones(100,int)
        for t in episodes['curriculum_transitions']:
            assert t['actual_episode_end'] and t['stage'] in (2,3) and t['policy_steps']>=proposal['curriculum_milestones'][t['stage']-2]
            for w in t['worlds']:assert 0<=w<100 and stages[w]<t['stage'];stages[w]=t['stage']
        assert np.all(stages==3)
        for step in v['checkpoints']:
            prefix=folder/f'step_{step}';c=json.loads(prefix.with_suffix('.json').read_text());epochs=step//5000*10;adam=epochs*20
            assert c['policy_steps']==c['trained_steps']==step and c['ppo_epochs']==epochs and c['adam_updates']==adam
            assert c['physical_Phi_and_RNG_unchanged_by_save'] and c['route_memory_unchanged_by_save'] and not c['checkpoint_promoted']
            assert all(x==0 for x in c['ongoing_phi']) and c['nonzero_shaping_transitions']==0
            b=c['last_rollout_bootstrap'];assert b['policy_steps']==step and b['pre_update_epoch_counter']==epochs-10
            assert len(b['values'])==len(b['dones'])==100 and np.isfinite(b['values']).all() and all(type(x) is bool for x in b['dones'])
            np.testing.assert_allclose(c['ongoing_route_clock_s'],np.asarray(c['ongoing_episode_policy_steps'])*.02,rtol=0,atol=1e-10)
            assert len(c['ongoing_route_y_m'])==100 and np.isfinite(c['ongoing_route_y_m']).all()
            assert sha(prefix.with_suffix('.zip'))==c['checkpoint']['checkpoint_sha256'] and sha(prefix.with_suffix('.pkl'))==c['checkpoint']['normalization_sha256']
            agent=PPO.load(str(prefix)+'.zip',device='cuda');check_agent(agent,adam)
            assert agent.num_timesteps==step and agent._n_updates==epochs and agent.observation_space.shape==(39,) and agent.action_space.shape==(v['dimensions'],)
            assert weight_digest(agent)!=init['weight_sha256']
            with open(str(prefix)+'.pkl','rb') as stream:norm=pickle.load(stream)
            assert norm.obs_rms.mean.shape==(39,) and np.isfinite(norm.obs_rms.mean).all() and np.isfinite(norm.obs_rms.var).all()
            if arm=='M3-zero':assert norm.obs_rms.mean[38]==0
            checkpoints+=1;hashes[str(prefix.with_suffix('.json').relative_to(out))]=sha(prefix.with_suffix('.json'));del agent,norm;gc.collect()
        for p in ['regular','controlled']:
            path=folder/(p+'_final.json');d,f=panel(path,proposal[p]);key=label+'/'+p
            stats[key]={k:v for k,v in d.items() if k!='runs'};rows[key]=d['runs'];patterns[key]=f;hashes[str(path.relative_to(out))]=sha(path)
            table.append(dict(seed=seed,arm=arm,panel=p,total=d['summary']['total'],success=d['summary']['success_count'],yaw_score_deg=d['summary']['mean_yaw_score_deg'],physical=d['physical'],design=d['design'],**f))
        print('VERIFIED',label,'checkpoints',checkpoints,flush=True)
    for seed in proposal['training_seeds']:
        a,b=[inits[f'{arm}/{seed}'] for arm in ['M3-zero','M3-route']]
        assert a['weight_sha256']==b['weight_sha256'] and a['world_sha256']==b['world_sha256']
    pairs=[];swaps={}
    for seed in proposal['training_seeds']:
        a,b=[stats[f'{arm}/{seed}/regular'] for arm in ['M3-zero','M3-route']]
        pairs.append(dict(seed=seed,regular_success_change=b['summary']['success_count']-a['summary']['success_count']))
        for p in ['regular','controlled']:
            ar,br=[rows[f'{arm}/{seed}/{p}'] for arm in ['M3-zero','M3-route']]
            swaps[f'{seed}/{p}']=dict(lost_success=[b['seed'] for a,b in zip(ar,br) if a['success'] and not b['success']],gained_success=[b['seed'] for a,b in zip(ar,br) if not a['success'] and b['success']],new_design_failures=[b['seed'] for a,b in zip(ar,br) if a['design_joint_passed'] and not b['design_joint_passed']])
    mean_gain=sum(p['regular_success_change'] for p in pairs)/(3*96)
    count=lambda arm,p,field:sum(stats[f'{arm}/{s}/{p}'][field] for s in proposal['training_seeds'])
    succeeded=lambda arm,p:sum(stats[f'{arm}/{s}/{p}']['summary']['success_count'] for s in proposal['training_seeds'])
    information=(mean_gain>=proposal['information_gate']['mean_regular_success_gain'] and sum(p['regular_success_change']>0 for p in pairs)>=2
        and succeeded('M3-route','controlled')>=succeeded('M3-zero','controlled')
        and all(count('M3-route',p,k)>=count('M3-zero',p,k) for p in ['regular','controlled'] for k in ['physical','design']))
    candidate_gates={}
    for arm in proposal['formal_expansion_gate']['candidates']:
        preserved={};complete=True;yaw=True
        for s in proposal['training_seeds']:
            for p in ['regular','controlled']:
                r=rows[f'{arm}/{s}/{p}'];complete=complete and stats[f'{arm}/{s}/{p}']['summary']['complete']
                for label in proposal['classical']:
                    b=rows[label+'/'+p]
                    preserved[f'{s}/{p}/{label}']=all((not x['success'] or y['success']) and (not x['physical_safety_passed'] or y['physical_safety_passed']) and (not x['design_joint_passed'] or y['design_joint_passed']) for x,y in zip(b,r))
            j=stats[f'{arm}/{s}/regular']['summary']['mean_yaw_score_deg'];ref=stats['B1/regular']['summary']['mean_yaw_score_deg']
            yaw=yaw and j is not None and ref is not None and j<=ref+.05
        best=max(stats[label+'/controlled']['summary']['success_count'] for label in proposal['classical'])
        gain=(succeeded(arm,'controlled')/3-best)/40
        candidate_gates[arm]=dict(preserved_all_classical_passes=all(preserved.values()),preservation_details=preserved,complete=complete,regular_yaw_nondegradation=yaw,controlled_success_gain=gain,gate=all(preserved.values()) and complete and yaw and gain>=.05)
    aggregate={arm:{p:dict(success=succeeded(arm,p),total=3*len(proposal[p]),physical=count(arm,p,'physical'),design=count(arm,p,'design')) for p in ['regular','controlled']} for arm in proposal['arms']}
    assert checkpoints==120 and len(rows)==30 and sum(len(v) for v in rows.values())==proposal['evaluation_budget_episodes']==2040
    report=dict(verified=True,training_steps=2400000,training_episodes=training_episodes,checkpoints_loaded=120,evaluation_episodes=2040,
        paired_information_effect=pairs,mean_regular_success_gain=mean_gain,information_gate=information,formal_expansion=candidate_gates,aggregate=aggregate,
        failure_patterns=patterns,case_swaps=swaps,artifact_sha256=hashes,proposal_sha256=sha(out/'proposal.json'),trainer_contract_sha256=sha(out/'trainer_contract.json'),reviewer_sha256=sha(__file__),
        scope='Independent source/counters/120 saved CUDA actors-values-Adam/RMS, initial pairing, curriculum and every development task row/gate. Training episode final-height success not reconstructed: final FK is not logged there.3 training seed clusters; repeated136 cases are not408 independent environments. No new physics/training or old final use.')
    (out/'study_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'development_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)
    disposition=dict(five_seed_training_admitted=any(g['gate'] for g in candidate_gates.values()),information_gate=information,
        decision='Expand only the first registered passing candidate' if any(g['gate'] for g in candidate_gates.values()) else 'Stop this pilot expansion; no budget/seed/checkpoint selection or five-seed training.',
        next='Round120 direction/method/extra-experiment review and confirmed-redundancy cleanup; distinguish observer initialization, hidden controller/task history and state-design violations before any new learning.',
        study_review_sha256=sha(out/'study_review.json'),old_models_and_failures_preserved=True)
    (out/'disposition.json').write_text(json.dumps(disposition,indent=2)+'\n')
    print(json.dumps(dict(verified=True,information_gain_pp=mean_gain*100,information_gate=information,formal={a:g['gate'] for a,g in candidate_gates.items()},aggregate=aggregate)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
