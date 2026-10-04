"""Independent engineering/checkpoint/classical admission audit before2.4M pilot."""
from pathlib import Path
import json,sys,pickle,math
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from stable_baselines3 import PPO
from review_yaw_sector import sha,success
from smoke_reward_training import check_agent,weight_digest

OUT=Path(__file__).resolve().parents[1]
d=json.loads((OUT/'proposal.json').read_text());e=json.loads((OUT/'engineering_verification.json').read_text())
f=json.loads((OUT/'classical_feasibility.json').read_text());scale=json.loads((OUT/'initial_effort_audit.json').read_text())
assert d['version']=='route-action-pilot-v1.1' and d['total_training_budget']==2400000 and not d['old_gate_or_final_used']
assert e['verified'] and e['total_training_steps']==4000 and e['source_sha256']==d['source_sha256']
assert all(sha(ROOT/n)==v for n,v in d['source_sha256'].items())
assert f['proposal_sha256']==sha(OUT/'proposal.json')
assert scale['state_bank_sha256']==sha(ROOT/'wheelleg_warp/results/twentyninth_round_capped_support_20261003/state_bank.npz')
assert scale['reference_states']==888 and scale['gaussian_samples']==256
loaded={};initial={}
for arm,a in d['arms'].items():
    folder=OUT/'engineering'/arm/'1609';r=json.loads((folder/'verification.json').read_text());init=json.loads((folder/'initialization.json').read_text())
    assert r==e['records'][arm] and r['verified'] and r['engineering_only'] and not r['weights_promoted']
    assert r['policy_steps']==1000 and r['ppo_epochs']==20 and r['adam_updates']==40
    assert r['dimensions']==a['dimensions'] and r['observations']==39 and r['forced_physical_or_route_resets_midrun']==0
    assert r['source_sha256']==d['source_sha256'] and r['checkpoints']==[500,1000]
    initial[arm]=init
    for step in [500,1000]:
        prefix=folder/f'step_{step}';record=json.loads(prefix.with_suffix('.json').read_text())
        assert record['policy_steps']==record['trained_steps']==step and record['ppo_epochs']==step//500*10
        assert record['adam_updates']==step//500*20 and record['route_memory_unchanged_by_save'] and record['physical_Phi_and_RNG_unchanged_by_save']
        assert sha(prefix.with_suffix('.zip'))==record['checkpoint']['checkpoint_sha256'] and sha(prefix.with_suffix('.pkl'))==record['checkpoint']['normalization_sha256']
        agent=PPO.load(str(prefix)+'.zip',device='cuda');check_agent(agent,step//500*20)
        assert agent.num_timesteps==step and agent._n_updates==step//500*10 and agent.observation_space.shape==(39,) and agent.action_space.shape==(a['dimensions'],)
        assert weight_digest(agent)!=init['weight_sha256']
        with open(str(prefix)+'.pkl','rb') as stream:norm=pickle.load(stream)
        assert norm.obs_rms.mean.shape==(39,) and np.isfinite(norm.obs_rms.mean).all() and np.isfinite(norm.obs_rms.var).all()
    loaded[arm]=dict(policy_steps=1000,epochs=20,adam=40,dimensions=a['dimensions'])
assert initial['M3-zero']['weight_sha256']==initial['M3-route']['weight_sha256'] and initial['M3-zero']['world_sha256']==initial['M3-route']['world_sha256']
assert scale['arms']['M3-zero']['statistics']==scale['arms']['M3-route']['statistics']
assert scale['arms']['L2-route']['statistics']['40']['actual_motor_rms_normalized'][4:]==[0.,0.]
episodes=0;stats={}
for label in d['classical']:
    for panel in ['regular','controlled']:
        path=OUT/'classical'/(label+'_'+panel+'.json');r=json.loads(path.read_text());rows=r['runs'];scores=[]
        assert len(rows)==len(d[panel])
        for row,case in zip(rows,d[panel]):
            assert row['seed']==case['seed'] and row['scenario']==case['scenario'] and row['success']==success(row)
            assert row['physical_evidence_steps']==row['physical_steps']>0
            if row['reason']=='completed':
                duration=row['physical_steps']*.0005;assert math.isclose(duration,row['duration_s'],abs_tol=1e-8) and duration>=row['arrival_s']+2-1e-8
                scores.append(row['rms_deg'][2]*math.sqrt(duration/(3.5+1.5*row['task_goal_progress_m']/abs(row['scenario']['speed']))))
            else:scores.append(None)
        assert r['summary']['total']==len(rows) and r['summary']['success_count']==sum(x['success'] for x in rows)
        assert r['physical']==sum(x['physical_safety_passed'] for x in rows) and r['design']==sum(x['design_joint_passed'] for x in rows)
        complete=all(s is not None for s in scores);assert r['summary']['complete']==complete
        if complete:assert math.isclose(sum(scores)/len(scores),r['summary']['mean_yaw_score_deg'],abs_tol=1e-12)
        else:assert r['summary']['mean_yaw_score_deg'] is None
        stats[label+'_'+panel]={k:v for k,v in r.items() if k!='runs'}
        assert stats[label+'_'+panel]==f['records'][label+'_'+panel];episodes+=len(rows)
best=max(stats[label+'_controlled']['summary']['success_count'] for label in d['classical'])
assert episodes==408 and f['formal_gate_numerically_attainable']==(40-best>=2 and all(stats['B1_'+p]['summary']['complete'] for p in ['regular','controlled']))
assert math.isclose(f['controlled_maximum_gain_pp'],100*(40-best)/40)
assert f['formal_gate_numerically_attainable']
report=dict(verified=True,engineering_training_steps=4000,classical_episodes=episodes,loaded=loaded,controlled_success_headroom_pp=f['controlled_maximum_gain_pp'],
    proposal_sha256=sha(OUT/'proposal.json'),engineering_sha256=sha(OUT/'engineering_verification.json'),classical_sha256=sha(OUT/'classical_feasibility.json'),
    reviewer_sha256=sha(__file__),scope='Source, actual dimension/checkpoints/CUDA Adam/normalization, initial pairing, static effort accounting and classical task/headroom audited. No claim that learned methods will pass.')
(OUT/'admission_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report))
