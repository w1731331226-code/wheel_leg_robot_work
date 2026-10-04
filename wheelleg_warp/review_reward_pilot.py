"""Read-only completed-run/pair audit; no replays, selection or training."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from train_height_comparison import summary
from training_contract import digest
from dashboard.live_env import atomic_json

S=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reward_pilot_v1'

def run():
    proposal=json.loads((S/'proposal.json').read_text());contract=json.loads((S/'trainer_contract.json').read_text())
    assert contract['proposal_sha256']==digest(S/'proposal.json')
    assert all(digest(ROOT/name)==value for name,value in contract['source_sha256'].items())
    completed={};pending=[];pairs={};artifacts={};consumed=0
    for seed in proposal['training_seeds']:
        for arm in proposal['arms']:
            p=S/'runs'/arm/str(seed);key=f'{arm}/{seed}'
            progress=json.loads((p/'progress.json').read_text()) if (p/'progress.json').is_file() else {}
            consumed+=progress.get('sampled_steps',0)
            if progress.get('status')!='complete':pending.append(dict(run=key,progress=progress));continue
            report=json.loads((p/'verification.json').read_text());data=json.loads((p/'development_final.json').read_text())
            init=json.loads((p/'initialization.json').read_text())
            assert report['verified'] and report['policy_steps']==progress['sampled_steps']==progress['trained_steps']==200000
            assert report['ppo_epochs']==400 and report['forced_physical_resets_midrun']==0 and not report['partial_episode_discarded_midrun']
            assert report['checkpoints']==list(range(20000,200001,20000))
            for name,value in report['source_sha256'].items():assert value==contract['source_sha256'][name]==digest(ROOT/name)
            for steps in report['checkpoints']:
                r=json.loads((p/f'step_{steps}.json').read_text())
                assert r['trained_steps']==r['policy_steps']==steps and r['ppo_epochs']==steps//5000*10 and r['adam_updates']==steps//5000*200
                assert r['adam']['moment_devices']==['cuda'] and r['physical_Phi_and_RNG_unchanged_by_save']
                b=r['last_rollout_bootstrap'];assert b['policy_steps']==steps and b['pre_update_epoch_counter']==r['ppo_epochs']-10
                assert len(b['values'])==len(b['dones'])==100 and np.isfinite(b['values']).all()
                phi=np.asarray(r['ongoing_phi']);length=np.asarray(r['ongoing_episode_policy_steps'])
                expected=.99**length*phi
                np.testing.assert_allclose(expected,r['discounted_nonterminal_phi_boundary'],atol=1e-12,rtol=0)
                assert np.all(abs(np.asarray(r['potential_discount_errors'])-expected)<=np.asarray(r['rounding_bounds'])+1e-10)
                assert digest(p/f'step_{steps}.zip')==r['checkpoint']['checkpoint_sha256']
                assert digest(p/f'step_{steps}.pkl')==r['checkpoint']['normalization_sha256']
            rows=data['runs'];assert len(rows)==96
            assert [r['seed'] for r in rows]==[r['seed'] for r in proposal['development']]
            assert all(r['scenario']==s['scenario'] for r,s in zip(rows,proposal['development']))
            assert all(r['physical_steps']==r['physical_evidence_steps']>0 for r in rows)
            assert summary(rows)==data['summary']
            agent=PPO.load(str(p/'step_200000.zip'),device='cpu')
            assert agent.num_timesteps==200000 and agent._n_updates==400
            assert all(torch.isfinite(v).all() for v in agent.policy.state_dict().values())
            states=agent.policy.optimizer.state_dict()['state']
            assert {int(v['step'].item()) for v in states.values()}=={8000}
            assert all(torch.isfinite(v).all() for s in states.values() for v in s.values() if isinstance(v,torch.Tensor))
            completed[key]=dict(summary=data['summary'],physical=sum(r['physical_safety_passed'] for r in rows),
                design=sum(r['design_joint_passed'] for r in rows),initial=init,completed_training_episodes=report['completed_episodes'],
                shaping_transitions=report['nonzero_shaping_transitions'])
            for name in ['initialization.json','verification.json','development_final.json','step_200000.json','step_200000.zip','step_200000.pkl']:
                artifacts[str((p/name).relative_to(S))]=digest(p/name)
        left=completed.get(f'original/{seed}');right=completed.get(f'potential/{seed}')
        if left and right:
            assert left['initial']['initial_weight_sha256']==right['initial']['initial_weight_sha256']
            assert left['initial']['initial_world_sha256']==right['initial']['initial_world_sha256']
            pairs[str(seed)]=dict(success_difference=right['summary']['success_count']-left['summary']['success_count'],
                success_fraction_difference=(right['summary']['success_count']-left['summary']['success_count'])/96,
                yaw_score_difference_deg=right['summary']['mean_yaw_score_deg']-left['summary']['mean_yaw_score_deg'],
                physical_failure_difference=left['physical']-right['physical'],design_failure_difference=left['design']-right['design'])
    assert consumed<=proposal['total_maximum_pilot_policy_steps']
    gate=None
    if len(pairs)==3:
        delta=np.mean([p['success_fraction_difference'] for p in pairs.values()]);yaw=np.mean([p['yaw_score_difference_deg'] for p in pairs.values()])
        gate=bool(delta>=.05 and sum(p['success_difference']>0 for p in pairs.values())>=2 and yaw<=.05 and
            np.mean([p['physical_failure_difference'] for p in pairs.values()])<=0 and np.mean([p['design_failure_difference'] for p in pairs.values()])<=0)
    result=dict(verified=True,completed_runs=len(completed),completed_seed_pairs=len(pairs),observed_sampled_budget=consumed,
        completed=completed,pending=pending,pairs=pairs,continuation_gate=gate,
        scope='Fixed final200k public development, paired3seeds required; not independent gate, significance claim or advantage over strong classic. CPU reload only audits saved models, actual learningCUDA checked in producer.',
        artifact_sha256=artifacts,source_sha256=digest(Path(__file__)))
    atomic_json(S/'pair_review.json',result)
    print('AUDITED',len(completed),'runs',len(pairs),'pairs; budget',consumed,'gate',gate,flush=True)
    for seed,pair in pairs.items():print('PAIR',seed,pair,flush=True)

if __name__=='__main__':run()
