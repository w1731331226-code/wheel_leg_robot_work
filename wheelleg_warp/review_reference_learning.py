"""Read only closed learners; never restart, select or tune active main jobs."""
import json,pickle,math
import numpy as np
from stable_baselines3 import PPO
from review_yaw_sector import ROOT,sha,success
from reference_budget_train import OUT
from dashboard.live_env import atomic_json


def run():
    p=json.loads((OUT/'proposal.json').read_text());contract=json.loads((OUT/'trainer_contract.json').read_text());progress=json.loads((OUT/'main_progress.json').read_text())
    assert all(sha(ROOT/n)==v for n,v in contract['source_sha256'].items()) and contract['proposal_sha256']==sha(OUT/'proposal.json')
    completed=list(progress['completed_runs']);records=[];checkpoint_count=0
    for job in completed:
        arm=job['arm'];seed=job['seed'];d=OUT/'runs'/arm/str(seed);v=json.loads((d/'verification.json').read_text());init=json.loads((d/'initialization.json').read_text())
        assert v['verified'] and v['policy_steps']==200000 and v['ppo_epochs']==400 and v['adam_updates']==8000 and not init['engineering_weights_reused']
        assert v['checkpoints']==list(range(20000,200001,20000)) and v['main_source_contract_sha256']==sha(OUT/'trainer_contract.json')
        checkpoints=[]
        for step in v['checkpoints']:
            prefix=d/f'step_{step}';meta=json.loads(prefix.with_suffix('.json').read_text());assert sha(prefix.with_suffix('.zip'))==meta['checkpoint']['checkpoint_sha256'] and sha(prefix.with_suffix('.pkl'))==meta['checkpoint']['normalization_sha256']
            model=PPO.load(str(prefix)+'.zip',device='cpu');assert model.num_timesteps==step and model._n_updates==step//5000*10
            assert all(np.isfinite(x.detach().numpy()).all() for x in model.policy.parameters())
            states=model.policy.optimizer.state_dict()['state'];assert states and {int(s['step'].item()) for s in states.values()}=={step//5000*200}
            assert meta['adam']['moment_devices']==['cuda'] and meta['physical_Phi_and_RNG_unchanged_by_save'] and meta['route_memory_unchanged_by_save']
            with open(str(prefix)+'.pkl','rb') as f:norm=pickle.load(f)
            assert norm.obs_rms.mean.shape==(39,) and np.isfinite(norm.obs_rms.mean).all() and np.isfinite(norm.obs_rms.var).all() and np.all(norm.obs_rms.var>=0)
            checkpoints.append(dict(step=step,checkpoint_sha256=meta['checkpoint']['checkpoint_sha256'],normalization_sha256=meta['checkpoint']['normalization_sha256']));checkpoint_count+=1
        episodes=json.loads((d/'episodes.json').read_text());assert len(episodes['episodes'])==v['completed_training_episodes'] and all(t['actual_episode_end'] for t in episodes['curriculum_transitions'])
        for row in episodes['episodes']:
            stats=np.array(row['allocation_stats']);assert np.isfinite(stats).all() and stats[6]==stats[8]==0
        panels={}
        for panel in ['regular','controlled']:
            path=d/(panel+'_final.json');e=json.loads(path.read_text());assert len(e['runs'])==len(p[panel]);scores=[]
            for row,case in zip(e['runs'],p[panel]):
                assert row['seed']==case['seed'] and row['scenario']==case['scenario'] and row['success']==success(row)
                physical=min(row['min_actual_A_leg_m'],row['min_actual_B_leg_m'])>=row['geometric_limit_m'] and row['min_eight_joint_margin_rad']>=0 and max(row['max_actual_torque_excess_Nm'],row['max_command_torque_excess_Nm'])<=1e-6
                assert row['physical_safety_passed']==physical and row['design_joint_passed']==(row['min_active_design_margin_rad']>=0)
                assert row['physical_evidence_steps']==row['physical_steps']>0
                scores.append(row['rms_deg'][2]*math.sqrt(row['duration_s']/(3.5+1.5*row['task_goal_progress_m']/abs(row['scenario']['speed']))) if row['reason']=='completed' else None)
            assert e['summary']['success_count']==sum(x['success'] for x in e['runs']) and e['physical']==sum(x['physical_safety_passed'] for x in e['runs']) and e['design']==sum(x['design_joint_passed'] for x in e['runs'])
            assert e['summary']['complete']==all(x is not None for x in scores)
            if all(x is not None for x in scores):assert math.isclose(e['summary']['mean_yaw_score_deg'],sum(scores)/len(scores),abs_tol=1e-12)
            else:assert e['summary']['mean_yaw_score_deg'] is None
            panels[panel]=dict(sha256=sha(path),summary=e['summary'],physical=e['physical'],design=e['design'])
        records.append(dict(arm=arm,seed=seed,verified=True,policy_steps=200000,completed_training_episodes=v['completed_training_episodes'],verification_sha256=sha(d/'verification.json'),episodes_sha256=sha(d/'episodes.json'),checkpoints=checkpoints,panels=panels))
        print('VERIFIED CLOSED',arm,seed,flush=True)
    atomic_json(OUT/'closed_runs_review.json',dict(verified=True,completed_runs_snapshot=completed,records=records,trained_closed_policy_steps=len(records)*200000,checkpoint_loads_verified=checkpoint_count,
        snapshot_progress_status=progress['status'],trainer_contract_sha256=sha(OUT/'trainer_contract.json'),reviewer_sha256=sha(__file__),
        scope='Only closed runs inspected onCPU savedfiles. Active job untouched; incomplete arms/seeds cannot establish mechanism/advantage. No reruns/hyperparameter or primary changes.'))
    print('PASS closed-run snapshot',len(records),'checkpoints',checkpoint_count,flush=True)


if __name__=='__main__':run()
