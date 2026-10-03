"""Saved training episode windows; no sampling, evaluation or policy changes."""
from pathlib import Path
import json,sys
import numpy as np
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from training_contract import digest,verify_checkpoint

HERE=Path(__file__).resolve().parent
P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
destination=HERE/'round65_episode_log_audit.json'
assert not destination.exists(),'Frozen audit exists'
protocol=json.loads((P/'protocol.json').read_text());records=[]
for method in ('M3','B2-V'):
    for steps in (20000,200000,500000):
        row=json.loads((P/'runs'/method/'1610'/f'step_{steps}.json').read_text())
        verify_checkpoint(row['path'],row)
        assert [x['scenario'] for x in row['runs']]==[x['scenario'] for x in protocol['selection']]
        agent=PPO.load(row['path']+'.zip',device='cpu')
        assert agent.num_timesteps==steps and agent._n_updates==steps//5000*10
        window=list(agent.ep_info_buffer or [])
        assert agent.ep_info_buffer.maxlen==100 and len(window)<=100
        assert all(set(x)=={'r','l'} and np.isfinite(x['r']) and x['l']>0 for x in window)
        values=[x['r'] for x in window]
        records.append(dict(method=method,seed=1610,policy_steps=steps,
            recorded_recent_training_episodes=len(window),window_capacity=100,
            mean_episode_return=float(np.mean(values)) if values else None,
            episode_return_range=[min(values),max(values)] if values else None,
            mean_episode_policy_length=float(np.mean([x['l'] for x in window])) if window else None,
            episode_records_have_scene_or_success_fields=False,
            development_summary=row['summary'],checkpoint_sha256=row['checkpoint_sha256'],
            normalization_sha256=row['normalization_sha256']))
assert all(r['recorded_recent_training_episodes']==0 for r in records if r['policy_steps']==20000)
assert all(r['recorded_recent_training_episodes']==100 for r in records if r['policy_steps']>20000)
result=dict(round=65,passed=True,records=records,
    inference='Recent asynchronous training episode windows, not IID/full training success rates. Empty20k logs are not evidence of learning complete tasks. Changing training return and development performance do not identify overfitting or reward causality.',
    log_source='NativeEnv info episode r=state20 accumulator,l=ceil(physical_steps/40); SB3 BaseAlgorithm updates ep_info_buffer from info episode',
    extra_task_evaluations=0,gate_or_final_simulated=False,
    protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__))
destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for row in records:
    print(row['method'],row['policy_steps'],'episodes',row['recorded_recent_training_episodes'],
          'mean_return',row['mean_episode_return'],'development',row['development_summary'])
print('PASS saved window semantics and counts; no new task evaluations')
