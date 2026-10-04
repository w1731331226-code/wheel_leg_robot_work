"""Read-only reloading check of route-engineering source and checkpoint evidence."""
from pathlib import Path
import json,hashlib,sys,pickle
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from stable_baselines3 import PPO
from smoke_reward_training import weight_digest,check_agent
import numpy as np

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
OUT=Path(__file__).resolve().parents[1]
interface=ROOT/'wheelleg_warp/results/paper_recovery_20261004/route_interface_v1'
ireg=json.loads((interface/'registration.json').read_text());iv=json.loads((interface/'verification.json').read_text())
assert iv['verified'] and iv['training_steps']==0 and ireg['maximum_actor_steps_per_mode']==2500
assert ireg['curriculum_milestones']==[20,10000] and not ireg['old_gate_or_final_used']
assert iv['source_sha256']==ireg['source_sha256'] and all(sha(ROOT/n)==v for n,v in iv['source_sha256'].items())
for mode,r in iv['reports'].items():
    assert mode in ['zero','route'] and r['actor_steps']<=2500 and min(r['per_world_episodes'])>=2
    assert sum(r['per_world_episodes'])==r['completed_episodes']==32 and r['actual_curriculum_transitions']==20 and r['stage3_worlds']==10
    assert r['physical_control_buffers_unchanged_by_wrapper'] and r['base_packet_reward_done_and_terminal_exact']
reg=json.loads((OUT/'registration.json').read_text());v=json.loads((OUT/'verification.json').read_text())
assert reg['interface_verification_sha256']==sha(interface/'verification.json') and v['verified'] and v['total_training_steps']==2000
assert reg['environments']==10 and reg['policy_steps_per_arm']==1000 and reg['updates_per_arm']==2 and not reg['old_gate_or_final_used']
assert v['source_sha256']==reg['source_sha256'] and all(sha(ROOT/n)==h for n,h in v['source_sha256'].items())
initial=set();loaded={}
for mode in ['zero','route']:
    r=json.loads((OUT/(mode+'.json')).read_text());assert r==v['reports'][mode]
    prefix=OUT/mode
    assert sha(prefix.with_suffix('.zip'))==r['checkpoint']['checkpoint_sha256'] and sha(prefix.with_suffix('.pkl'))==r['checkpoint']['normalization_sha256']
    agent=PPO.load(str(prefix)+'.zip',device='cuda');adam=check_agent(agent,40)
    assert agent.observation_space.shape==(39,) and agent.num_timesteps==1000 and agent._n_updates==20 and not r['checkpoint_promoted']
    assert weight_digest(agent)==r['final_weight_sha256']!=r['initial_weight_sha256'];initial.add(r['initial_weight_sha256'])
    with open(str(prefix)+'.pkl','rb') as stream:norm=pickle.load(stream)
    assert norm.obs_rms.mean.shape==(39,) and np.isfinite(norm.obs_rms.var).all()
    assert norm.obs_rms.mean[38]==r['route_mean'] and norm.obs_rms.var[38]==r['route_variance']
    assert all(x>0 for x in r['route_input_weight_max_deltas']) if mode=='route' else r['route_input_weight_max_deltas']==[0.,0.]
    assert all(np.isfinite(x) for x in r['losses'].values())
    loaded[mode]=dict(policy_steps=agent.num_timesteps,ppo_epochs=agent._n_updates,adam=adam,checkpoint_sha256=sha(prefix.with_suffix('.zip')))
assert len(initial)==1
result=dict(verified=True,interface_verification_sha256=sha(interface/'verification.json'),training_verification_sha256=sha(OUT/'verification.json'),
    same_initial_weight_sha256=initial.pop(),loaded_checkpoints=loaded,source_sha256=sha(__file__),
    scope='Independent source/hash/ledger and saved CUDA weights/Adam/counters/RMS verification; runtime exact-restoration and physical identity assertions reviewed in frozen producer. No new physics, method-performance gate or trajectory resume.')
(OUT/'artifact_review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result))
