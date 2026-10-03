"""Read-only failure/normalization/action audit on fixed development evidence."""
from pathlib import Path
from collections import Counter,defaultdict
import json,pickle,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from training_contract import digest,verify_checkpoint
HERE=Path(__file__).resolve().parent;P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3';RUN=P/'runs/M3/1609'

def failures(r):
    s=r['scenario'];out=[]
    if r['reason']!='completed':out.append('incomplete')
    if not r['physical_safety_passed']:out.append('physical')
    if not r['design_joint_passed']:out.append('design1p4')
    attitude=r['relative_peak_deg'] if s.get('relative_attitude') else r['peak_deg']
    if max(attitude)>5 or r['peak_deg'][2]>5 or (s.get('relative_attitude') and max(r['peak_deg'][:2])>10):out.append('attitude')
    if r['velocity_rmse']>.2*abs(s['speed']):out.append('speed')
    if r['stop_distance_m']>.6:out.append('stop_distance')
    if r['tail_speed_m_s']>.03:out.append('tail_speed')
    if r['height_rmse_m']>.02 or abs(r['final_mean_fk_leg_m']-r['target_leg_m'])>.02:out.append('height')
    if not r['terrain_evidence_passed']:out.append('terrain')
    if r['touched_contact_mask']&r['required_contact_mask']!=r['required_contact_mask']:out.append('obstacle_contact')
    return out

bank=ROOT/'wheelleg_warp/results/twentyninth_round_capped_support_20261003/state_bank.npz'
with np.load(bank,allow_pickle=False) as saved:obs=saved['obs'].copy()
assert len(obs)==888 and np.all(obs[:,32:]==0)
reports=[]
for steps in (20000,200000,500000):
    record=json.loads((RUN/f'step_{steps}.json').read_text());verify_checkpoint(record['path'],record)
    agent=PPO.load(record['path']+'.zip',device='cpu')
    with open(record['path']+'.pkl','rb') as f:normalizer=pickle.load(f)
    assert agent.num_timesteps==steps and all(torch.isfinite(v).all() for v in agent.policy.state_dict().values())
    assert all(torch.isfinite(v).all() for state in agent.policy.optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor))
    assert np.isfinite(normalizer.obs_rms.mean).all() and np.isfinite(normalizer.obs_rms.var).all() and np.all(normalizer.obs_rms.var>=0)
    z=normalizer.normalize_obs(obs)
    with torch.no_grad():mu=agent.policy.get_distribution(torch.as_tensor(z)).distribution.mean.numpy()
    by_terrain=defaultdict(lambda:[0,0]);bad=Counter();instances=[]
    for r in record['runs']:
        f=failures(r);assert r['success'] or f,'Failed row has no independent failure reason'
        assert not r['success'] or not f,'Success contradicts independent gates'
        by_terrain[r['scenario']['terrain']][0]+=int(r['success']);by_terrain[r['scenario']['terrain']][1]+=1
        if not r['success']:bad.update(f);instances.append(dict(seed=r['seed'],height=r['target_leg_m'],terrain=r['scenario']['terrain'],failures=f))
    reports.append(dict(policy_steps=steps,summary=record['summary'],failure_counts_nonexclusive=dict(bad),terrain_success_total=dict(by_terrain),failed_cases=instances,
        finite_policy_optimizer_rms=True,training_epochs=agent._n_updates,optimizer_steps=sorted({int(v['step'].item()) for v in agent.policy.optimizer.state.values()}),
        learned_std=agent.policy.log_std.exp().detach().numpy().tolist(),fixed_zero_request_bank_mean_absolute_policy_output=np.mean(abs(mu),axis=0).tolist(),
        fixed_zero_request_bank_action_clip_fraction=float(np.mean(abs(mu)>1)),normalizer_observation_count=float(normalizer.obs_rms.count),
        diagnostic_scope='Frozen888zero-Actor states, current checkpoint normalization; action audit only, not task replay or causal proof'))
classical=json.loads((P/'classical_selection.json').read_text())
result=dict(passed=True,protocol_sha256=digest(P/'protocol.json'),bank_sha256=digest(bank),method='M3',seed=1609,reports=reports,
 selected_b1=classical['selected']['summary'],configuration_changed=False,extra_task_evaluations=0,gate_or_final_simulated=False,
 decision='Continue registered2M and paired reference runs; deterioration is real development evidence, not tensor failure. Investigate reward/partial observability only after frozen comparison, no mid-run tuning.',
 verifier_sha256=digest(__file__))
(HERE/'M3_1609_halfmillion_audit.json').write_text(json.dumps(result,indent=2)+'\n')
for r in reports:print('AUDIT',r['policy_steps'],r['summary'],'failures',r['failure_counts_nonexclusive'],'std',r['learned_std'],'clip',r['fixed_zero_request_bank_action_clip_fraction'],flush=True)
