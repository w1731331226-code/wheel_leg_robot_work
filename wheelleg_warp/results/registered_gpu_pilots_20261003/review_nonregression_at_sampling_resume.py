"""Fixed-development nonregression audit; not the one-use research gate."""
from pathlib import Path
import argparse,sys,json,pickle
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from training_contract import digest,verify_checkpoint
from score_yaw_gate import at_least
HERE=Path(__file__).resolve().parent;P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3';RUN=P/'runs/M3/1609'
parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=1000000);parser.add_argument('--method',default='M3');parser.add_argument('--seed',type=int,default=1609);parser.add_argument('--interruption-audit',type=Path);args=parser.parse_args();RUN=P/'runs'/args.method/str(args.seed);stem=f'{args.method}_{args.seed}';destination=HERE/f'{stem}_nonregression_{args.steps}.json'
assert not destination.exists(),'Frozen report exists'
p=json.loads((P/'protocol.json').read_text());assert p['environments']==100 and p['evaluation_interval']==20000 and p['ppo']['n_steps']==50 and p['ppo']['batch_size']==250 and p['ppo']['n_epochs']==10
assert 0<args.steps<=p['policy_steps_per_seed'] and args.steps%p['evaluation_interval']==0
review=json.loads((HERE/f'{stem}_stage_{args.steps}.json').read_text());best=json.loads((RUN/f"step_{review['best_policy_steps']}.json").read_text())
b0=json.loads((P/'classical_selection/B0.json').read_text());b1=json.loads((P/'classical_selection.json').read_text())['selected']
assert [r['seed'] for r in best['runs']]==[r['seed'] for r in b0['runs']]==[r['seed'] for r in b1['runs']]
comparisons=[]
for r,base,strong in zip(best['runs'],b0['runs'],b1['runs']):
    assert r['scenario']==base['scenario']==strong['scenario']
    comparisons.append(dict(seed=r['seed'],lost_B0_success=base['success'] and not r['success'],
        speed_nondegradation_pass=at_least(base['velocity_rmse']*1.05+.005,r['velocity_rmse']),
        arrival_nondegradation_pass=r['arrival_s'] is not None and base['arrival_s'] is not None and at_least(base['arrival_s']*1.05+.05,r['arrival_s']),
        candidate_success=r['success'],candidate_yaw_rms=r['rms_deg'][2],B0_yaw_rms=base['rms_deg'][2],B1_yaw_rms=strong['rms_deg'][2]))
end=json.loads((RUN/f'step_{args.steps}.json').read_text());verify_checkpoint(end['path'],end);a=PPO.load(end['path']+'.zip',device='cpu')
expected_epochs=(args.steps//5000)*10;expected_adam=(args.steps//5000)*200;interruption=None
if args.interruption_audit:
    interruption=json.loads(args.interruption_audit.read_text())
    assert interruption['method']==args.method and interruption['seed']==args.seed
    assert interruption['protocol_sha256']==digest(P/'protocol.json')
    saved=interruption['checkpoint_snapshot'];assert saved['resumable']
    assert Path(saved['path']).resolve().parent==RUN.resolve();verify_checkpoint(saved['path'],saved)
    start=interruption['consumed_policy_steps'];assert 0<start<args.steps
    resumed=PPO.load(saved['path']+'.zip',device='cpu')
    assert resumed.num_timesteps==start and resumed._n_updates==interruption['training_epochs']
    assert {int(state['step'].item()) for state in resumed.policy.optimizer.state.values()}=={interruption['Adam_updates']}
    assert start-interruption['latest_scored_steps']==interruption['partial_rollout_discarded_charged_steps']
    expected_epochs=interruption['training_epochs'];expected_adam=interruption['Adam_updates']
    # Match the frozen learn_exact scheduler: finish each evaluation interval
    # with a shorter rollout; discarded pre-interruption samples remain charged.
    for target in range((start//20000+1)*20000,args.steps+1,20000):
        while start<target:
            chunk=min(5000,target-start);assert chunk%100==0
            expected_epochs+=10;expected_adam+=10*((chunk+249)//250);start+=chunk
    assert start==args.steps
assert a.num_timesteps==args.steps and a._n_updates==expected_epochs
assert all(torch.isfinite(v).all() for v in a.policy.state_dict().values())
assert all(torch.isfinite(v).all() for state in a.policy.optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor))
assert {int(state['step'].item()) for state in a.policy.optimizer.state.values()}=={expected_adam}
with open(end['path']+'.pkl','rb') as f:normalizer=pickle.load(f)
assert np.isfinite(normalizer.obs_rms.mean).all() and np.isfinite(normalizer.obs_rms.var).all() and np.all(normalizer.obs_rms.var>=0)
result=dict(passed=True,scope='Read-only development32case diagnostic; full3seed paired research gate not evaluated',stage_steps=args.steps,
 best_steps=best['policy_steps'],best_summary=best['summary'],B0_summary=b0['summary'],B1_summary=b1['summary'],
 lost_B0_successes=sum(x['lost_B0_success'] for x in comparisons),speed_passes=sum(x['speed_nondegradation_pass'] for x in comparisons),arrival_passes=sum(x['arrival_nondegradation_pass'] for x in comparisons),
 comparisons=comparisons,finite_policy_adam_rms=True,training_epochs=a._n_updates,Adam_minibatch_updates=expected_adam,segmented_resume=interruption is not None,interruption_audit=str(args.interruption_audit) if args.interruption_audit else None,interruption_audit_sha256=digest(args.interruption_audit) if args.interruption_audit else None,rms_count=float(normalizer.obs_rms.count),
 research_claim_passed=False,gate_or_final_simulated=False,protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__))
destination.write_text(json.dumps(result,indent=2)+'\n')
print('AUDIT',args.steps,'best',best['policy_steps'],'lostB0',result['lost_B0_successes'],'speed',result['speed_passes'],'arrival',result['arrival_passes'],'/32; exact budget-derived epochs/Adam updates finite')
