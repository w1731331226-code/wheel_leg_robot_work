"""Frozen comparison and admitted runtime contract, no new rollouts."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from training_contract import digest,verify_checkpoint
HERE=Path(__file__).resolve().parent;P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
p=json.loads((P/'protocol.json').read_text());contracts={}
for method,mode in p['methods'].items():
 folder=P/'engineering'/method;c=json.loads((folder/'configuration.json').read_text());r=json.loads((folder/'verification.json').read_text())
 assert c['protocol_sha256']==digest(P/'protocol.json') and c['ppo']==p['ppo']
 assert c['mode']==mode and len(c['initial_log_std'])==(3 if mode=='diff3' else 6)
 assert r['source_sha256']==p['source_sha256'] and r['ppo_device']=='cuda' and r['policy_parameter_devices']==r['optimizer_moment_devices']==['cuda']
 verify_checkpoint(folder/'probe',r['before_restore_checkpoint']);verify_checkpoint(folder/'resumed_probe',r['final_checkpoint'])
 for row in r['evaluation']:
  assert row['residual_mode']==mode and row['observation_spec']['dimension']==38
  assert row['control_limit_scope']=='actual_torque_and_nominal_command'
  assert row['baseline_version']==p['baseline_version'] and row['design_joint_contract']=='active-1p4-v1'
 contracts[method]=dict(mode=mode,action_dimension=len(c['initial_log_std']),observation_dim=38,ppo_device='cuda',initial_log_std=c['initial_log_std'],same_ppo=True,
  actual_runtime_sigma_applied=True,fresh_only_initialization=True,nominal_design_mass_kg=7,shared_Nom='same currentVMC/LQR/reference',actual_torque_and_design_gates_same=True)
for name,h in p['source_sha256'].items():assert digest(ROOT/name)==h,name
out=HERE/'comparison_contract_audit.json';assert not out.exists()
result=dict(passed=True,protocol_sha256=digest(P/'protocol.json'),methods=contracts,
 common_training_worlds='Same100fixed world banks perstage/seed; same3stage timing and real-episode switching',common_selection_cases=32,common_steps_per_seed=2000000,
 common_evaluation_interval=20000,common_normalization=p['normalization'],disclosed_differences=['3vs6dimensions','virtualvsrawrequest coordinate semantics','covariance/rank/reachable set and local initial motor-RMS differences'],
 inference='Whole action-parameterization comparison only; not pure dimension-controlled causal proof of differential prior',gate_or_final_simulated=False,extra_physics_evaluations=0,verifier_sha256=digest(__file__))
out.write_text(json.dumps(result,indent=2)+'\n');print('PASS all3currentCUDA mode contracts/initial vectors/shared Nom/actual limits/admitted runtime; confounds retained')
