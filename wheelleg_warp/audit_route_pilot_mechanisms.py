"""Round120 offline diagnosis; no model training, physics or gate revision."""
from pathlib import Path
import json,math,warnings
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from route_pilot_env import agent
from train_height_comparison import protocol
from smoke_reward_training import weight_digest
from review_yaw_sector import sha,ROOT
warnings.filterwarnings('ignore',message='You are trying to run PPO on the GPU.*',category=UserWarning)

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/route_pilot_v1'
BASE=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'


def run():
    # Match run_one's CPU QR initialization arithmetic before comparing hashes.
    torch.set_num_threads(1)
    review=json.loads((OUT/'study_review.json').read_text());assert review['verified'] and not review['information_gate']
    proposal=json.loads((OUT/'proposal.json').read_text());assert all(sha(ROOT/n)==v for n,v in proposal['source_sha256'].items())
    groups={};failures=[]
    for seed in proposal['training_seeds']:
        for arm in proposal['arms']:
            for panel in ['regular','controlled']:
                path=OUT/'runs'/arm/str(seed)/(panel+'_final.json');assert sha(path)==review['artifact_sha256'][str(path.relative_to(OUT))]
                for r in json.loads(path.read_text())['runs']:
                    h=r['scenario']['stand_height_m'];bucket='endpoint115' if abs(h-.115)<1e-9 else 'low' if h<.16 else 'mid' if h<.32 else 'high'
                    key=panel+'/'+bucket;g=groups.setdefault(key,dict(total=0,task_failed=0,design_failed=0,physical_failed=0,yaw_failed=0,terrain_evidence_failed=0))
                    g['total']+=1;g['task_failed']+=int(not r['success']);g['design_failed']+=int(not r['design_joint_passed']);g['physical_failed']+=int(not r['physical_safety_passed'])
                    g['yaw_failed']+=int(r['peak_deg'][2]>5);g['terrain_evidence_failed']+=int(not r['terrain_evidence_passed'])
                    if not r['design_joint_passed']:
                        failures.append(dict(seed=seed,arm=arm,panel=panel,case=r['seed'],height_m=h,
                            working_joint_overshoot_rad=-r['min_active_design_margin_rad'],physical_joint_margin_rad=r['min_eight_joint_margin_rad'],
                            min_actual_leg_m=min(r['min_actual_A_leg_m'],r['min_actual_B_leg_m']),success=r['success'],physical_passed=r['physical_safety_passed']))
    # Reconstruct exact fresh policies from their registered seed/config.
    # No fitted weights or validation-based model selection is involved.
    bankpath=ROOT/'wheelleg_warp/results/twentyninth_round_capped_support_20261003/state_bank.npz'
    with np.load(bankpath,allow_pickle=False) as z:raw=z['obs'].copy()
    normalized=np.c_[np.clip(raw/np.sqrt(1+1e-8),-10,10),np.zeros(len(raw))].astype(np.float32)
    p=protocol(BASE);sensitivity={}
    for seed in proposal['training_seeds']:
        def factory():
            env=gym.Env();env.observation_space=gym.spaces.Box(-np.inf,np.inf,(39,),dtype=np.float32);env.action_space=gym.spaces.Box(-1,1,(3,),dtype=np.float32);return env
        env=DummyVecEnv([factory])
        try:
            model=agent(p,'M3-route',seed,env);initial=weight_digest(model)
            for arm in ['M3-zero','M3-route']:
                logged=json.loads((OUT/'runs'/arm/str(seed)/'initialization.json').read_text());assert initial==logged['weight_sha256']
            def response(y):
                obs=normalized.copy();obs[:,38]=y
                with torch.no_grad():
                    tensor=torch.as_tensor(obs,device=model.device)
                    mean=model.policy.get_distribution(tensor).distribution.mean.cpu().numpy()
                    value=model.policy.predict_values(tensor).cpu().numpy().reshape(-1)
                return np.clip(mean,-1,1),value
            mean0,value0=response(0);before={}
            for y in [-3.,3.]:
                mean,value=response(y);before[str(y)]=dict(max_action_change=float(abs(mean-mean0).max()),
                    coordinate_rms_change=np.sqrt(np.mean((mean-mean0)**2,axis=0)).tolist(),
                    max_wheel_request_change_Nm=float(abs(mean[:,2]-mean0[:,2]).max()*.3),max_value_change=float(abs(value-value0).max()))
            # Counterfactual initial-function control only, never a trained or
            # promoted policy. Both actor and critic column are zeroed.
            with torch.no_grad():
                model.policy.mlp_extractor.policy_net[0].weight[:,38].zero_()
                model.policy.mlp_extractor.value_net[0].weight[:,38].zero_()
            changed0,changed_value0=response(0)
            np.testing.assert_array_equal(changed0,mean0);np.testing.assert_array_equal(changed_value0,value0)
            for y in [-3.,3.]:
                mean,value=response(y);np.testing.assert_array_equal(mean,mean0);np.testing.assert_array_equal(value,value0)
            sensitivity[str(seed)]=dict(initial_weight_sha256=initial,normalized_route_feature_range=[-3,3],original=before,
                zero_column_restores_exact_actor_and_critic_initial_function=True)
        finally:env.close()
    report=dict(verified=True,study_review_sha256=sha(OUT/'study_review.json'),state_bank_sha256=sha(bankpath),
        source_sha256=sha(__file__),failure_height_groups=groups,design_failure_cases=failures,initial_function_sensitivity=sensitivity,
        training_updates=0,physics_episodes=0,old_gate_or_final_used=False,
        interpretation='Torque-feasible outputs did not preserve the1.4rad design domain in these finite evaluations. Extra feature exposure changes fresh-policy and value functions despite equal parameters; zeroing its first-layer columns removes this initial-function change algebraically.',
        limits='Overlapping predicates are not root causes. Static reference inputs use initial RMS identity and hypothetical normalized route values; not actual training occupancy or a proof that initialization caused the negative pilot. Zero-column experiment is not learning evidence or novelty. Dynamic state-safety still requires model/contact analysis.')
    (OUT/'mechanism_audit_120.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True,groups=groups,design_failures=len(failures),initial_function_sensitivity=sensitivity)))


if __name__=='__main__':run()
