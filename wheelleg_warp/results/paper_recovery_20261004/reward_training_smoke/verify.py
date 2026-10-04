"""Independent saved-counter, weight/Adam and both RMS reload audit; no training."""
from pathlib import Path
import hashlib,json,pickle,sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

p=Path(__file__).resolve().parent;root=p.parents[3]
sys.path[:0]=[str(root/'wheelleg_warp'),str(root/'wheelleg_ppo/tools')]
from smoke_reward_training import make_env
from train_height_comparison import equal
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
pre=json.loads((p/'registration.json').read_text());result=json.loads((p/'result.json').read_text())
assert pre['policy_steps_per_arm']==12000 and result['total_engineering_policy_steps']==24000
for name,value in pre['source_sha256'].items():assert value==result['source_sha256'][name]==sha(root/'wheelleg_warp'/name)
frozen=root/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json'
config=json.loads(frozen.read_text());assert pre['protocol_sha256']==sha(frozen)
assert all(sha(root/name)==value for name,value in config['source_sha256'].items())
initial=[];loaded_checks=[];artifacts={}
for arm in ['original','potential']:
    report=json.loads((p/arm/'verification.json').read_text());initial.append(report['initial_weight_sha256'])
    assert report['policy_steps']==12000 and report['ppo_epochs']==240 and report['adam_updates']==480
    assert report['actor_value_device']=='cuda' and report['physics_device'].startswith('cuda')
    assert report['weights_optimizer_RMS_restored_exactly'] and not report['physical_state_potential_RNG_trajectory_restored']
    assert report['episodes'] and set(report['stages_after_resume'])=={3}
    raw,potential,checked=make_env(config,arm,start=12000)
    try:
        for steps in [6000,12000]:
            prefix=p/arm/f'step_{steps}';hashes=report[f'checkpoint_{steps}']
            assert sha(prefix.with_suffix('.zip'))==hashes['checkpoint_sha256']
            assert sha(prefix.with_suffix('.pkl'))==hashes['normalization_sha256']
            with prefix.with_suffix('.pkl').open('rb') as f:encoded=pickle.load(f)
            restored=VecNormalize.load(str(prefix)+'.pkl',checked)
            for name in ['obs_rms','ret_rms']:
                for field in ['mean','var','count']:equal(getattr(getattr(encoded,name),field),getattr(getattr(restored,name),field))
                assert np.isfinite(getattr(restored,name).mean).all() and np.isfinite(getattr(restored,name).var).all()
            assert abs(restored.ret_rms.count-(steps+.0001))<1e-8
            agent=PPO.load(str(prefix)+'.zip',device='cpu')
            assert agent.num_timesteps==steps and agent._n_updates==steps//500*10
            assert agent.observation_space.shape==(38,) and agent.action_space.shape==(3,)
            assert agent.gamma==restored.gamma==.99 and restored.norm_reward is False
            state=agent.policy.optimizer.state_dict()['state'];assert state
            assert {int(v['step'].item()) for v in state.values()}=={steps//500*20}
            assert all(torch.isfinite(v).all() for v in agent.policy.state_dict().values())
            assert all(torch.isfinite(v).all() for s in state.values() for v in s.values() if isinstance(v,torch.Tensor))
            loaded_checks.append(dict(arm=arm,steps=steps,observation_RMS_exact=True,reward_RMS_exact=True,
                obs_count=restored.obs_rms.count,reward_count=restored.ret_rms.count,adam_steps=steps//500*20))
            artifacts[str(prefix.relative_to(p))+'.zip']=sha(prefix.with_suffix('.zip'))
            artifacts[str(prefix.relative_to(p))+'.pkl']=sha(prefix.with_suffix('.pkl'))
    finally:checked.close()
assert initial[0]==initial[1]==result['matched_initial_weight_sha256']
potential_report=json.loads((p/'potential/verification.json').read_text())
assert potential_report['nonzero_shaping_transitions']>0 and potential_report['potential_complete_episodes']>0
verification=dict(verified=True,total_engineering_steps=24000,initial_weights_matched=True,checks=loaded_checks,
    source_training_scope='Actual actor/value/Adam/physics CUDA evidence from primary run; these artifact reloads use CPU for saved model inspection and no physics steps/training.',
    restoration_scope='Runtime before/after save/load compared weights/Adam/observationRMS; independent encoded-file reload checks both observation and rewardRMS. Physical episodes, Phi and RNG trajectory restart; full trajectory continuation not claimed.',
    study_status='Engineering only: one seed,10worlds, two segmented12k arms; no performance comparison, no new independent evaluation, no checkpoint promotion.',
    next='Round100 objective review and duplicate cleanup, then freeze a bounded new same-information study only if contribution/protocol checks pass.',
    artifact_sha256=artifacts,verifier_sha256=sha(Path(__file__)))
(p/'independent_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
print('PASS both arms24000steps; four saved actor/Adam counters and both observation/reward RMS reloads; no additional learning')
