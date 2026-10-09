"""Read actual16 checkpoints/Adam/RMS/episodes; never samples or trains."""
import json
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO
from reference_learning_engineering import OUT
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run():
    assert not (OUT/'engineering_review.json').exists()
    config=json.loads((OUT/'runtime_config.json').read_text());completion=json.loads((OUT/'completion.json').read_text())
    source=json.loads((OUT/'source_admission.json').read_text());delivery=json.loads((OUT/'short_engineering_delivery.json').read_text())
    assert completion['verified'] and completion['total_policy_samples']==8000 and not completion['formal_PPO_admitted']
    assert delivery['completion_sha256']==sha(OUT/'completion.json')
    assert all(sha(ROOT/f)==h for f,h in source['source_sha256'].items())
    records=[];initials={};worlds={};rng=np.random.default_rng(31931)
    observations=rng.normal(size=(128,481)).astype(np.float32)
    for arm,dim in (('M_ref3',3),('U_ref6',6)):
        directory=OUT/arm
        initial=torch.load(directory/'initial_policy.pt',map_location='cpu',weights_only=False)
        assert not initial['optimizer']['state'];initials[arm]=initial
        with np.load(directory/'initial_world.npz') as z:worlds[arm]={name:z[name].copy() for name in z.files}
        previous=initial['policy'];previous_count=10.0001
        for step in range(500,4001,500):
            descriptor=json.loads((directory/f'step_{step}.json').read_text())
            archive=directory/f'step_{step}.zip';rms_file=directory/f'step_{step}.pkl'
            assert descriptor['model_sha256']==sha(archive) and descriptor['normalization_sha256']==sha(rms_file)
            agent=PPO.load(str(archive),device='cpu')
            assert agent.num_timesteps==step and agent._n_updates==step//500*10
            assert agent.observation_space.shape==(481,) and agent.action_space.shape==(dim,)
            weights=agent.policy.state_dict();assert all(torch.isfinite(t).all() for t in weights.values())
            states=agent.policy.optimizer.state_dict()['state'];assert states
            for state in states.values():
                assert int(state['step'].item())==step//500*20
                assert all(torch.isfinite(x).all() for x in state.values() if isinstance(x,torch.Tensor))
            changed=any(not torch.equal(weights[name],previous[name]) for name in weights);assert changed
            assert descriptor['training_metrics'] and all(np.isfinite(v) for v in descriptor['training_metrics'].values())
            with rms_file.open('rb') as file:normalizer=pickle.load(file)
            assert normalizer.obs_rms.mean.shape==(481,) and np.isfinite(normalizer.obs_rms.var).all()
            np.testing.assert_allclose(normalizer.obs_rms.count,step+10.0001,rtol=0,atol=1e-7)
            assert normalizer.obs_rms.count>previous_count
            output=agent.predict(observations,deterministic=True)[0];assert output.shape==(128,dim) and np.isfinite(output).all()
            records.append(dict(arm=arm,step=step,epochs=agent._n_updates,adam_steps=step//500*20,
                actual_parameters=sum(p.numel() for p in agent.policy.parameters()),weights_changed_since_previous=True,
                normalization_count=float(normalizer.obs_rms.count),model_sha256=sha(archive),RMS_sha256=sha(rms_file)))
            previous=weights;previous_count=normalizer.obs_rms.count
        episodes=json.loads((directory/'episodes.json').read_text())['episodes'];assert len(episodes)==10
        assert all(e['physical_safety_passed'] and e['design_joint_passed'] for e in episodes)
        with np.load(directory/'final_history.npz') as z:
            assert z['frames'].shape==(10,10,39) and z['inputs'].shape==(10,9,8)
            assert z['elapsed'].shape==(10,9) and z['valid'].shape==(10,10)
            assert all(np.isfinite(z[n]).all() for n in z.files)
        print('RECEIVED319',arm,'8actualcheckpoints',flush=True)
    a,b=[initials[arm]['policy'] for arm in ('M_ref3','U_ref6')]
    shared=[name for name in a if name.startswith('mlp_extractor.') or name.startswith('value_net.')]
    assert shared and all(torch.equal(a[n],b[n]) for n in shared)
    for w in (a,b):
        assert torch.count_nonzero(w['action_net.weight'])==0 and torch.count_nonzero(w['action_net.bias'])==0
    assert all(np.array_equal(worlds['M_ref3'][name],worlds['U_ref6'][name]) for name in worlds['M_ref3'])
    assert len(records)==16
    atomic_json(OUT/'engineering_review.json',dict(verified=True,round=319,records=records,
        completion_sha256=sha(OUT/'completion.json'),source_admission_sha256=sha(OUT/'source_admission.json'),reviewer_sha256=sha(__file__),
        initial_feature_critic_world_equal=True,all16realcheckpoint_updates_andfiniteAdam_RMS_checked=True,
        completed_training_episodes=20,all_episode_physical_design_passed=True,
        actual_graph_world_steps=320000,baseline_constructor_FD_calls=len(completion['baseline_constructor_FD_calls']),
        actual_policy_samples=8000,new_learning_samples=0,new_physics_steps=0,scientific_evaluations=0,
        matched_physical_exploration_claim=False,formal_PPO_admitted=False,goal_complete=False,
        limits='CPUreload checks areindependent artifactvalidation,noCPUrobotrollout orscienceperformance. Initialactuation sourceonlyfirstsubstep; fulllearning trajectories differ. Engineeringcounts are notmethodbenefit.',
        next='320mandatory direction review/cleanup decides whetherfreshmatchedthree-seed prequalification merits boundedsource/runtime/curriculum protocol. Do notpromote shortcheckpoint orstartformal5.'))
    print('PASS319 independent16policy/Adam/RMS/update/source/episode receipt;0newlearn/physics',flush=True)


if __name__=='__main__':run()
