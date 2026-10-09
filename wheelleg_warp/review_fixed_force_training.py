"""Read all60 actual study checkpoints/Adam/RMS and6complete curricula."""
import json
import pickle
import numpy as np
import torch
from stable_baselines3 import PPO
from train_fixed_force_study import OUT,verify
from smoke_reward_training import weight_digest
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run():
    p=verify();completion=json.loads((OUT/'training/completion.json').read_text())
    assert not (OUT/'training_review.json').exists() and completion['verified'] and completion['policy_samples']==1200000 and len(completion['reports'])==6
    records=[];inputs={};total_episodes=0;observations=np.random.default_rng(33731).normal(size=(64,481)).astype(np.float32)
    for seed in p['seeds']:
        initial={};worlds={}
        for arm,dim in (('D3',3),('V6',6)):
            name=f'{arm}_{seed}';d=OUT/'training'/name;r=completion['reports'][name]
            assert r['policy_samples']==200000 and r['actual_graph_world_steps']==8000000 and r['reload_verified']
            initial[arm]=torch.load(d/'initial_policy.pt',map_location='cpu',weights_only=False);assert not initial[arm]['optimizer']['state']
            with np.load(d/'initial_world.npz') as z:worlds[arm]={k:z[k].copy() for k in z.files}
            previous=initial[arm]['policy'];previous_count=.0001
            for step in p['checkpoints']:
                f=d/f'step_{step}.json';meta=json.loads(f.read_text());model_file=d/f'step_{step}.zip';norm_file=d/f'step_{step}.pkl'
                assert sha(model_file)==meta['model_sha256'] and sha(norm_file)==meta['normalization_sha256'] and meta['prequalification_only'] and not meta['engineering_only']
                model=PPO.load(model_file,device='cpu');assert model.num_timesteps==step and model._n_updates==step//5000*10 and model.action_space.shape==(dim,) and model.observation_space.shape==(481,)
                weights=model.policy.state_dict();assert weight_digest(model)==meta['weights_sha256'] and all(torch.isfinite(v).all() for v in weights.values()) and any(not torch.equal(v,previous[k]) for k,v in weights.items())
                states=model.policy.optimizer.state_dict()['state'];assert states
                for state in states.values():
                    assert int(state['step'].item())==step//5000*200 and all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
                with norm_file.open('rb') as stream:norm=pickle.load(stream)
                assert norm.obs_rms.mean.shape==(481,) and np.isfinite(norm.obs_rms.mean).all() and np.isfinite(norm.obs_rms.var).all()
                np.testing.assert_allclose(norm.obs_rms.count,step+100.0001,rtol=0,atol=1e-7);assert norm.obs_rms.count>previous_count
                pred=model.predict(norm.normalize_obs(observations.copy()),deterministic=True)[0];assert pred.shape==(64,dim) and np.isfinite(pred).all() and abs(pred).max()<=1
                assert meta['training_metrics'] and all(np.isfinite(v) for v in meta['training_metrics'].values())
                records.append(dict(run=name,step=step,epochs=model._n_updates,Adam_steps=step//5000*200,model_sha256=sha(model_file),RMS_sha256=sha(norm_file),metadata_sha256=sha(f)))
                previous=weights;previous_count=norm.obs_rms.count
            episodes=json.loads((d/'episodes.json').read_text());rows=episodes['episodes'];assert rows and all(row['physical_safety_passed'] and row['design_joint_passed'] and 0<=row['world_index']<100 for row in rows)
            total_episodes+=len(rows);assert all(row['actual_episode_end'] for row in episodes['curriculum_transitions'])
            with np.load(d/'final_state.npz') as z:
                assert z['frames'].shape==(100,10,39) and z['inputs'].shape==(100,9,8) and z['guard_memory'].shape==(100,16)
                assert all(np.isfinite(z[k]).all() for k in z.files) and 3 in z['stages']
            for n in ('episodes.json','final_state.npz','verification.json','initial_policy.pt','initial_world.npz'):
                inputs[str((d/n).relative_to(ROOT))]=sha(d/n)
            print('RECEIVED337',name,'10actualcheckpoint/Adam/RMS andfinalstate',flush=True)
        a,b=initial['D3']['policy'],initial['V6']['policy'];keys=[k for k in a if k.startswith(('mlp_extractor.','value_net.'))]
        assert keys and all(torch.equal(a[k],b[k]) for k in keys)
        for state in (a,b):assert not torch.count_nonzero(state['action_net.weight']) and not torch.count_nonzero(state['action_net.bias'])
        for k in worlds['D3']:np.testing.assert_array_equal(worlds['D3'][k],worlds['V6'][k])
    assert len(records)==60
    atomic_json(OUT/'training_review.json',dict(round=337,verified=True,completion_sha256=sha(OUT/'training/completion.json'),reviewer_sha256=sha(__file__),records=records,input_sha256=inputs,
        actual_policy_samples_received=1200000,actual_graph_world_steps=48000000,training_episodes=total_episodes,scientific_evaluations=0,new_learning_samples=0,formal5_admitted=False,
        scope='actualweights/optimizer/RMS/commoninit/curriculum/physical-design training receipt;notmethodadvantage orcheckpointselection. Onlyfinal200k mayenterregistered984eval'))


if __name__=='__main__':run()
