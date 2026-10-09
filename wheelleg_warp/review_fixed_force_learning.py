"""Independent receipt of real short-force checkpoints and physical guard traces."""
import json
import pickle
import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO
from learn_fixed_force_engineering import OUT
from smoke_reward_training import weight_digest
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run():
    assert not (OUT/'engineering_review.json').exists()
    c=json.loads((OUT/'completion.json').read_text());a=json.loads((OUT/'source_admission.json').read_text());d=json.loads((OUT/'delivery.json').read_text())
    assert c['verified'] and c['total_policy_samples']==8000 and d['completion_sha256']==sha(OUT/'completion.json')
    assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    observations=np.random.default_rng(33231).normal(size=(128,481)).astype(np.float32)
    m=mujoco.MjModel.from_xml_path(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'))
    ids=np.array([m.jnt_qposadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
    vids=np.array([m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
    records=[];initials={};worlds={};guard_stats={};raw_steps=0;artifacts={}
    for arm,dim in (('D3',3),('V6',6)):
        directory=OUT/arm;initial=torch.load(directory/'initial_policy.pt',map_location='cpu',weights_only=False)
        assert not initial['optimizer']['state'];initials[arm]=initial
        with np.load(directory/'initial_world.npz') as z:worlds[arm]={k:z[k].copy() for k in z.files}
        previous=initial['policy']
        for step in range(500,4001,500):
            file=directory/f'step_{step}.json';meta=json.loads(file.read_text());archive=directory/f'step_{step}.zip';rms=directory/f'step_{step}.pkl'
            assert sha(archive)==meta['model_sha256'] and sha(rms)==meta['normalization_sha256']
            model=PPO.load(archive,device='cpu');assert model.num_timesteps==step and model._n_updates==step//500*10
            assert model.observation_space.shape==(481,) and model.action_space.shape==(dim,)
            weights=model.policy.state_dict();assert all(torch.isfinite(v).all() for v in weights.values()) and weight_digest(model)==meta['weights_sha256']
            assert any(not torch.equal(weights[k],previous[k]) for k in weights)
            states=model.policy.optimizer.state_dict()['state'];assert states
            for state in states.values():
                assert int(state['step'].item())==step//500*20
                assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
            assert meta['training_metrics'] and all(np.isfinite(v) for v in meta['training_metrics'].values())
            with rms.open('rb') as stream:norm=pickle.load(stream)
            assert norm.obs_rms.mean.shape==(481,) and np.isfinite(norm.obs_rms.mean).all() and np.isfinite(norm.obs_rms.var).all() and (norm.obs_rms.var>=0).all()
            np.testing.assert_allclose(norm.obs_rms.count,step+10.0001,rtol=0,atol=1e-7)
            normalized=norm.normalize_obs(observations.copy());pred=model.predict(normalized,deterministic=True)[0]
            assert pred.shape==(128,dim) and np.isfinite(pred).all() and abs(pred).max()<=1
            records.append(dict(arm=arm,step=step,epochs=model._n_updates,Adam_steps=step//500*20,parameters=sum(v.numel() for v in model.policy.parameters()),model_sha256=sha(archive),RMS_sha256=sha(rms),metadata_sha256=sha(file),weight_update_verified=True))
            previous=weights
        with np.load(directory/'final_history.npz') as z:
            assert z['frames'].shape==(10,10,39) and z['inputs'].shape==(10,9,8) and z['elapsed'].shape==(10,9) and z['valid'].shape==(10,10)
            assert all(np.isfinite(z[k]).all() for k in z.files)
        rows=json.loads((directory/'episodes.json').read_text())['episodes'];assert len(rows)==10
        gs=[]
        for row in rows:
            assert row['physical_safety_passed'] and row['design_joint_passed'] and row['physical_evidence_steps']==row['physical_steps']
            f=directory/row['complete_trace']['path'];assert sha(f)==row['complete_trace']['sha256'];artifacts[str(f.relative_to(ROOT))]=sha(f)
            with np.load(f) as z:pre=z['pre'];post=z['post'];trace=z['trace'];cols={str(k):i for i,k in enumerate(z['columns'])}
            assert len(trace)==row['physical_steps'];np.testing.assert_array_equal(pre[1:,:33],post[:-1,:33])
            margin=1.4-abs(post[:,ids]).max(axis=1);np.testing.assert_array_equal(margin,trace[:,cols['design_margin']]);assert margin.min()==row['min_active_design_margin_rad'] and margin.min()>=0
            gf=directory/row['joint_guard_trace']['path'];assert sha(gf)==row['joint_guard_trace']['sha256'];artifacts[str(gf.relative_to(ROOT))]=sha(gf)
            with np.load(gf) as z:g=z['trace']
            assert g.shape==(len(trace),62);np.testing.assert_array_equal(g[:,50:54],pre[:,ids]);np.testing.assert_array_equal(g[:,54:58],pre[:,17+vids]);np.testing.assert_array_equal(g[:,20:26],pre[:,33:39]);assert (g[:,28:30]==1).all() and g[:,61].max()<=1e-6 and g[:,58].max()<=1e-5
            raw_steps+=len(g);gs.append(g)
        g=np.concatenate(gs);guard_stats[arm]=dict(training_episodes=10,minimum_actual_design_margin_rad=min(r['min_active_design_margin_rad'] for r in rows),predicted_infeasible=0,model_error_max_rad_s2=float(abs(g[:,38:42]).max()),secondary_barrier_min=float(g[:,59].min()),nominal_corrected_substeps=int((g[:,60]>1e-12).sum()),residual_reduced_substeps=int((g[:,27]<1-1e-9).sum()))
        print('RECEIVED332',arm,'8realmodels/Adam/RMS+10physicalguardepisodes',flush=True)
    ia,ib=initials['D3']['policy'],initials['V6']['policy'];shared=[k for k in ia if k.startswith(('mlp_extractor.','value_net.'))]
    assert shared and all(torch.equal(ia[k],ib[k]) for k in shared)
    for initial in (ia,ib):
        assert not torch.count_nonzero(initial['action_net.weight']) and not torch.count_nonzero(initial['action_net.bias'])
        torch.testing.assert_close(initial['log_std'],torch.full_like(initial['log_std'],np.log(.25)),rtol=0,atol=1e-7)
    for k in worlds['D3']:np.testing.assert_array_equal(worlds['D3'][k],worlds['V6'][k])
    assert len(records)==16
    atomic_json(OUT/'engineering_review.json',dict(round=332,verified=True,completion_sha256=sha(OUT/'completion.json'),delivery_sha256=sha(OUT/'delivery.json'),reviewer_sha256=sha(__file__),source_admission_sha256=sha(OUT/'source_admission.json'),
        records=records,guard_stats=guard_stats,physical_archive_first_episode_steps=raw_steps,physical_artifact_sha256=artifacts,initial_feature_value_world_equal=True,
        actual_learning_samples_received=8000,new_learning_samples=0,new_robot_physics_steps=0,scientific_evaluations=0,formal5_admitted=False,
        limits='CPUmodel file validation and128syntheticpredicts areengineering only;noCPUrobot/qualifiednew3 seeds/independentperformance/100worldguard training validation;unknownmodel andfullnoise bounds remain',
        next='333implement sharedguard light100world/curriculum+trainedforce evaluator reuse;335directionreview beforeanynew scientificlearning budget;no shortcheckpoint selection'))
    print('PASS332 independent16actualpolicy/Adam/RMS/commoninit/world/history/20guardepisodes;0newphysics/learning')


if __name__=='__main__':run()
