"""Fresh guarded fixed-Nom force study, blocked until explicit frozen admission."""
import json
import time
import numpy as np
import torch
import mujoco
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from fixed_reference_force import make_env
from reference_learning_engineering import agent,Ledger
from smoke_reward_training import weight_digest,check_agent
from train_height_comparison import equal
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_prequalification_v1'


class StudyLedger(Ledger):
    def __init__(self,norm,directory,curriculum,raw):
        super().__init__(norm,directory);self.curriculum=curriculum;self.raw=raw
    def save(self):
        if self.model.num_timesteps%20000:return
        super().save();path=self.directory/f'step_{self.model.num_timesteps}.json'
        row=json.loads(path.read_text());row['engineering_only']=False;row['prequalification_only']=True
        row['guard']=dict(self.raw._joint_guard_stats);row['curriculum_stage_counts']=np.bincount(self.curriculum.stages,minlength=4).tolist()
        atomic_json(path,row)
    def _on_step(self):
        for w,info in enumerate(self.locals['infos']):
            if 'episode' in info:
                self.episodes.append({**{k:v for k,v in info.items() if k!='terminal_observation'},'world_index':w})
                assert info['physical_safety_passed'] and info['design_joint_passed'],'Training physical/design violation'
        return True


def verify():
    p=json.loads((OUT/'proposal.json').read_text());a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json') and all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    assert p['worlds']==100 and p['arms']==['D3','V6'] and len(p['seeds'])==len(set(p['seeds']))==3
    assert p['policy_samples_per_run']==200000 and p['checkpoints']==list(range(20000,200001,20000))
    assert p['ppo']['n_steps']==50 and p['ppo']['batch_size']==250 and p['ppo']['n_epochs']==10
    return p


def run():
    p=verify();directory=OUT/'training';directory.mkdir(exist_ok=False);torch.set_num_threads(1)
    reports={};calls=0;fd_calls=[];graph=wp.capture_launch;fd=mujoco.mjd_transitionFD
    def capture(*args,**kwargs):
        nonlocal calls
        calls+=1;assert calls<=12000,'Fixed six-run graph budget exceeded';return graph(*args,**kwargs)
    def counted(*args,**kwargs):fd_calls.append(float(args[2]));return fd(*args,**kwargs)
    wp.capture_launch=capture;mujoco.mjd_transitionFD=counted
    atomic_json(directory/'started.json',dict(admission_sha256=sha(OUT/'source_admission.json'),proposal_sha256=sha(OUT/'proposal.json')))
    try:
        for seed in p['seeds']:
            world=None;shared=None
            for arm in p['arms']:
                verify();name=f'{arm}_{seed}';out=directory/name;out.mkdir(exist_ok=False);before=calls;begin=time.perf_counter()
                curriculum,raw,history,norm=make_env(p,arm,seed,guarded=True);model=None;ledger=None
                try:
                    current=[v.numpy().copy() for v in (raw.q0,raw.param,raw.k['gains'],raw.k['reference'])]
                    if world is None:world=current
                    else:equal(world,current)
                    np.savez_compressed(out/'initial_world.npz',q=current[0],param=current[1],gains=current[2],reference=current[3])
                    model,shared=agent({**p,'engineering_seed':seed},norm,shared);initial=weight_digest(model)
                    torch.save(dict(policy=model.policy.state_dict(),optimizer=model.policy.optimizer.state_dict(),RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),out/'initial_policy.pt')
                    ledger=StudyLedger(norm,out,curriculum,raw);start=time.perf_counter();model.learn(total_timesteps=200000,callback=ledger)
                    assert model.num_timesteps==curriculum.policy_steps==200000 and model._n_updates==400 and calls-before==2000
                    assert ledger.saved==p['checkpoints'] and weight_digest(model)!=initial;check_agent(model,8000)
                    loaded=PPO.load(out/'step_200000.zip',device='cuda');equal(model.policy.state_dict(),loaded.policy.state_dict());equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict())
                    restored=VecNormalize.load(out/'step_200000.pkl',norm.venv)
                    for field in ('obs_rms','ret_rms'):
                        for key in ('mean','var','count'):equal(getattr(getattr(norm,field),key),getattr(getattr(restored,field),key))
                    equal(model.predict(model._last_obs,deterministic=True)[0],loaded.predict(model._last_obs,deterministic=True)[0])
                    np.testing.assert_allclose(norm.obs_rms.count,200100.0001,rtol=0,atol=1e-7)
                    valid=sum(r['physical_steps'] for r in ledger.episodes)+int(raw.state.numpy()[:,0].sum())
                    assert valid==raw._joint_guard_stats['valid_substeps'],'Guard/episode/partial physical counter mismatch'
                    atomic_json(out/'episodes.json',dict(episodes=ledger.episodes,curriculum_transitions=curriculum.transition_log))
                    np.savez_compressed(out/'final_state.npz',q=raw.data.qpos.numpy(),v=raw.data.qvel.numpy(),terminal_q=raw.stopped_q.numpy(),terminal_v=raw.stopped_v.numpy(),frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid,guard_memory=raw._joint_guard_buffers[0].numpy(),guard_last=raw._joint_guard_buffers[1].numpy(),stages=curriculum.stages)
                    reports[name]=dict(verified=True,policy_samples=200000,epochs=400,Adam_steps=8000,actual_graph_world_steps=(calls-before)*40*100,completed_episode_steps=sum(r['physical_steps'] for r in ledger.episodes),valid_guard_substeps=valid,
                        guard=dict(raw._joint_guard_stats),total_wall_seconds=time.perf_counter()-begin,learn_checkpoint_reload_wall_seconds=time.perf_counter()-start,
                        policy_device=str(model.device),physics_device=str(raw.data.qpos.device),reload_verified=True,initial_weights=initial,parameters=sum(v.numel() for v in model.policy.parameters()))
                    atomic_json(out/'verification.json',reports[name])
                except BaseException as error:
                    if model is not None:model.save(out/'interrupted_model.zip');norm.save(out/'interrupted_RMS.pkl')
                    np.savez_compressed(out/'interrupted_state.npz',q=raw.data.qpos.numpy(),v=raw.data.qvel.numpy(),terminal_q=raw.stopped_q.numpy(),terminal_v=raw.stopped_v.numpy(),state=raw.state.numpy(),control=raw.k['state'].numpy(),guard_memory=raw._joint_guard_buffers[0].numpy(),guard_last=raw._joint_guard_buffers[1].numpy(),frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)
                    torch.save(dict(RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),out/'interrupted_RNG.pt')
                    atomic_json(out/'interrupted_episodes.json',dict(episodes=[] if ledger is None else ledger.episodes,guard=dict(raw._joint_guard_stats),transitions=curriculum.transition_log))
                    atomic_json(out/'failure.json',dict(error=repr(error),policy_samples=0 if model is None else model.num_timesteps,actual_capture_calls=calls-before,implicit_retry=False));raise
                finally:norm.close()
        verify();atomic_json(directory/'completion.json',dict(verified=True,reports=reports,policy_samples=1200000,actual_graph_world_steps=calls*40*100,FD_calls=fd_calls,scientific_evaluations=0,formal5_admitted=False))
    except BaseException as error:
        atomic_json(directory/'failure.json',dict(error=repr(error),completed=list(reports),actual_capture_calls=calls,actual_graph_world_steps=calls*40*100,FD_calls=fd_calls,implicit_retry=False));raise
    finally:wp.capture_launch=graph;mujoco.mjd_transitionFD=fd


if __name__=='__main__':run()
