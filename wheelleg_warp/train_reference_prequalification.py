"""Run the frozen six-run reference study only after explicit source admission."""
import json
import time
import numpy as np
import torch
import mujoco
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from reference_learning_runtime import make_env
from reference_learning_engineering import agent,Ledger
from smoke_reward_training import check_agent,weight_digest
from train_height_comparison import equal
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/reference_learning_prequalification_v1'


class StudyLedger(Ledger):
    def save(self):
        if self.model.num_timesteps%20000==0:
            super().save()
            path=self.directory/f'step_{self.model.num_timesteps}.json'
            record=json.loads(path.read_text());record['engineering_only']=False
            record['prequalification_only']=True;atomic_json(path,record)


def verify():
    p=json.loads((OUT/'proposal.json').read_text())
    admit=json.loads((OUT/'main_source_admission.json').read_text())
    assert admit['verified'] and admit['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/f)==h for f,h in admit['source_sha256'].items())
    assert p['worlds']==100 and p['seeds']==[32031,32032,32033] and p['arms']==['M_ref3','U_ref6']
    assert p['policy_samples_per_run']==200000 and p['checkpoints']==list(range(20000,200001,20000))
    return p


def run():
    p=verify();directory=OUT/'training';directory.mkdir(exist_ok=False)
    torch.set_num_threads(1);reports={};fd_calls=[];fd=mujoco.mjd_transitionFD;launch=wp.capture_launch;graph_calls=0
    def counted(*args,**kwargs):fd_calls.append(float(args[2]));return fd(*args,**kwargs)
    def capture(*args,**kwargs):
        nonlocal graph_calls
        graph_calls+=1;assert graph_calls<=12000,'Registered six-run graph budget exceeded'
        return launch(*args,**kwargs)
    mujoco.mjd_transitionFD=counted;wp.capture_launch=capture
    atomic_json(directory/'started.json',dict(admission_sha256=sha(OUT/'main_source_admission.json'),proposal_sha256=sha(OUT/'proposal.json')))
    try:
        for seed in p['seeds']:
            shared=None;world=None
            for arm in p['arms']:
                verify();name=f'{arm}_{seed}';out=directory/name;out.mkdir(exist_ok=False)
                start=time.perf_counter();before_graph=graph_calls;curriculum,raw,history,norm=make_env(p,arm,seed)
                model=None;ledger=None
                try:
                    current=[b.numpy().copy() for b in (raw.q0,raw.param,raw.k['gains'],raw.k['reference'])]
                    if world is None:world=current
                    else:equal(world,current)
                    np.savez_compressed(out/'initial_world.npz',q=current[0],param=current[1],gains=current[2],reference=current[3])
                    model,shared=agent({**p,'engineering_seed':seed},norm,shared)
                    initial=weight_digest(model)
                    torch.save(dict(policy=model.policy.state_dict(),optimizer=model.policy.optimizer.state_dict(),RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),out/'initial_policy.pt')
                    ledger=StudyLedger(norm,out);learn_start=time.perf_counter()
                    model.learn(total_timesteps=200000,callback=ledger)
                    assert model.num_timesteps==200000 and model._n_updates==400 and curriculum.policy_steps==200000
                    assert ledger.saved==list(range(20000,200001,20000)) and weight_digest(model)!=initial
                    check_agent(model,8000)
                    loaded=PPO.load(out/'step_200000.zip',device='cuda')
                    equal(model.policy.state_dict(),loaded.policy.state_dict());equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict())
                    restored=VecNormalize.load(out/'step_200000.pkl',norm.venv)
                    for field in ('obs_rms','ret_rms'):
                        for key in ('mean','var','count'):equal(getattr(getattr(norm,field),key),getattr(getattr(restored,field),key))
                    equal(model.predict(model._last_obs,deterministic=True)[0],loaded.predict(model._last_obs,deterministic=True)[0])
                    np.testing.assert_allclose(norm.obs_rms.count,200100.0001,rtol=0,atol=1e-7)
                    atomic_json(out/'episodes.json',dict(episodes=ledger.episodes,curriculum_transitions=curriculum.transition_log))
                    np.savez_compressed(out/'final_history.npz',frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid,stages=curriculum.stages)
                    assert graph_calls-before_graph==2000
                    report=dict(policy_samples=200000,epochs=400,Adam_steps=8000,actual_graph_world_steps=(graph_calls-before_graph)*40*100,completed_episodes=len(ledger.episodes),
                        first_complete_episode_world_steps=sum(r['physical_steps'] for r in ledger.episodes),total_wall_seconds=time.perf_counter()-start,
                        learn_checkpoint_reload_wall_seconds=time.perf_counter()-learn_start,policy_device=str(model.device),physics_device=str(raw.data.qpos.device),
                        parameters=sum(v.numel() for v in model.policy.parameters()),checkpoints=ledger.saved,initial_weights=initial,reload_verified=True)
                    atomic_json(out/'verification.json',report);reports[name]=report
                except BaseException as error:
                    if model is not None:model.save(out/'interrupted_model.zip');norm.save(out/'interrupted_RMS.pkl')
                    np.savez_compressed(out/'interrupted_world.npz',q=raw.data.qpos.numpy(),v=raw.data.qvel.numpy(),state=raw.state.numpy(),controller=raw.k['state'].numpy(),requested=raw._joint_requested.numpy(),filtered=raw._joint_buffers[4].numpy(),frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)
                    torch.save(dict(RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),out/'interrupted_RNG.pt')
                    atomic_json(out/'interrupted_episodes.json',dict(episodes=[] if ledger is None else ledger.episodes,curriculum_transitions=curriculum.transition_log))
                    atomic_json(out/'failure.json',dict(error=repr(error),policy_samples=0 if model is None else model.num_timesteps,implicit_retry=False))
                    raise
                finally:norm.close()
        verify()
        atomic_json(directory/'completion.json',dict(verified=True,reports=reports,policy_samples=1200000,actual_graph_world_steps=graph_calls*40*100,constructor_FD_calls=fd_calls,scientific_evaluations=0,formal5_admitted=False))
    except BaseException as error:
        atomic_json(directory/'failure.json',dict(error=repr(error),completed=list(reports),FD_calls=fd_calls,implicit_retry=False))
        raise
    finally:mujoco.mjd_transitionFD=fd;wp.capture_launch=launch


if __name__=='__main__':run()
