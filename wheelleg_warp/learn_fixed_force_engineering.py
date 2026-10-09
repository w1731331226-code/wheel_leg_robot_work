"""Source-gated fresh2x4000 force-policy GPU lifecycle, not paper evaluation."""
import json
import time
import numpy as np
import torch
import mujoco
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecCheckNan,VecNormalize
import execution_history_evaluation as evaluation
import joint_state_guard as guard
from fixed_reference_force import ForceActions,self_check
from reference_learning_engineering import agent,Ledger
from smoke_reward_training import check_agent,weight_digest
from train_height_comparison import equal
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_learning_engineering_v1'


def verify():
    p=json.loads((OUT/'proposal.json').read_text());a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json') and all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    assert p['worlds']==10 and p['policy_samples_per_arm']==4000 and p['arms']==['D3','V6']
    return p


def run():
    p=verify();assert not any((OUT/n).exists() for n in ('started.json','completion.json','failure.json'))
    self_check();torch.set_num_threads(1);reports={};shared=None;world=None;fd_calls=[];graph_calls=0
    fd=mujoco.mjd_transitionFD;graph=wp.capture_launch
    def counted(*args,**kwargs):fd_calls.append(float(args[2]));assert len(fd_calls)<=30;return fd(*args,**kwargs)
    def capture(*args,**kwargs):
        nonlocal graph_calls
        graph_calls+=1;assert graph_calls<=800;return graph(*args,**kwargs)
    mujoco.mjd_transitionFD=counted;wp.capture_launch=capture
    atomic_json(OUT/'started.json',dict(admission_sha256=sha(OUT/'source_admission.json'),total_policy_samples=8000,science_evaluation=False))
    try:
        for arm in p['arms']:
            verify();directory=OUT/arm;directory.mkdir(exist_ok=False);begin=time.perf_counter();before_calls=graph_calls
            raw,route,history,_,_=guard.instrument(lambda:evaluation.make_env(p['cases'],'H1',None,directory),p['cases'],directory)
            norm=VecNormalize(VecCheckNan(ForceActions(history,arm),raise_exception=True),**p['normalization'])
            model=None;ledger=None
            try:
                # Qualification reset must not contribute a second training observation batch.
                norm.training=False;obs=norm.reset();norm.training=True
                assert obs.shape==(10,481) and norm.action_space.shape==(3 if arm=='D3' else 6,)
                assert norm.obs_rms.count==.0001 and not raw._joint_guard_buffers[0].numpy().any()
                current=[v.numpy().copy() for v in (raw.q0,raw.param,raw.k['gains'],raw.k['reference'])]
                if world is None:world=current
                else:equal(world,current)
                np.savez_compressed(directory/'initial_world.npz',q=current[0],param=current[1],gains=current[2],reference=current[3])
                model,shared=agent(p,norm,shared);initial=weight_digest(model)
                assert model.num_timesteps==model._n_updates==0 and not model.policy.optimizer.state_dict()['state']
                torch.save(dict(policy=model.policy.state_dict(),optimizer=model.policy.optimizer.state_dict(),RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),directory/'initial_policy.pt')
                atomic_json(directory/'runtime_api.json',dict(verified=True,observation=481,actions=3 if arm=='D3' else 6,actual_guard_captures=80,guard_reset_zero=True,policy_device=str(model.device),physics_device=str(raw.data.qpos.device)))
                ledger=Ledger(norm,directory);start=time.perf_counter();model.learn(total_timesteps=4000,callback=ledger)
                assert model.num_timesteps==4000 and model._n_updates==80 and ledger.saved==list(range(500,4001,500))
                assert weight_digest(model)!=initial and graph_calls-before_calls==400;check_agent(model,160)
                loaded=PPO.load(directory/'step_4000.zip',device='cuda');equal(model.policy.state_dict(),loaded.policy.state_dict());equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict());check_agent(loaded,160)
                restored=VecNormalize.load(directory/'step_4000.pkl',norm.venv)
                for name in ('obs_rms','ret_rms'):
                    for field in ('mean','var','count'):equal(getattr(getattr(norm,name),field),getattr(getattr(restored,name),field))
                equal(model.predict(model._last_obs,deterministic=True)[0],loaded.predict(model._last_obs,deterministic=True)[0])
                np.testing.assert_allclose(norm.obs_rms.count,4010.0001,rtol=0,atol=1e-7)
                np.savez_compressed(directory/'final_history.npz',frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)
                atomic_json(directory/'episodes.json',dict(episodes=ledger.episodes))
                report=dict(verified=True,policy_samples=4000,epochs=80,Adam_steps=160,actual_graph_world_steps=160000,completed_episodes=len(ledger.episodes),
                    checkpoints=ledger.saved,learn_checkpoint_reload_wall_seconds=time.perf_counter()-start,total_wall_seconds=time.perf_counter()-begin,
                    shared_initialization=True,parameters=sum(v.numel() for v in model.policy.parameters()),initial_weights=initial,reload_model_Adam_RMS_prediction_exact=True,
                    policy_device=str(model.device),physics_device=str(raw.data.qpos.device),science_evaluations=0)
                atomic_json(directory/'verification.json',report);reports[arm]=report
                print('LEARN331',arm,'4000/80epochs/160Adam complete',flush=True)
            except BaseException as error:
                if model is not None:model.save(directory/'interrupted_model.zip');norm.save(directory/'interrupted_RMS.pkl')
                np.savez_compressed(directory/'interrupted_state.npz',q=raw.data.qpos.numpy(),v=raw.data.qvel.numpy(),terminal_q=raw.stopped_q.numpy(),terminal_v=raw.stopped_v.numpy(),state=raw.state.numpy(),control=raw.k['state'].numpy(),guard_memory=raw._joint_guard_buffers[0].numpy(),guard_last=raw._joint_guard_buffers[1].numpy(),frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)
                torch.save(dict(RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),directory/'interrupted_RNG.pt')
                atomic_json(directory/'interrupted_episodes.json',dict(episodes=[] if ledger is None else ledger.episodes))
                atomic_json(directory/'failure.json',dict(error=repr(error),samples=0 if model is None else model.num_timesteps,implicit_retry=False));raise
            finally:norm.close()
        verify();atomic_json(OUT/'completion.json',dict(verified=True,reports=reports,total_policy_samples=8000,actual_graph_world_steps=graph_calls*40*10,FD_calls=fd_calls,science_evaluations=0,formal5_admitted=False))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),completed=list(reports),actual_capture_calls=graph_calls,FD_calls=fd_calls,implicit_retry=False));raise
    finally:mujoco.mjd_transitionFD=fd;wp.capture_launch=graph


if __name__=='__main__':run()
