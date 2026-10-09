"""Fixed-budget real100-world lifecycle qualification, never science selection."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import reference_learning_runtime as runtime
from reference_learning_engineering import agent,Ledger
from run_joint_reference_zero_pair import evaluate
from smoke_reward_training import check_agent,weight_digest
from train_height_comparison import equal
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/reference_learning_prequalification_v1'


def run():
    directory=OUT/'runtime_qualification';q=json.loads((directory/'proposal.json').read_text())
    assert not any((directory/n).exists() for n in ('started.json','completion.json','failure.json'))
    assert all(sha(ROOT/n)==h for n,h in q['source_sha256'].items())
    p=json.loads((OUT/'proposal.json').read_text());seed=p['seeds'][0]
    config={**p,'engineering_seed':seed};torch.set_num_threads(1)
    reports={};shared=None;initial_world=None;calls=0;graph_steps=0;worlds=100;fd_calls=[]
    launch=runtime.wp.capture_launch;fd=mujoco.mjd_transitionFD
    def capture(*args,**kwargs):
        nonlocal calls,graph_steps
        calls+=1;graph_steps+=40*worlds;assert graph_steps<=q['physical_graph_world_step_budget']
        return launch(*args,**kwargs)
    def counted(*args,**kwargs):
        fd_calls.append(float(args[2]));assert len(fd_calls)<=q['constructor_FD_budget']
        return fd(*args,**kwargs)
    runtime.wp.capture_launch=capture;mujoco.mjd_transitionFD=counted
    atomic_json(directory/'started.json',dict(proposal_sha256=sha(directory/'proposal.json'),science_selection=False))
    try:
        for arm in p['arms']:
            out=directory/arm;out.mkdir(exist_ok=False);start=time.perf_counter()
            curriculum,raw,history,norm=runtime.make_env(p,arm,seed)
            try:
                assert raw.num_envs==100 and raw._joint_topology['captured_controller_calls']==40
                assert not hasattr(raw,'_complete_buffers')
                world=[x.numpy().copy() for x in (raw.q0,raw.param,raw.k['gains'],raw.k['reference'])]
                if initial_world is None:initial_world=world
                else:equal(initial_world,world)
                np.savez_compressed(out/'initial_world.npz',q=world[0],param=world[1],gains=world[2],reference=world[3])
                model,shared=agent(config,norm,shared);initial=weight_digest(model)
                torch.save(dict(policy=model.policy.state_dict(),optimizer=model.policy.optimizer.state_dict()),out/'initial_policy.pt')
                ledger=Ledger(norm,out);model.learn(total_timesteps=5000,callback=ledger)
                assert model.num_timesteps==5000 and model._n_updates==10 and ledger.saved==[5000]
                assert weight_digest(model)!=initial;check_agent(model,200)
                loaded=PPO.load(out/'step_5000.zip',device='cuda')
                equal(model.policy.state_dict(),loaded.policy.state_dict());equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict())
                restored=VecNormalize.load(out/'step_5000.pkl',norm.venv)
                for name in ('obs_rms','ret_rms'):
                    for field in ('mean','var','count'):equal(getattr(getattr(norm,name),field),getattr(getattr(restored,name),field))
                np.testing.assert_allclose(norm.obs_rms.count,5100.0001,rtol=0,atol=1e-7)
                norm.training=False;seen_stages=set();episodes=[];probes=[];resets=0
                # Accelerate only the engineering curriculum counter, never physical clocks or episode ends.
                curriculum.policy_steps=19900
                for tick in range(q['max_runtime_ticks_per_arm']):
                    before=curriculum.stages.copy()
                    action=np.zeros((100,3 if arm=='M_ref3' else 6),np.float32)
                    obs,reward,done,infos=norm.step(action)
                    assert obs.shape==(100,481) and np.isfinite(obs).all() and np.isfinite(reward).all()
                    changed=curriculum.stages!=before;assert not np.any(changed&~done)
                    if done.any():
                        resets+=int(done.sum())
                        for index in (0,1,2,3,4,5,7):np.testing.assert_array_equal(raw._joint_buffers[index].numpy()[done],0)
                        np.testing.assert_array_equal(history.inputs[done],0);np.testing.assert_array_equal(history.elapsed[done],0)
                        np.testing.assert_array_equal(history.valid[done,:-1],0);np.testing.assert_array_equal(history.valid[done,-1],1)
                        np.testing.assert_array_equal(raw._execution_previous_sensor_steps[done],0)
                        for w in np.flatnonzero(done):
                            info=infos[w];assert info['terminal_observation'].shape==(481,)
                            episodes.append({k:v for k,v in info.items() if k!='terminal_observation'})
                    if changed.any():
                        stage=int(curriculum.stages[changed][0]);seen_stages.add(stage)
                        cache=curriculum.cache[stage]
                        for buffer,key in ((raw.q0,'q0'),(raw.param,'param'),(raw.k['reference'],'reference')):
                            np.testing.assert_array_equal(buffer.numpy()[changed],cache[key][changed])
                        probes.append(dict(tick=tick,stage=stage,worlds=np.flatnonzero(changed).tolist(),done_worlds=np.flatnonzero(done).tolist()))
                        if stage==2:curriculum.policy_steps=99900
                    if seen_stages=={2,3}:break
                assert seen_stages=={2,3},'Both real episode-end transitions must be observed'
                atomic_json(out/'episodes.json',dict(episodes=episodes))
                np.savez_compressed(out/'final_history.npz',frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid,stages=curriculum.stages)
                report=dict(verified=True,policy_samples=5000,epochs=10,Adam_steps=200,real_terminal_resets=resets,
                    curriculum_probes=probes,engineering_counter_accelerated=True,all_episode_physical_design=all(r['physical_safety_passed'] and r['design_joint_passed'] for r in episodes),
                    wall_seconds=time.perf_counter()-start,physics_device=str(raw.data.qpos.device),policy_device=str(model.device),source_topology=raw._joint_topology)
                atomic_json(out/'verification.json',report);reports[arm]=report
            finally:norm.close()
        # Older short-engineering checkpoints exercise the frozen learned evaluation API only.
        for arm in p['arms']:
            old=OUT.parent/'reference_learning_engineering_v1'/arm
            model=PPO.load(old/'step_4000.zip',device='cuda')
            out=directory/f'evaluator_{arm}';out.mkdir(exist_ok=False)
            worlds=len(q['evaluation_cases'])
            result,checked=evaluate(q['evaluation_cases'],arm,out,q['evaluation_graph_budget_per_arm'],model=model,normalization=old/'step_4000.pkl')
            reports[f'evaluator_{arm}']=dict(verified=True,episodes=len(result['runs']),actual_graph_world_steps=result['actual_graph_world_steps'],checked=checked)
        assert all(sha(ROOT/n)==h for n,h in q['source_sha256'].items())
        atomic_json(directory/'completion.json',dict(verified=True,reports=reports,actual_capture_calls=calls,actual_graph_world_steps=graph_steps,constructor_FD_calls=fd_calls,
            engineering_learning_samples=10000,science_evaluations=0,main_training_admitted=False))
    except BaseException as error:
        atomic_json(directory/'failure.json',dict(error=repr(error),completed=list(reports),actual_capture_calls=calls,actual_graph_world_steps=graph_steps,FD_calls=fd_calls,implicit_retry=False))
        raise
    finally:runtime.wp.capture_launch=launch;mujoco.mjd_transitionFD=fd


if __name__=='__main__':run()
