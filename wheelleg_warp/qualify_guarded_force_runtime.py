"""Bounded real100-world curriculum and frozen learned-force evaluator qualification."""
import json
import numpy as np
import mujoco
import torch
import warp as wp
from stable_baselines3 import PPO
import fixed_reference_force as force
import joint_state_guard as guard
import execution_history_evaluation as evaluation
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/guarded_force_main_runtime_v1'


def run():
    q=json.loads((OUT/'proposal.json').read_text());assert not any((OUT/n).exists() for n in ('started.json','completion.json','failure.json'))
    assert all(sha(ROOT/f)==h for f,h in q['source_sha256'].items())
    protocol=json.loads((ROOT/q['bank_protocol']).read_text());reports={};steps=0;worlds=100;fd_calls=[];graph=wp.capture_launch;fd=mujoco.mjd_transitionFD;old=evaluation.make_env
    def capture(*args,**kwargs):
        nonlocal steps
        steps+=40*worlds;assert steps<=q['graph_world_step_budget'];return graph(*args,**kwargs)
    def counted(*args,**kwargs):fd_calls.append(float(args[2]));assert len(fd_calls)<=q['FD_budget'];return fd(*args,**kwargs)
    wp.capture_launch=capture;mujoco.mjd_transitionFD=counted;torch.set_num_threads(1)
    atomic_json(OUT/'started.json',dict(proposal_sha256=sha(OUT/'proposal.json'),learning_samples=0))
    try:
        for arm in ('D3','V6'):
            directory=OUT/arm;directory.mkdir(exist_ok=False)
            curriculum,raw,history,norm=force.make_env(protocol,arm,32031,guarded=True)
            try:
                norm.training=False;obs=norm.reset();assert obs.shape==(100,481) and norm.action_space.shape==(3 if arm=='D3' else 6,)
                assert raw._joint_guard_topology['captured_controller_calls']==40 and not raw._joint_guard_topology['dense_recording'] and not hasattr(raw,'_complete_buffers')
                assert not raw._joint_guard_buffers[0].numpy().any()
                curriculum.policy_steps=19900;observed=set();episodes=[];transitions=[]
                for tick in range(q['max_ticks_per_arm']):
                    previous=curriculum.stages.copy();obs,reward,done,infos=norm.step(np.zeros((100,3 if arm=='D3' else 6),np.float32))
                    assert obs.shape==(100,481) and np.isfinite(obs).all() and np.isfinite(reward).all()
                    changed=curriculum.stages!=previous;assert not (changed&~done).any()
                    if done.any():
                        np.testing.assert_array_equal(raw._joint_guard_buffers[0].numpy()[done],0)
                        np.testing.assert_array_equal(history.inputs[done],0);np.testing.assert_array_equal(history.elapsed[done],0);np.testing.assert_array_equal(history.valid[done,:-1],0);np.testing.assert_array_equal(history.valid[done,-1],1)
                        np.testing.assert_array_equal(raw._execution_previous_sensor_steps[done],0)
                        for w in np.flatnonzero(done):episodes.append({k:v for k,v in infos[w].items() if k!='terminal_observation'})
                    if changed.any():
                        stage=int(curriculum.stages[changed][0]);observed.add(stage);cache=curriculum.cache[stage]
                        for buffer,key in ((raw.q0,'q0'),(raw.param,'param'),(raw.k['reference'],'reference')):np.testing.assert_array_equal(buffer.numpy()[changed],cache[key][changed])
                        transitions.append(dict(tick=tick,stage=stage,worlds=np.flatnonzero(changed).tolist(),done=np.flatnonzero(done).tolist()))
                        if stage==2:curriculum.policy_steps=99900
                    if observed=={2,3}:break
                assert observed=={2,3}
                atomic_json(directory/'episodes.json',dict(episodes=episodes))
                np.savez_compressed(directory/'final_state.npz',q=raw.data.qpos.numpy(),v=raw.data.qvel.numpy(),terminal_q=raw.stopped_q.numpy(),terminal_v=raw.stopped_v.numpy(),frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid,guard_memory=raw._joint_guard_buffers[0].numpy(),guard_last=raw._joint_guard_buffers[1].numpy(),stages=curriculum.stages)
                reports[arm]=dict(verified=True,worlds=100,transitions=transitions,completed_episodes=len(episodes),topology=raw._joint_guard_topology,guard_stats=raw._joint_guard_stats,engineering_counter_accelerated=True,physical_design=all(r['physical_safety_passed'] and r['design_joint_passed'] for r in episodes))
                atomic_json(directory/'verification.json',reports[arm])
            finally:norm.close()
        def guarded(cases,arm,normalization,directory):return guard.instrument(lambda:old(cases,arm,normalization,directory),cases,directory)
        evaluation.make_env=guarded;worlds=len(q['evaluation_cases'])
        for arm in ('D3','V6'):
            directory=OUT/f'evaluator_{arm}';directory.mkdir(exist_ok=False);model=PPO.load(ROOT/q['models'][arm]['model'],device='cuda')
            result,checked=evaluation.evaluate(q['evaluation_cases'],'H1',force.PolicyActor(model,arm),ROOT/q['models'][arm]['normalization'],directory)
            reports[f'evaluator_{arm}']=dict(verified=True,episodes=len(result['runs']),physical=result['physical'],design=result['design'],checked=checked)
        assert all(sha(ROOT/f)==h for f,h in q['source_sha256'].items())
        atomic_json(OUT/'completion.json',dict(verified=True,reports=reports,actual_graph_world_steps=steps,FD_calls=fd_calls,new_learning_samples=0,scientific_evaluations=0,formal5_admitted=False))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),completed=list(reports),actual_graph_world_steps=steps,FD_calls=fd_calls,implicit_retry=False));raise
    finally:evaluation.make_env=old;wp.capture_launch=graph;mujoco.mjd_transitionFD=fd


if __name__=='__main__':run()
