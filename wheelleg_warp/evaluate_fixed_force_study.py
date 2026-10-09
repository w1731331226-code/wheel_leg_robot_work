"""Frozen guarded classics/final200k science queue; refuses incomplete admission."""
import json
import time
import torch
import mujoco
import warp as wp
from stable_baselines3 import PPO
import execution_history_evaluation as evaluation
import joint_state_guard as guard
from fixed_reference_force import PolicyActor
from train_fixed_force_study import OUT,verify
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run(kind):
    assert kind in ('baseline','models')
    p=json.loads((OUT/'proposal.json').read_text());a=json.loads((OUT/'evaluation_source.json').read_text())
    assert a['proposal_sha256']==sha(OUT/'proposal.json') and all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    conditions=p['classical_conditions']
    if kind=='models':
        verify();done=json.loads((OUT/'training/completion.json').read_text());assert done['verified'] and done['policy_samples']==1200000 and len(done['reports'])==6
        model_admission=json.loads((OUT/'model_evaluation_admission.json').read_text())
        assert model_admission['training_completion_sha256']==sha(OUT/'training/completion.json')
        assert all(sha(ROOT/f)==h for f,h in model_admission['model_sha256'].items())
        conditions=[dict(label=f'{arm}_{seed}',arm=arm,seed=seed) for seed in p['seeds'] for arm in p['arms']]
    directory=OUT/kind;directory.mkdir(exist_ok=False);torch.set_num_threads(1);records=[];steps=0;worlds=0;fd_calls=[];current=None;start=time.perf_counter()
    graph=wp.capture_launch;fd=mujoco.mjd_transitionFD;old=evaluation.make_env
    def capture(*args,**kwargs):
        nonlocal steps
        steps+=40*worlds;assert steps<=a['graph_world_step_budget'];return graph(*args,**kwargs)
    def counted(*args,**kwargs):fd_calls.append(float(args[2]));assert len(fd_calls)<=a['FD_budget'];return fd(*args,**kwargs)
    def make_env(cases,arm,normalization,out):return guard.instrument(lambda:old(cases,arm,normalization,out),cases,out)
    evaluation.make_env=make_env;wp.capture_launch=capture;mujoco.mjd_transitionFD=counted
    atomic_json(directory/'started.json',dict(source_sha256=sha(OUT/'evaluation_source.json'),proposal_sha256=sha(OUT/'proposal.json')))
    try:
        for condition in conditions:
            model=None;normalization=None
            if kind=='models':
                path=OUT/'training'/condition['label'];r=json.loads((path/'verification.json').read_text());assert r['verified'] and r['policy_samples']==200000
                model=PolicyActor(PPO.load(path/'step_200000.zip',device='cuda'),condition['arm']);normalization=path/'step_200000.pkl'
            for job in p['eval_jobs']:
                assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
                if kind=='models':assert all(sha(ROOT/f)==h for f,h in model_admission['model_sha256'].items())
                current=dict(condition=condition['label'],panel=job['panel'],batch=job['batch'])
                dest=directory/condition['label']/job['panel']/str(job['batch']);dest.mkdir(parents=True,exist_ok=False);worlds=len(job['cases']);before=steps
                result,checked=evaluation.evaluate(job['cases'],'H1' if model else condition['arm'],model,normalization,dest,a['yaw_config'] if model is None else None)
                entry=dict(**current,path=str((dest/'result.json').relative_to(directory)),sha256=sha(dest/'result.json'),indices=job['indices'],episodes=worlds,
                    physical=result['physical'],design=result['design'],success=result['summary']['success_count'],actual_graph_world_steps=steps-before,first_episode_world_steps=sum(r['physical_steps'] for r in result['runs']),checked=checked)
                records.append(entry);atomic_json(directory/'progress.json',dict(records=records,episodes=sum(r['episodes'] for r in records),actual_graph_world_steps=steps))
                print('EVAL335',condition['label'],job['panel'],job['batch'],entry['success'],'/',worlds,flush=True)
        expected=492 if kind=='baseline' else 984;assert sum(r['episodes'] for r in records)==expected
        atomic_json(directory/'completion.json',dict(verified=True,records=records,episodes=expected,actual_graph_world_steps=steps,first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),FD_calls=fd_calls,wall_seconds=time.perf_counter()-start,new_learning_samples=0,formal5_admitted=False))
    except BaseException as error:
        atomic_json(directory/'failure.json',dict(error=repr(error),current=current,records=records,actual_graph_world_steps=steps,FD_calls=fd_calls,implicit_retry=False));raise
    finally:evaluation.make_env=old;wp.capture_launch=graph;mujoco.mjd_transitionFD=fd


if __name__=='__main__':
    import sys
    run(sys.argv[1])
