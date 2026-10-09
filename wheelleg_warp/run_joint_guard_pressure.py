"""One frozen independent-development pressure queue; no learning or tuning."""
import json
import time
from pathlib import Path
import numpy as np
import torch
import mujoco
import warp as wp
import execution_history_evaluation as evaluation
import joint_state_guard as guard
from fixed_reference_force import PriorActor
from reference_learning_engineering import agent
from test_reference_learning_policy import NoStep
from smoke_reward_training import weight_digest
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/joint_guard_pressure_v1'


def run():
    p=json.loads((OUT/'proposal.json').read_text())
    assert not any((OUT/n).exists() for n in ('started.json','completion.json','failure.json'))
    assert all(sha(ROOT/f)==h for f,h in p['source_sha256'].items())
    assert len(p['cases'])==12 and p['episodes']==72
    torch.set_num_threads(1);models={};shared=None
    for arm,n in (('D3',3),('V6',6)):
        model,shared=agent(p,NoStep(n),shared);models[arm]=model
        torch.save(dict(policy=model.policy.state_dict(),optimizer=model.policy.optimizer.state_dict()),OUT/f'initial_{arm}.pt')
    original=evaluation.make_env;fd=mujoco.mjd_transitionFD;graph=wp.capture_launch;steps=0;fd_calls=[];records=[];current=None;world=None;start=time.perf_counter()
    def counted(*args,**kwargs):
        fd_calls.append(float(args[2]));assert len(fd_calls)<=p['FD_budget'];return fd(*args,**kwargs)
    def capture(*args,**kwargs):
        nonlocal steps
        steps+=40*len(p['cases']);assert steps<=p['graph_world_step_budget'];return graph(*args,**kwargs)
    def guarded(cases,arm,normalization,directory):return guard.instrument(lambda:original(cases,arm,normalization,directory),cases,directory)
    mujoco.mjd_transitionFD=counted;wp.capture_launch=capture
    atomic_json(OUT/'started.json',dict(proposal_sha256=sha(OUT/'proposal.json'),learning_samples=0))
    try:
        for mode in ('original','guarded'):
            evaluation.make_env=original if mode=='original' else guarded
            for arm in ('zero','D3','V6'):
                assert all(sha(ROOT/f)==h for f,h in p['source_sha256'].items())
                current=f'{mode}_{arm}';directory=OUT/current;directory.mkdir(exist_ok=False)
                torch.manual_seed(p['action_seed']);torch.save(dict(RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),directory/'initial_RNG.pt')
                actor=None if arm=='zero' else PriorActor(models[arm],arm)
                digest=None if actor is None else weight_digest(models[arm]);before=steps
                result,checked=evaluation.evaluate(p['cases'],'B0' if arm=='zero' else 'H1',actor,None,directory)
                if actor is not None:
                    assert digest==weight_digest(models[arm]) and models[arm].num_timesteps==models[arm]._n_updates==0 and not models[arm].policy.optimizer.state_dict()['state']
                initial=[];infeasible=0;pred_max=None;error_max=None
                for row in result['runs']:
                    with np.load(directory/row['complete_trace']['path']) as z:initial.append(z['pre'][0,:33].copy())
                    if mode=='guarded':
                        with np.load(directory/row['joint_guard_trace']['path']) as z:g=z['trace']
                        assert len(g)==row['physical_steps'] and np.isfinite(g).all()
                        infeasible+=int((g[:,28:30]==0).any(axis=1).sum())
                        pred_max=max(float(g[:,58].max()),-np.inf if pred_max is None else pred_max)
                        error_max=max(float(abs(g[:,38:42]).max()),0 if error_max is None else error_max)
                initial=np.stack(initial)
                if world is None:world=initial
                else:np.testing.assert_array_equal(initial,world)
                record=dict(condition=current,mode=mode,arm=arm,episodes=12,physical=result['physical'],design=result['design'],success=result['summary']['success_count'],
                    actual_graph_world_steps=steps-before,first_episode_world_steps=sum(r['physical_steps'] for r in result['runs']),
                    model_prediction_infeasible_substeps=infeasible,predicted_CBF_max_violation=pred_max,model_error_max_rad_s2=error_max,
                    result_sha256=sha(directory/'result.json'),checked=checked)
                records.append(record);atomic_json(OUT/'progress.json',dict(records=records,actual_graph_world_steps=steps))
                print('PRESSURE329',current,record['physical'],record['design'],record['success'],flush=True)
        assert sum(r['episodes'] for r in records)==72
        qualified=all(r['physical']==r['design']==12 and r['model_prediction_infeasible_substeps']==0 for r in records if r['mode']=='guarded')
        atomic_json(OUT/'completion.json',dict(verified=True,pressure_gate_passed=qualified,records=records,actual_graph_world_steps=steps,
            first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),FD_calls=fd_calls,wall_seconds=time.perf_counter()-start,
            learning_samples=0,new_learning_admitted=False,formal5_admitted=False))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),condition=current,records=records,actual_graph_world_steps=steps,FD_calls=fd_calls,implicit_retry=False));raise
    finally:evaluation.make_env=original;mujoco.mjd_transitionFD=fd;wp.capture_launch=graph


if __name__=='__main__':run()
