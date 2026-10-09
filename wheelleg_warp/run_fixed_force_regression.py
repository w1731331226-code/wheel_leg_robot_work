"""One registered33-episode fixed-reference force-space dynamic regression."""
import json
import time
import numpy as np
import torch
import mujoco
import warp as wp
from test_reference_learning_policy import NoStep
from reference_learning_engineering import agent
from fixed_reference_force import PriorActor,self_check
from execution_history_evaluation import evaluate
from review_yaw_sector import ROOT,sha
from smoke_reward_training import weight_digest
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_dynamic_regression_v1'


def run():
    p=json.loads((OUT/'proposal.json').read_text())
    assert not any((OUT/f).exists() for f in ('started.json','completion.json','failure.json'))
    assert all(sha(ROOT/f)==h for f,h in p['source_sha256'].items())
    self_check();torch.set_num_threads(1);shared=None;models={}
    for arm,n in (('D3',3),('V6',6)):
        model,shared=agent(p,NoStep(n),shared);models[arm]=model
        assert model.num_timesteps==model._n_updates==0 and not model.policy.optimizer.state_dict()['state']
        assert all(torch.count_nonzero(v)==0 for v in model.policy.action_net.parameters())
        torch.save(dict(policy=model.policy.state_dict(),optimizer=model.policy.optimizer.state_dict()),OUT/f'initial_{arm}.pt')
    reports=[];steps=0;calls=[];current=None;world=None;fd=mujoco.mjd_transitionFD;graph=wp.capture_launch;start=time.perf_counter()
    def counted(*args,**kwargs):calls.append(float(args[2]));assert len(calls)<=p['FD_budget'];return fd(*args,**kwargs)
    def capture(*args,**kwargs):
        nonlocal steps
        steps+=40*len(p['cases']);assert steps<=p['graph_world_step_budget'];return graph(*args,**kwargs)
    mujoco.mjd_transitionFD=counted;wp.capture_launch=capture
    atomic_json(OUT/'started.json',dict(proposal_sha256=sha(OUT/'proposal.json'),learning_samples=0))
    try:
        for label in ('zero','D3','V6'):
            current=label;directory=OUT/label;directory.mkdir(exist_ok=False);torch.manual_seed(p['action_seed'])
            torch.save(dict(RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),directory/'initial_RNG.pt')
            actor=None if label=='zero' else PriorActor(models[label],label)
            initial=None if actor is None else weight_digest(models[label]);before=steps
            result,checked=evaluate(p['cases'],'B0' if label=='zero' else 'H1',actor,None,directory)
            if actor is not None:
                assert weight_digest(models[label])==initial and models[label].num_timesteps==models[label]._n_updates==0 and not models[label].policy.optimizer.state_dict()['state']
            # Dense pre-state is the common physical initial condition.
            initial_rows=[]
            for row in result['runs']:
                with np.load(directory/row['complete_trace']['path']) as z:initial_rows.append(z['pre'][0].copy())
                with np.load(directory/row['role_trace']['path']) as z:
                    t=z['trace'];np.testing.assert_array_equal(t[:,2],row['scenario']['stand_height_m']);np.testing.assert_array_equal(t[:,3],t[:,2])
                    np.testing.assert_allclose(t[:,7:9].mean(axis=1),t[:,2],rtol=0,atol=1e-12)
                if label=='D3':
                    with np.load(directory/row['actor_trace']['path']) as z:commands=z['trace'][:,962:]
                    np.testing.assert_array_equal(commands[:,::2]+commands[:,1::2],0)
                    with np.load(directory/row['parking_trace']['path']) as z:
                        t=z['trace'];np.testing.assert_array_equal(t[:,11:17:2]+t[:,12:17:2],0)
                        np.testing.assert_array_equal(t[:,23:29:2]+t[:,24:29:2],0)
            initial_rows=np.stack(initial_rows)
            if world is None:world=initial_rows
            else:np.testing.assert_array_equal(initial_rows[:,:33],world[:,:33])
            entry=dict(condition=label,episodes=len(p['cases']),physical=sum(r['physical_safety_passed'] for r in result['runs']),
                design=sum(r['design_joint_passed'] for r in result['runs']),success=sum(r['success'] for r in result['runs']),
                actual_graph_world_steps=steps-before,first_episode_world_steps=sum(r['physical_steps'] for r in result['runs']),
                result_sha256=sha(directory/'result.json'),checked=checked,nominal_reference_height_unchanged=True,untrained_prior=True)
            reports.append(entry);atomic_json(OUT/'progress.json',dict(records=reports,actual_graph_world_steps=steps))
        assert sum(r['episodes'] for r in reports)==33
        gate=all(r['physical']==r['design']==11 for r in reports)
        atomic_json(OUT/'completion.json',dict(verified=True,engineering_gate_passed=gate,records=reports,actual_graph_world_steps=steps,
            first_episode_world_steps=sum(r['first_episode_world_steps'] for r in reports),constructor_FD_calls=calls,wall_seconds=time.perf_counter()-start,
            learning_samples=0,formal5_admitted=False,new_learning_admitted=False,limits='fresh zeroheadGaussian prior,raw481 actorinputs withno normalizer;zerohead means distribution doesnotdepend onobs. Singleaction seed andseen engineeringcases,notlearnedadvantage/exactexplorationequivalence/dynamicinvariancecertificate'))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),condition=current,records=reports,actual_graph_world_steps=steps,FD_calls=calls,implicit_retry=False));raise
    finally:mujoco.mjd_transitionFD=fd;wp.capture_launch=graph


if __name__=='__main__':run()
