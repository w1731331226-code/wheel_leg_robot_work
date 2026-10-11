"""One registered 18-episode classical route-reference diagnostic."""
import fcntl
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import mujoco
import torch
import warp as wp
import execution_history_evaluation as evaluation
import joint_state_guard as guard
from probe_route_feedback import actions
from run_reflection_retention import receive_initial
from run_wheel_interference import runtime
from train_fixed_force_study import OUT as STUDY
from review_fixed_force_evaluation import chain_minima, physical_metrics
from review_yaw_sector import ROOT, sha
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
from dashboard.live_env import atomic_json

OUT = STUDY.parent / 'classical_route_reference_diagnostic_v1'


def shifted(obs, candidate, arm, target):
    if target not in (0., .05, -.05):
        raise ValueError('Unregistered route reference')
    packet = np.array(obs, dtype=np.float64, copy=True)
    if packet.ndim != 2 or packet.shape[1] != 39:
        raise ValueError('Expected raw39 packet')
    packet[:, 38] -= target
    command, reference = actions(packet, candidate, arm)
    log = np.column_stack((obs, np.full(len(packet), target), packet[:, 38], reference, command[:, 2]))
    return command, reference, log


def self_check():
    candidate = dict(name='B1_3', kp=.4, kd=.3, roll_gain=0)
    obs = np.random.default_rng(356).normal(size=(80, 39)).astype(np.float32)
    obs[:, 9] = np.tile([0., .05, -.05, .7, -.7], 16)
    obs[:4, 38] = [1e6, -1e6, 0., .05]
    before = obs.copy()
    original = actions(obs, candidate, 1)
    zero = shifted(obs, candidate, 1, 0.)
    for left, right in zip(original, zero[:2]):
        np.testing.assert_array_equal(left, right)
    mirrored = obs.copy(); mirrored[:, [0, 2, 5, 38]] *= -1
    for target in (0., .05, -.05):
        command, reference, log = shifted(obs, candidate, 1, target)
        other, opposite, _ = shifted(mirrored, candidate, 1, -target)
        np.testing.assert_array_equal(other, -command)
        np.testing.assert_array_equal(opposite, -reference)
        np.testing.assert_array_equal(obs, before)
        np.testing.assert_array_equal(log[:, :39], before)
        np.testing.assert_array_equal(log[:, 40], before[:, 38].astype(float) - target)
        assert not reference[abs(obs[:, 9].astype(np.float64)) <= .05].any()
        assert abs(reference).max() <= np.radians(3) and abs(command).max() <= 1
    for bad in (.051, float('nan')):
        try: shifted(obs, candidate, 1, bad)
        except ValueError: pass
        else: raise AssertionError('Unregistered offset accepted')
    cache = {}; snap = dict(q=np.zeros((1, 17)), warm=np.zeros((1, 16)))
    receive_initial(cache, 1, snap); receive_initial(cache, 2, {k:v+1 for k,v in snap.items()})
    try: receive_initial(cache, 1, {k:v+1 for k,v in snap.items()})
    except AssertionError: pass
    else: raise AssertionError('Changed same-case reset accepted')
    print('PASS356 zero equivalence, raw preservation, mirrored offsets, stopped reference, caps and case resets; 0 physics', flush=True)


def inputs():
    p = json.loads((OUT/'proposal.json').read_text())
    direction = json.loads((OUT.parent/'round355_direction_review.json').read_text())
    assert direction['diagnostic_proposal_sha256'] == sha(OUT/'proposal.json')
    assert p['evaluations'] == len(p['conditions']) == 18 and len(p['cases']) == 6
    assert p['new_learning_samples'] == p['model_forward_row_budget'] == 0
    assert p['environment_arm'] == 'B1-route' and not p['formal5_admitted']
    assert p['classical_config'] == dict(candidate=dict(name='B1_3', kp=.4, kd=.3, roll_gain=0), arm=1)
    expected = [dict(label=f'{c["seed"]}_{a["label"]}', case=c, controller=a['label'], target_y_m=a['target_y_m'])
                for c in p['cases'] for a in p['controllers']]
    assert expected == p['conditions']
    assert [c['target_y_m'] for c in p['controllers']] == [0., .05, -.05]
    for mapping in (p['source_sha256'], p['evidence_sha256']):
        assert all(sha(ROOT/f) == h for f,h in mapping.items())
    return p


def freeze():
    p = inputs(); assert not (OUT/'source_admission.json').exists(); self_check()
    sources = dict(p['source_sha256'])
    for file in [Path(__file__), ROOT/'wheelleg_warp/run_wheel_interference.py', ROOT/'wheelleg_warp/review_fixed_force_evaluation.py',
                 ROOT/'wheelleg_warp/run_frozen_reflection.py', ROOT/'wheelleg_warp/audit_policy_reflection.py']:
        sources[str(file.resolve().relative_to(ROOT))] = sha(file)
    atomic_json(OUT/'source_admission.json', dict(round=356, verified=True, proposal_sha256=sha(OUT/'proposal.json'),
        source_sha256=sources, runtime=runtime(), operator_unit_passed=True, new_physics_steps=0,
        new_model_forward_rows=0, new_learning_samples=0, formal5_admitted=False))


def verify():
    p = inputs(); a = json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256'] == sha(OUT/'proposal.json')
    assert all(sha(ROOT/f) == h for f,h in a['source_sha256'].items()) and a['runtime'] == runtime()
    if shutil.disk_usage(ROOT).free < 2*1024**3:
        raise RuntimeError('Need 2GiB free for diagnostic archives')
    return p, a


def run():
    with (ROOT/'.git/project-write.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); p, a = verify()
        assert not any((OUT/f).exists() for f in ('started.json', 'completion.json', 'failure.json'))
        torch.set_num_threads(1); records=[]; steps=0; action_rows=0; fd_calls=[]; holder={}; initial={}; current=None
        start=time.perf_counter(); old=evaluation.make_env; old_actions=evaluation.actions; graph=wp.capture_launch; fd=mujoco.mjd_transitionFD
        def capture(*args, **kwargs):
            nonlocal steps
            assert steps+40 <= p['actual_graph_world_step_budget']; steps+=40
            return graph(*args, **kwargs)
        def counted(*args, **kwargs):
            assert len(fd_calls) < p['constructor_FD_budget']; fd_calls.append(float(args[2]))
            return fd(*args, **kwargs)
        def command(obs, candidate, arm):
            nonlocal action_rows
            np.testing.assert_array_equal(obs, holder['history'].frames[:, -1])
            assert len(obs)==1 and action_rows+len(obs) <= p['classical_action_row_budget']
            action_rows+=len(obs)
            value, reference, log=shifted(obs, candidate, arm, current['target_y_m'])
            holder['route_logs'].append(log)
            return value, reference
        def factory(cases, arm, normalization, directory):
            assert len(cases)==1 and arm=='B1-route' and normalization is None
            def build():
                result=old(cases, arm, normalization, directory); holder.update(raw=result[0], history=result[2]); return result
            result=guard.instrument(build, cases, directory); raw, _, history, _, env=result
            holder.update(guard_chunks=[], route_logs=[])
            reset, async_, wait=env.reset, raw.step_async, raw.step_wait
            def reset_all():
                value=reset()
                snap=dict(q=raw.data.qpos.numpy(), v=raw.data.qvel.numpy(), ctrl=raw.data.ctrl.numpy(), time=raw.data.time.numpy(),
                    warm=raw.data.qacc_warmstart.numpy(), sensors=raw.data.sensordata.numpy(), param=raw.param.numpy(), reference=raw.k['reference'].numpy(),
                    controller_memory=raw.k['state'].numpy(), guard_memory=raw._joint_guard_buffers[0].numpy(), frames=history.frames.copy(),
                    inputs=history.inputs.copy(), elapsed=history.elapsed.copy(), valid=history.valid.copy(), geom_xpos=raw.data.geom_xpos.numpy(), geom_xmat=raw.data.geom_xmat.numpy())
                receive_initial(initial, cases[0]['seed'], snap); np.savez_compressed(directory/'initial.npz', **snap)
                return value
            def step_async(value):
                assert raw.state.numpy()[0,0]+40 <= p['per_job_first_episode_steps_cap']
                return async_(value)
            def step_wait():
                value=wait(); g=raw._joint_guard_buffers[1].numpy()[:,0]; holder['guard_chunks'].append(g[g[:,0]==1].copy())
                t=raw._complete_buffers[0].numpy()[:,0]; keep=t[:,0]==1
                pre=raw._complete_buffers[4].numpy()[:,0][keep]; post=raw._complete_buffers[5].numpy()[:,0][keep]
                if len(post):
                    lengths,_=chain_minima(post[:,:17], raw.cpu); margin,actual,issued=physical_metrics(pre,post,raw.cpu)
                    assert min(lengths)>=HEIGHT_115_GEOMETRIC_MIN and margin>=0 and max(actual,issued)<=1e-6 and t[keep,24].min()>=0, 'Recorded physical/design violation'
                return value
            env.reset,raw.step_async,raw.step_wait=reset_all,step_async,step_wait
            return result
        def preserve(directory):
            if holder.get('route_logs'):
                np.savez_compressed(directory/'route_requests.npz', trace=np.concatenate(holder['route_logs']), layout=np.array(['raw39','target_y','route_error','reference_yaw','wheel_request']))
            if 'raw' in holder:
                raw=holder['raw']; h=holder['history']; extra={}
                if hasattr(raw,'_joint_guard_buffers'): extra=dict(guard_memory=raw._joint_guard_buffers[0].numpy(),guard_last=raw._joint_guard_buffers[1].numpy())
                np.savez_compressed(directory/'terminal.npz', current_q=raw.data.qpos.numpy(),current_v=raw.data.qvel.numpy(),terminal_q=raw.stopped_q.numpy(),terminal_v=raw.stopped_v.numpy(),
                    state=raw.state.numpy(),controller_memory=raw.k['state'].numpy(),frames=h.frames,inputs=h.inputs,**extra)
                if holder.get('guard_chunks'): np.savez_compressed(directory/'guard_prefix.npz',trace=np.concatenate(holder['guard_chunks']))
        evaluation.make_env=factory; evaluation.actions=command; wp.capture_launch=capture; mujoco.mjd_transitionFD=counted
        atomic_json(OUT/'started.json',dict(pid=os.getpid(),proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'source_admission.json'),exclusive_project_lock=True))
        directory=OUT
        try:
            for current in p['conditions']:
                verify(); holder.clear(); directory=OUT/current['label']; directory.mkdir(exist_ok=False)
                before=steps; before_actions=action_rows
                result,checked=evaluation.evaluate([current['case']], 'B1-route', None, None, directory, p['classical_config'])
                preserve(directory); row=result['runs'][0]
                with np.load(directory/row['actor_trace']['path']) as z: actor=z['trace']
                logs=np.concatenate(holder['route_logs']); submitted=actor[:,962:]
                np.testing.assert_array_equal(logs[:,:39], actor[:,351:390])
                np.testing.assert_array_equal(submitted[:,:4],0)
                np.testing.assert_array_equal(submitted[:,4],logs[:,42]); np.testing.assert_array_equal(submitted[:,5],-logs[:,42])
                assert action_rows-before_actions==len(actor)==len(logs) and result['physical']==result['design']==1
                records.append(dict(condition=current['label'],case=current['case']['seed'],controller=current['controller'],target_y_m=current['target_y_m'],
                    path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),success=row['success'],physical=result['physical'],design=result['design'],
                    actual_graph_world_steps=steps-before,classical_action_rows=action_rows-before_actions,first_episode_world_steps=row['physical_steps'],checked=checked,
                    initial_sha256=sha(directory/'initial.npz'),terminal_sha256=sha(directory/'terminal.npz'),route_requests_sha256=sha(directory/'route_requests.npz')))
                atomic_json(OUT/'progress.json',dict(records=records,evaluations=len(records),actual_graph_world_steps=steps,classical_action_rows=action_rows,FD_calls=fd_calls))
                print('ROUTE356',current['label'],'task',row['success'],'yaw',row['peak_deg'][2],flush=True)
            verify(); assert len(records)==18 and action_rows==sum(r['classical_action_rows'] for r in records)
            atomic_json(OUT/'completion.json',dict(round=356,verified=True,records=records,evaluations=18,proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'source_admission.json'),
                actual_graph_world_steps=steps,first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),classical_action_rows=action_rows,FD_calls=fd_calls,
                wall_seconds=time.perf_counter()-start,new_model_forward_rows=0,new_learning_samples=0,formal5_admitted=False))
        except BaseException as error:
            preservation_error=None
            try: preserve(directory)
            except BaseException as saving_error: preservation_error=repr(saving_error)
            atomic_json(OUT/'failure.json',dict(error=repr(error),preservation_error=preservation_error,condition=current,records=records,actual_graph_world_steps=steps,
                classical_action_rows_reserved=action_rows,FD_calls=fd_calls,partial_raw_available='raw' in holder,implicit_retry=False))
            raise
        finally:
            evaluation.make_env=old; evaluation.actions=old_actions; wp.capture_launch=graph; mujoco.mjd_transitionFD=fd


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['self-check','freeze','run']); args=parser.parse_args()
    {'self-check':self_check,'freeze':freeze,'run':run}[args.command]()
