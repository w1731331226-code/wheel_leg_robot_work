"""Single328 queue with complete reused records; no training or automatic replay."""
import fcntl
import json
import os
import time
from pathlib import Path
import mujoco
import numpy as np
import torch
import execution_history_evaluation as evaluation
import continuous_nominal_task_adapter as adapter
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_task_pair_v1'


def verify(p):
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    for r in p['old_archived_baseline']:assert sha(ROOT/r['path'])==r['sha256']


def preflight(p):
    assert not (OUT/'source_admission.json').exists()
    case=p['eval_jobs'][0]['cases'][0]
    reports=[];initial=[]
    for arm in ('old_B0','map_B0'):
        directory=OUT/'source_preflight'/arm;directory.mkdir(parents=True,exist_ok=False)
        make=evaluation.make_env
        if arm=='map_B0':evaluation.make_env=adapter.make_map
        try:raw,route,history,norm,env=evaluation.make_env([case],'B0',None,directory)
        finally:evaluation.make_env=make
        try:
            env.reset();initial.append([raw.data.qpos.numpy().copy(),raw.data.qvel.numpy().copy(),
                raw.state.numpy().copy(),raw.k['state'].numpy().copy(),raw.k['gains'].numpy().copy(),
                raw.k['feed'].numpy().copy(),raw.k['reference'].numpy().copy(),raw.param.numpy().copy()])
            assert not raw.data.time.numpy().any()
            assert raw._execution_topology['control_steps']==40
            report=dict(arm=arm,physical_steps=0,controller_graph_launched=False,
                        collector=raw._execution_topology,parking=raw._parking_contract)
            if arm=='map_B0':
                report['map']=raw._continuous_map_topology
                np.testing.assert_array_equal(raw._continuous_map_buffers[0].numpy(),0.)
                np.testing.assert_array_equal(raw._continuous_map_buffers[1].numpy(),0.)
            reports.append(report)
        finally:env.close()
    for a,b in zip(*initial):np.testing.assert_array_equal(a,b)
    sources={**p['source_sha256'],
        'wheelleg_warp/continuous_nominal_task_adapter.py':sha(adapter.__file__),
        'wheelleg_warp/run_continuous_nominal_task_pair.py':sha(__file__)}
    atomic_json(OUT/'source_admission.json',dict(verified=True,proposal_sha256=sha(OUT/'proposal.json'),
        source_sha256=sources,reports=reports,same_reset_states_bank_parameters=True,
        evaluations_admitted=328,preflight_physical_steps=0,new_training_samples=0,
        limits='Capture/owner/reset/preexisting recorder interface preflight only. Complete mapper traces checked during actual328queue,not independent extra sourceepisodes.'))
    print('ADMITTED284 capture/reset/source;328taskqueue;0preflightphysics',flush=True)
    return sources


def run():
    assert not any((OUT/n).exists() for n in ('source_admission.json','progress.json','completion.json','interruption.json'))
    torch.set_num_threads(1)
    p=json.loads((OUT/'proposal.json').read_text());verify(p)
    assert p['evaluations']==328 and sum(len(j['cases']) for j in p['eval_jobs'])==164
    fd_calls=[];fd=mujoco.mjd_transitionFD
    def counted_fd(*args,**kwargs):
        fd_calls.append(float(args[2]));return fd(*args,**kwargs)
    mujoco.mjd_transitionFD=counted_fd
    reserved=finished=0;records=[];current=None;start=time.monotonic()
    atomic_json(OUT/'worker.json',dict(pid=os.getpid(),runner_sha256=sha(__file__),proposal_sha256=sha(OUT/'proposal.json'),
                                    exclusive_project_write_lock=True))
    try:
        sources=preflight(p)
        atomic_json(OUT/'preflight_cost.json',dict(baseline_constructor_transitionFD_calls=len(fd_calls),physics_steps=0))
        for arm in p['arms']:
            for job in p['eval_jobs']:
                verify(p);assert all(sha(ROOT/n)==h for n,h in sources.items())
                current=dict(arm=arm,panel=job['panel'],batch=job['batch'])
                directory=OUT/'evaluations'/arm/job['panel']/f'batch_{job["batch"]}'
                directory.mkdir(parents=True,exist_ok=False)
                reserved+=len(job['cases'])
                atomic_json(OUT/'progress.json',dict(status='evaluating',current=current,reserved=reserved,
                    completed=finished,records=records,baseline_constructor_transitionFD_calls=len(fd_calls)))
                make=evaluation.make_env
                if arm=='map_B0':evaluation.make_env=adapter.make_map
                try:result,checked=evaluation.evaluate(job['cases'],'B0',None,None,directory)
                except BaseException:
                    if arm=='map_B0' and adapter.LAST_RAW is not None:
                        adapter.preserve(adapter.LAST_RAW,directory)
                    raise
                finally:evaluation.make_env=make
                finished+=len(job['cases'])
                records.append(dict(**current,indices=job['indices'],path=str((directory/'result.json').relative_to(OUT)),
                    sha256=sha(directory/'result.json'),summary=result['summary'],**checked))
                atomic_json(OUT/'ledger.json',dict(reserved=reserved,completed=finished,records=records,
                    baseline_constructor_transitionFD_calls=len(fd_calls)))
                print('TASKPAIR',arm,job['panel'],job['batch'],finished,'/328',flush=True)
        assert finished==reserved==328
        verify(p);assert all(sha(ROOT/n)==h for n,h in sources.items())
        atomic_json(OUT/'completion.json',dict(verified=True,evaluations=328,records=records,
            admission_sha256=sha(OUT/'source_admission.json'),baseline_constructor_transitionFD_calls=len(fd_calls),
            new_experimental_transitionFD_calls=0,optimization_calls=0,new_training_samples=0,
            queue_wall_seconds=time.monotonic()-start,production_admitted=False,
            limits='Actual328firstepisode outcomes andcomplete evidence. Archivedbaseline reproduction,engineeringgate/fullraw independent review pending. No PPO/novelmethod/fullCPU-GPUadmission.'))
        atomic_json(OUT/'progress.json',dict(status='complete',completed=328,reserved=328,
            baseline_constructor_transitionFD_calls=len(fd_calls)))
        print('DONE284328taskpair;baseline/task/rawgate reviewpending',flush=True)
    except BaseException as error:
        atomic_json(OUT/'interruption.json',dict(error=repr(error),current=current,reserved=reserved,
            completed=finished,records=records,baseline_constructor_transitionFD_calls=len(fd_calls),
            implicit_resume=False))
        raise
    finally:mujoco.mjd_transitionFD=fd


if __name__=='__main__':
    with (ROOT/'.git/project-write.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run()
