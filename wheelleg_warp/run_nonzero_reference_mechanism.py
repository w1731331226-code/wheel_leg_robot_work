"""One prospective200-episode fixed-feedback queue; no PPO or gain search."""
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from joint_reference_classical import dispatch, unit
from run_joint_reference_zero_pair import evaluate
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/nonzero_reference_mechanism_v1'


def admit():
    assert not (OUT/'runner_admission.json').exists()
    p=json.loads((OUT/'proposal.json').read_text());unit(p['yaw_config'])
    x=np.random.default_rng(30631).normal(size=(128,481));x[:,360]=.7
    yaw=dispatch(x,'joint_yaw',p['yaw_config']);old=dispatch(x,'old_B1_route',p['yaw_config'])
    np.testing.assert_array_equal(yaw[:,:2],0)
    np.testing.assert_array_equal(old[:,4],yaw[:,2]);np.testing.assert_array_equal(old[:,5],-yaw[:,2])
    for arm in p['arms']:
        value=dispatch(x,arm,p['yaw_config']);assert np.isfinite(value).all() and np.max(abs(value))<=1
    center=dispatch(x,'joint_center',p['yaw_config']);lower=dispatch(x,'joint_lower',p['yaw_config'])
    np.testing.assert_array_equal(center[:,2],yaw[:,2]);np.testing.assert_array_equal(lower[:,2],yaw[:,2])
    np.testing.assert_array_equal(lower[:,1],center[:,1]);np.testing.assert_array_equal(lower[:,0],-abs(lower[:,1]))
    manifest=json.loads((ROOT/'wheelleg_warp/results/paper_recovery_20261004/execution_input_history_v1/study_contract.json').read_text())['source_sha256']
    assert all(sha(ROOT/n)==h for n,h in manifest.items())
    for name in ('run_nonzero_reference_mechanism.py','run_joint_reference_zero_pair.py','joint_reference_classical.py','joint_reference_adapter.py','joint_reference_prepare.py','joint_reference_control.py'):
        manifest['wheelleg_warp/'+name]=sha(ROOT/'wheelleg_warp'/name)
    atomic_json(OUT/'runner_admission.json',dict(verified=True,round=306,proposal_sha256=sha(OUT/'proposal.json'),
        source_sha256=manifest,synthetic_dispatch_rows=128,same_yaw_request_forall_nonzero_arms=True,
        public481_only=True,total_episodes=200,world_step_budget=p['physical_graph_world_step_budget'],
        FD_budget=p['constructor_FD_budget'],model_updates=0,
        limits='Reused validated10episode evaluator;onlybatch size/actiondispatch generalized. Fullnonzero phases mayfail;preserveandstop,notPPOadmission.'))
    print('ADMITTED306 dispatch/source200 fixedqueue;0physics preflight',flush=True)


def run():
    assert not any((OUT/n).exists() for n in ('started.json','completion.json','failure.json'))
    p=json.loads((OUT/'proposal.json').read_text());a=json.loads((OUT/'runner_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json')
    results=[];steps=0;fd_calls=[];fd=mujoco.mjd_transitionFD;current=None;start=time.perf_counter()
    def counted(*args,**kwargs):
        fd_calls.append(float(args[2]));assert len(fd_calls)<=p['constructor_FD_budget']
        return fd(*args,**kwargs)
    mujoco.mjd_transitionFD=counted
    atomic_json(OUT/'started.json',dict(admission_sha256=sha(OUT/'runner_admission.json'),episodes=200))
    try:
        for arm in p['arms']:
            for job in p['jobs']:
                assert all(sha(ROOT/n)==h for n,h in a['source_sha256'].items())
                current=dict(arm=arm,batch=job['batch']);directory=OUT/arm/f'batch_{job["batch"]}'
                directory.mkdir(parents=True,exist_ok=False)
                result,checked=evaluate(job['cases'],arm,directory,p['physical_graph_world_step_budget']-steps,p['yaw_config'])
                steps+=result['actual_graph_world_steps']
                entry=dict(**current,path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),
                    evaluations=len(job['cases']),indices=job['indices'],success=result['summary']['success_count'],
                    physical=result['physical'],design=result['design'],actual_graph_world_steps=result['actual_graph_world_steps'],
                    first_episode_world_steps=result['first_episode_world_steps'],checked=checked)
                results.append(entry)
                atomic_json(OUT/'progress.json',dict(completed=sum(x['evaluations'] for x in results),records=results,
                    actual_graph_world_steps=steps,FD_calls=len(fd_calls)))
                print('COMPLETED306',arm,job['batch'],entry['success'],'/',entry['evaluations'],flush=True)
        assert sum(x['evaluations'] for x in results)==200
        atomic_json(OUT/'completion.json',dict(verified=True,round=306,records=results,evaluations=200,
            actual_graph_world_steps=steps,first_episode_world_steps=sum(x['first_episode_world_steps'] for x in results),
            baseline_constructor_FD_calls=fd_calls,wall_seconds=time.perf_counter()-start,
            admission_sha256=sha(OUT/'runner_admission.json'),new_training_samples=0,formal_PPO_admitted=False,
            next='307 independentwhole200/gates/raw comparison; do notselectfeedback orclaimbenefitfrompartialcounts.'))
        print('DONE306200 fixedfeedback queue',flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),current=current,records=results,
            completed_evaluations=sum(x['evaluations'] for x in results),checked_actual_graph_world_steps=steps,
            FD_calls=fd_calls,implicit_retry=False,formal_PPO_admitted=False))
        raise
    finally:
        mujoco.mjd_transitionFD=fd


if __name__=='__main__':
    import sys
    {'admit':admit,'run':run}[sys.argv[1]]()
