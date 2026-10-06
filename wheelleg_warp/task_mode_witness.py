"""Single registered fixed-policy witness queue and immutable-input verification."""
import inspect
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from mujoco_warp._src.support import contact_force_fn

from task_mode_recorder import OUT, COL, evaluate
from review_yaw_sector import ROOT, sha
from score_reference_learning import load_rows
from smoke_reward_training import weight_digest
from dashboard.live_env import atomic_json


def verify():
    p=json.loads((OUT/'proposal.json').read_text()); c=json.loads((OUT/'source_contract.json').read_text())
    a=json.loads((OUT/'admission_review.json').read_text()); r=json.loads((OUT/'runner_contract.json').read_text())
    assert c['proposal_sha256']==sha(OUT/'proposal.json') and c['columns']==COL
    assert all(sha(ROOT/n)==v for n,v in c['source_sha256'].items())
    assert c['unit_sha256']==a['unit_sha256']==sha(OUT/'unit.json')
    assert a['source_contract_sha256']==sha(OUT/'source_contract.json') and a['verified'] and a['source_admitted_for_next_queue']
    assert a['supplemental_unit_sha256']==sha(OUT/'supplemental_unit.json') and a['supplemental_source_sha256']==sha(ROOT/'wheelleg_warp/test_task_mode_recorder.py')
    assert c['contact_api_sha256']==sha(Path(inspect.getfile(contact_force_fn.func)))
    assert r['runner_source_sha256']==sha(__file__)
    for name,value in r['input_sha256'].items(): assert sha(OUT/name)==value,name
    assert len(p['cases'])*(len(p['models'])+len(p['classical']))==p['budget_captured_first_episode_evaluations']==205
    for m in p['models']:
        prefix=Path(m['prefix']); assert sha(prefix.with_suffix('.zip'))==m['checkpoint']['checkpoint_sha256'] and sha(prefix.with_suffix('.pkl'))==m['checkpoint']['normalization_sha256']
    return p


def freeze():
    assert not (OUT/'runner_contract.json').exists()
    names=['proposal.json','source_contract.json','unit.json','admission_review.json','supplemental_unit.json']
    atomic_json(OUT/'runner_contract.json',dict(runner_source_sha256=sha(__file__),input_sha256={n:sha(OUT/n) for n in names},budget_captured_first_episode_evaluations=205,training_updates=0))
    verify(); print('FROZEN one205-case queue; no evaluations run',flush=True)


def check_traces(directory,result,cases):
    rows=load_rows(directory/'result.json',cases)['runs']; assert rows==result['runs']; sampled=0
    geometry=json.loads((directory/'geometry.json').read_text()); assert geometry['case_ids']==[c['seed'] for c in cases]
    for row in rows:
        info=row['task_mode_trace']; path=directory/info['path']; assert sha(path)==info['sha256']
        with np.load(path,allow_pickle=False) as data:
            assert data['columns'].tolist()==COL; trace=data['trace']
            assert trace.shape==(info['rows'],len(COL)) and np.isfinite(trace).all() and len(trace)>0
            assert np.all(trace[:,0]==1) and np.all(np.diff(trace[:,1])>0)
            np.testing.assert_allclose(trace[:,2],(trace[:,1]-1)*.0005,rtol=0,atol=1e-12)
            np.testing.assert_allclose(trace[:,3],trace[:,1]*.0005,rtol=0,atol=1e-12)
            assert row['task_mode_counts']==[float(row['physical_steps']),float(len(trace))]
            assert trace[-1,1]<=row['physical_steps']
            sampled+=len(trace)
    return dict(evaluations=len(rows),sampled_rows=sampled,geometry_sha256=sha(directory/'geometry.json'),all_traces_checked=True)


def run():
    p=verify(); destination=OUT/'runs'; destination.mkdir(exist_ok=False); ledger=[]; completed=0
    jobs=[dict(label=name,classical=config) for name,config in p['classical'].items()]+p['models']
    try:
        for job in jobs:
            verify(); label=job['label']; directory=destination/label; directory.mkdir(exist_ok=False)
            atomic_json(OUT/'progress.json',dict(status='running',completed_evaluations=completed,pending_job=label))
            model=None; norm=None; classical=job.get('classical'); arm='M3-route'
            if classical is None:
                prefix=Path(job['prefix']); model=PPO.load(str(prefix)+'.zip',device='cuda'); norm=str(prefix)+'.pkl'; arm='V6-route'
                before=(model.num_timesteps,model._n_updates,weight_digest(model))
            result=evaluate(p['cases'],arm,model=model,norm=norm,classical=classical,directory=directory)
            if model is not None: assert before==(model.num_timesteps,model._n_updates,weight_digest(model))
            atomic_json(directory/'result.json',result); checked=check_traces(directory,result,p['cases'])
            ledger.append(dict(label=label,path='runs/'+label+'/result.json',sha256=sha(directory/'result.json'),**checked))
            completed+=len(result['runs']); atomic_json(OUT/'completed_jobs.json',dict(completed_evaluations=completed,records=ledger)); verify()
            print('COMPLETED',completed,label,'success',result['summary']['success_count'],'sampled',checked['sampled_rows'],flush=True)
        assert completed==205
        atomic_json(OUT/'completion.json',dict(completed_evaluations=completed,records=ledger,runner_contract_sha256=sha(OUT/'runner_contract.json'),training_updates=0,model_and_RMS_immutable=True))
        atomic_json(OUT/'progress.json',dict(status='complete',completed_evaluations=completed))
    except BaseException as error:
        atomic_json(OUT/'interruption.json',dict(completed_evaluations=completed,error=str(error),pending_job_consumption_unknown=True,silently_resumable=False)); raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['freeze','run']); globals()[parser.parse_args().command]()
