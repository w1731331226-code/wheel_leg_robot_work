"""Explicit recovery at a fully audited job boundary; B0 is never replayed."""
import json
from pathlib import Path

from stable_baselines3 import PPO
from task_mode_witness import OUT, verify, check_traces
from task_mode_recorder import evaluate
from smoke_reward_training import weight_digest
from review_yaw_sector import sha
from dashboard.live_env import atomic_json


def freeze():
    p=verify(); a=json.loads((OUT/'recovered_B0_audit.json').read_text()); assert a['verified'] and a['known_completed_evaluations']==41
    assert not (OUT/'recovery_contract.json').exists()
    atomic_json(OUT/'recovery_contract.json',dict(source_sha256=sha(__file__),audit_sha256=sha(OUT/'recovered_B0_audit.json'),original_runner_contract_sha256=sha(OUT/'runner_contract.json'),
        completed_job=a['record'],remaining_jobs=['B1-route',*[m['label'] for m in p['models']]],remaining_evaluations=164,total_budget=205,
        deviation='One intended queue segmented after a post-write tuple/list comparison failure. All41 B0 data audited; no incomplete physical episode resumed or rerun. No controls, model/RMS, recorder, cases, task or threshold changes.'))
    print('REGISTERED explicit41+164 recovery, no B0 replay',flush=True)


def run():
    p=verify(); c=json.loads((OUT/'recovery_contract.json').read_text()); assert c['source_sha256']==sha(__file__) and c['audit_sha256']==sha(OUT/'recovered_B0_audit.json')
    assert c['original_runner_contract_sha256']==sha(OUT/'runner_contract.json')
    first=c['completed_job']; assert sha(OUT/first['path'])==first['sha256']
    # Recheck archived baseline, not another physical evaluation.
    baseline=json.loads((OUT/first['path']).read_text()); check_traces((OUT/first['path']).parent,baseline,p['cases'])
    ledger=[first]; completed=41; assert not (OUT/'completion.json').exists()
    jobs=[dict(label='B1-route',classical=p['classical']['B1-route'])]+p['models']
    assert [x['label'] for x in jobs]==c['remaining_jobs']
    try:
        for job in jobs:
            verify(); label=job['label']; directory=OUT/'runs'/label; directory.mkdir(exist_ok=False)
            atomic_json(OUT/'progress.json',dict(status='running_recovery',completed_evaluations=completed,pending_job=label,B0_replayed=False))
            classical=job.get('classical'); model=None; norm=None; arm='M3-route'
            if classical is None:
                prefix=Path(job['prefix']); model=PPO.load(str(prefix)+'.zip',device='cuda'); norm=str(prefix)+'.pkl'; arm='V6-route'; before=(model.num_timesteps,model._n_updates,weight_digest(model))
            result=evaluate(p['cases'],arm,model=model,norm=norm,classical=classical,directory=directory)
            if model is not None: assert before==(model.num_timesteps,model._n_updates,weight_digest(model))
            atomic_json(directory/'result.json',result)
            # JSON intentionally normalizes tuple configuration fields to lists.
            canonical=json.loads(json.dumps(result,allow_nan=False)); checked=check_traces(directory,canonical,p['cases'])
            ledger.append(dict(label=label,path='runs/'+label+'/result.json',sha256=sha(directory/'result.json'),**checked)); completed+=len(result['runs'])
            atomic_json(OUT/'completed_jobs.json',dict(completed_evaluations=completed,records=ledger,recovery_contract_sha256=sha(OUT/'recovery_contract.json'))); verify()
            print('COMPLETED',completed,label,'success',result['summary']['success_count'],flush=True)
        assert completed==205
        atomic_json(OUT/'completion.json',dict(completed_evaluations=completed,records=ledger,runner_contract_sha256=sha(OUT/'runner_contract.json'),recovery_contract_sha256=sha(OUT/'recovery_contract.json'),training_updates=0,model_and_RMS_immutable=True,execution_segments=[41,164],B0_replayed=False))
        atomic_json(OUT/'progress.json',dict(status='complete',completed_evaluations=205,execution_segments=[41,164],B0_replayed=False))
    except BaseException as error:
        atomic_json(OUT/'recovery_interruption.json',dict(completed_evaluations=completed,error=str(error),pending_job_consumption_unknown=True,silently_resumable=False)); raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['freeze','run']); globals()[parser.parse_args().command]()
