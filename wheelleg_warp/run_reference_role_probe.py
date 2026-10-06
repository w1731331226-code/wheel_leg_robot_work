"""One160-new-episode role contrast,80 immutable source-qualified reused rows."""
import json
from pathlib import Path
import numpy as np
import mujoco_warp._src.forward as forward
import reference_role_probe as probe
import run_nom_yaw_filter_probe as gyro_runner

rec=probe.rec;OUT=probe.OUT;ROOT=rec.ROOT;sha=rec.sha;write=rec.atomic_json
prior=gyro_runner.prior


def verify():
    p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'source_contract.json').read_text())
    a=json.loads((OUT/'admission_review.json').read_text());r=json.loads((OUT/'runner_contract.json').read_text())
    assert c['verified'] and a['verified'] and a['source_admitted_for_160_queue'] and a['original80_reuse_source_qualified']
    assert c['proposal_sha256']==a['proposal_sha256']==sha(OUT/'proposal.json')
    assert c['unit_sha256']==a['unit_sha256']==sha(OUT/'unit.json')
    assert a['source_contract_sha256']==sha(OUT/'source_contract.json') and a['supplemental_sha256']==sha(OUT/'supplemental_terminal_unit.json')
    assert p['noise_file_sha256']==c['noise_file_sha256']==sha(ROOT/p['noise_file'])
    assert a['installed_forward_sha256']==sha(Path(forward.__file__))
    assert all(sha(ROOT/n)==v for n,v in c['source_sha256'].items())
    assert r['source_sha256']==sha(Path(__file__)) and all(sha(ROOT/n)==v for n,v in r['input_sha256'].items())
    assert r['new_budget']==p['new_episode_budget']==160 and p['training_budget']==0
    probe.check_control_copy()
    with np.load(ROOT/p['noise_file'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['noise'],probe.noise.noise_table(p['cases'],.02))
        assert z['case_ids'].tolist()==[x['seed'] for x in p['cases']]
    for ref in p['reuse_references'].values():assert sha(ROOT/ref['path'])==ref['sha256']
    return p


def freeze():
    assert not (OUT/'runner_contract.json').exists()
    p=json.loads((OUT/'proposal.json').read_text())
    inputs={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ('proposal.json','source_contract.json','admission_review.json','unit.json','supplemental_terminal_unit.json')}
    inputs[p['noise_file']]=p['noise_file_sha256']
    for ref in p['reuse_references'].values():inputs[ref['path']]=ref['sha256']
    for module in (gyro_runner,prior):inputs[str(Path(module.__file__).relative_to(ROOT))]=sha(Path(module.__file__))
    write(OUT/'runner_contract.json',dict(source_sha256=sha(Path(__file__)),input_sha256=inputs,new_budget=160,reused_rows=80,updates=0,
        contract='Single8-job queue, canonical JSON/full+gyro+role checks, consumed partial buffers retained; no silent resume.'))
    verify();print('FROZEN160new+80qualified reused;0physics evaluations run',flush=True)


def role_check(directory,cases,arm):
    rows=json.loads((directory/'result.json').read_text())['runs'];total=0
    for row in rows:
        file=directory/row['role_trace']['path'];assert sha(file)==row['role_trace']['sha256']
        with np.load(file,allow_pickle=False) as z:
            assert z['columns'].tolist()==probe.COL;data=z['trace']
            probe.check_log(data,{'floor_only':1,'consistent_pair':2}[arm])
            assert len(data)==row['physical_steps']==row['role_trace']['rows'];total+=len(data)
        with np.load(directory/row['complete_trace']['path'],allow_pickle=False) as z:
            np.testing.assert_allclose(data[:,6],z['trace'][:,27],atol=1e-12,rtol=0)
            np.testing.assert_allclose(data[:,7:9],z['trace'][:,29:31],atol=1e-12,rtol=0)
    return total


def preserve(raw,directory,cases):
    report=gyro_runner.preserve(raw,directory,cases)
    np.savez_compressed(directory/'interrupted_role_last_buffers.npz',reference=raw._role_buffers[0].numpy(),monitor=raw._role_buffers[1].numpy())
    for w,chunks in enumerate(raw._role_chunks):
        if chunks:np.savez_compressed(directory/f'interrupted_role_prefix_{cases[w]["seed"]}.npz',trace=np.concatenate(chunks))
    report['role_complete_case_ids']=[cases[w]['seed'] for w in sorted(raw._role_frozen)]
    return report


def run():
    p=verify();dest=OUT/'runs';dest.mkdir(exist_ok=False)
    jobs=[(a['name'],std,label,config) for a in p['arms'] if not a['reuse'] for std in p['noise_std_rad_s'] for label,config in p['classical'].items()]
    assert len(jobs)*len(p['cases'])==160
    ledger=[];finished=reserved=0;raw=None;directory=None
    try:
        for arm,std,label,classical in jobs:
            verify();raw=None;noise='clean' if std==0 else 'noisy';key=arm+'/'+noise+'/'+label
            directory=dest/arm/noise/label;directory.mkdir(parents=True,exist_ok=False)
            factory=rec.old.evaluator.raw_env
            def wrapped(cases,mode):
                nonlocal raw
                raw=probe.instrument(cases,mode,arm,std,directory);wait=raw.step_wait;batches=0
                def observed():
                    nonlocal batches
                    result=wait();batches+=1
                    if batches%25==0:write(OUT/'progress.json',dict(status='running',job=key,reserved_evaluations=reserved,completed_evaluations=finished,
                        first_episodes_saved=len(raw._role_frozen),first_physics_steps=int(raw._complete_buffers[2].numpy()[:,0].sum())))
                    return result
                raw.step_wait=observed;return raw
            reserved+=len(p['cases']);write(OUT/'progress.json',dict(status='starting',job=key,reserved_evaluations=reserved,completed_evaluations=finished))
            rec.old.evaluator.raw_env=wrapped
            try:result=rec.old.evaluator.evaluate(p['cases'],'M3-route',classical=classical)
            finally:rec.old.evaluator.raw_env=factory
            result=json.loads(json.dumps(result,allow_nan=False));write(directory/'result.json',result);finished+=len(result['runs'])
            old=json.loads((ROOT/p['reuse_references'][noise+'/'+label]['path']).read_text())
            checked=prior.check_files(directory,p['cases'],{r['seed']:r for r in old['runs']})
            assert gyro_runner.gyro_check(directory,p['cases'],dict(alpha=.025,noise_std_rad_s=std))==role_check(directory,p['cases'],arm)==checked['physics_steps']
            ledger.append(dict(arm=arm,noise=noise,classical=label,path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),
                gyro_contract_sha256=sha(directory/'gyro_contract.json'),role_contract_sha256=sha(directory/'role_contract.json'),**checked))
            write(OUT/'completed_jobs.json',dict(completed_evaluations=finished,records=ledger));verify()
            print('COMPLETED',finished,key,'success',result['summary']['success_count'],'physical',result['physical'],'design',result['design'],flush=True)
        assert finished==reserved==160
        write(OUT/'completion.json',dict(verified=True,completed_new_evaluations=160,reused_original_records=80,comparison_records=240,unique_development_cases=20,
            records=ledger,reuse_references=p['reuse_references'],runner_contract_sha256=sha(OUT/'runner_contract.json'),
            new_first_episode_physics_steps=sum(j['physics_steps'] for j in ledger),new_contact_records=sum(j['contacts'] for j in ledger),training_updates=0,
            scope='20existing development cases repeated byfixedlaws/noise/role arms; not240independent cases/trainingseeds. First-episode counts omit autoreset surplus.'))
        write(OUT/'progress.json',dict(status='complete',completed_new_evaluations=160,reused_records=80))
    except BaseException as error:
        partial=preserve(raw,directory,p['cases']) if raw is not None else None
        write(OUT/'interruption.json',dict(error=repr(error),finished_evaluations=finished,reserved_evaluations=reserved,checked_jobs=ledger,partial=partial,silently_resumable=False));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run'])
    globals()[parser.parse_args().command]()
