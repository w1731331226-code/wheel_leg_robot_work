"""Single registered120-new-episode queue,40 immutable source-qualified reused rows."""
import inspect
import json
from pathlib import Path
import numpy as np
from mujoco_warp._src.support import contact_force_fn
import mujoco_warp._src.forward as forward

import nom_yaw_filter_probe as probe
import complete_contact_witness as prior

OUT,ROOT,sha,write=probe.OUT,probe.rec.ROOT,probe.rec.sha,probe.rec.atomic_json


def verify():
    p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'source_contract.json').read_text())
    a=json.loads((OUT/'admission_review.json').read_text());r=json.loads((OUT/'runner_contract.json').read_text())
    assert c['verified'] and a['verified'] and a['source_admitted_for_120_queue'] and a['original_clean40_reuse_source_qualified']
    assert c['proposal_sha256']==a['proposal_sha256']==sha(OUT/'proposal.json')
    assert c['unit_sha256']==a['unit_sha256']==sha(OUT/'unit.json')
    assert a['source_contract_sha256']==sha(OUT/'source_contract.json') and a['supplemental_sha256']==sha(OUT/'supplemental_terminal_unit.json')
    assert c['noise_file_sha256']==a['noise_file_sha256']==sha(OUT/'noise_table.npz')
    assert c['contact_API_sha256']==sha(Path(inspect.getfile(contact_force_fn.func)))
    assert a['installed_forward_sha256']==sha(Path(forward.__file__))
    assert all(sha(ROOT/n)==v for n,v in c['source_sha256'].items())
    assert r['runner_source_sha256']==sha(Path(__file__)) and r['checker_source_sha256']==sha(Path(prior.__file__))
    assert all(sha(ROOT/n)==v for n,v in r['input_sha256'].items())
    assert p['new_episode_budget']==r['new_budget']==120 and p['learning_budget']==0
    assert np.__version__==c['numpy_version']
    with np.load(OUT/'noise_table.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['noise'],probe.noise_table(p['cases'],.02))
        assert z['case_ids'].tolist()==[c['seed'] for c in p['cases']]
    for ref in p['reuse_references'].values():assert sha(ROOT/ref['path'])==ref['sha256']
    return p


def freeze():
    assert not (OUT/'runner_contract.json').exists()
    inputs={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in
        ('proposal.json','source_contract.json','admission_review.json','unit.json','supplemental_terminal_unit.json','noise_table.npz')}
    p=json.loads((OUT/'proposal.json').read_text())
    for ref in p['reuse_references'].values():inputs[ref['path']]=ref['sha256']
    write(OUT/'runner_contract.json',dict(runner_source_sha256=sha(Path(__file__)),checker_source_sha256=sha(Path(prior.__file__)),input_sha256=inputs,new_budget=120,reused_rows=40,updates=0,
        serialization='Canonical JSON before readback; original score labels/metrics never rewritten.',
        interruption='Preserve full recorder prefixes/buffers plus gyro prefixes/buffer and consumption; never auto-resume.'))
    verify();print('FROZEN120 new+40qualified reused rows;0 learning/evaluations run',flush=True)


def gyro_check(directory,cases,condition):
    result=json.loads((directory/'result.json').read_text());noise=probe.noise_table(cases,condition['noise_std_rad_s'])
    total=0
    for w,row in enumerate(result['runs']):
        file=directory/row['gyro_trace']['path'];assert sha(file)==row['gyro_trace']['sha256']
        with np.load(file,allow_pickle=False) as z:
            assert z['columns'].tolist()==probe.COL
            data=z['trace'];probe.check_log(data,noise[w],condition['alpha'])
            assert len(data)==row['physical_steps']==row['gyro_trace']['rows'];total+=len(data)
    return total


def preserve(raw,directory,cases):
    report=prior.preserve_partial(raw,directory,cases)
    np.savez_compressed(directory/'interrupted_gyro_last_buffers.npz',monitor=raw._gyro_buffers[1].numpy(),error=raw._gyro_buffers[2].numpy())
    for w,chunks in enumerate(raw._gyro_chunks):
        if chunks:np.savez_compressed(directory/f'interrupted_gyro_prefix_{cases[w]["seed"]}.npz',trace=np.concatenate(chunks))
    report['gyro_complete_case_ids']=[cases[w]['seed'] for w in sorted(raw._gyro_frozen)]
    return report


def run():
    p=verify();dest=OUT/'runs';dest.mkdir(exist_ok=False)
    jobs=[(c,label,config) for c in p['conditions'] if not c['reuse'] for label,config in p['classical'].items()]
    assert len(jobs)*len(p['cases'])==120
    ledger=[];finished=reserved=0;raw=None;directory=None
    try:
        for condition,label,classical in jobs:
            verify();raw=None
            key=condition['label']+'/'+label;directory=dest/condition['label']/label;directory.mkdir(parents=True,exist_ok=False)
            factory=probe.rec.old.evaluator.raw_env
            def wrapped(cases,mode):
                nonlocal raw
                raw=probe.instrument(cases,mode,condition,directory);wait=raw.step_wait;batches=0
                def observed():
                    nonlocal batches
                    result=wait();batches+=1
                    if batches%25==0:
                        write(OUT/'progress.json',dict(status='running',job=key,reserved_evaluations=reserved,completed_evaluations=finished,
                            first_episodes_saved=len(raw._gyro_frozen),recorded_first_physics_steps=int(raw._complete_buffers[2].numpy()[:,0].sum())))
                    return result
                raw.step_wait=observed;return raw
            reserved+=len(p['cases']);write(OUT/'progress.json',dict(status='starting',job=key,reserved_evaluations=reserved,completed_evaluations=finished))
            probe.rec.old.evaluator.raw_env=wrapped
            try:result=probe.rec.old.evaluator.evaluate(p['cases'],'M3-route',classical=classical)
            finally:probe.rec.old.evaluator.raw_env=factory
            result=json.loads(json.dumps(result,allow_nan=False));write(directory/'result.json',result);finished+=len(result['runs'])
            original=json.loads((ROOT/p['reuse_references'][label]['path']).read_text())
            checked=prior.check_files(directory,p['cases'],{r['seed']:r for r in original['runs']})
            assert gyro_check(directory,p['cases'],condition)==checked['physics_steps']
            ledger.append(dict(condition=condition['label'],classical=label,path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),
                gyro_contract_sha256=sha(directory/'gyro_contract.json'),**checked))
            write(OUT/'completed_jobs.json',dict(completed_evaluations=finished,records=ledger));verify()
            print('COMPLETED',finished,key,'success',result['summary']['success_count'],'physical',result['physical'],'design',result['design'],flush=True)
        assert finished==reserved==120
        write(OUT/'completion.json',dict(verified=True,completed_new_evaluations=120,reused_original_clean_records=40,comparison_records=160,
            unique_development_cases=20,records=ledger,reuse_references=p['reuse_references'],runner_contract_sha256=sha(OUT/'runner_contract.json'),
            new_first_episode_physics_steps=sum(j['physics_steps'] for j in ledger),new_contact_records=sum(j['contacts'] for j in ledger),
            training_updates=0,scope='Two fixed controllers/four conditions on20 existing development cases; not160 independent cases/training seeds. First-episode counts exclude extra autoreset work.'))
        write(OUT/'progress.json',dict(status='complete',completed_new_evaluations=120,reused_records=40))
    except BaseException as error:
        partial=preserve(raw,directory,p['cases']) if raw is not None else None
        write(OUT/'interruption.json',dict(error=repr(error),finished_evaluations=finished,reserved_evaluations=reserved,checked_jobs=ledger,partial=partial,silently_resumable=False))
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run'])
    globals()[parser.parse_args().command]()
