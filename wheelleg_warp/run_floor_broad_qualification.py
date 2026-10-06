"""Single344 first-episode broad qualification; immutable references/source reuse."""
import json
from pathlib import Path
import numpy as np
import mujoco_warp._src.forward as forward
import floor_broad_adapter as adapter
import run_reference_role_probe as prior

OUT=adapter.OUT;rec=adapter.rec;ROOT=rec.ROOT;sha=rec.sha;write=rec.atomic_json


def jobs(p):
    result=[]
    for panel,arms in [('regular',['floor_only']),('controlled_remaining',['floor_only']),('legacy',['original','floor_only'])]:
        for arm in arms:
            for law in p['classical']:
                for batch,group in enumerate(adapter.batches(p['panels'][panel])):
                    result.append(dict(panel=panel,arm=arm,law=law,batch=batch,indices=[i for i,c in group],cases=[c for i,c in group]))
    assert sum(len(j['cases']) for j in result)==344
    return result


def verify():
    p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'source_contract.json').read_text())
    a=json.loads((OUT/'admission_review.json').read_text());r=json.loads((OUT/'runner_contract.json').read_text())
    assert c['verified'] and a['verified'] and a['source_admitted_for_344_queue']
    assert c['proposal_sha256']==a['proposal_sha256']==sha(OUT/'proposal.json')
    assert c['unit_sha256']==a['unit_sha256']==sha(OUT/'unit.json') and a['source_contract_sha256']==sha(OUT/'source_contract.json')
    assert a['supplemental_sha256']==sha(OUT/'supplemental_metadata_unit.json')
    assert a['installed_forward_sha256']==sha(Path(forward.__file__))
    assert all(sha(ROOT/n)==v for n,v in c['source_sha256'].items())
    assert r['source_sha256']==sha(Path(__file__)) and r['jobs']==jobs(p)
    assert all(sha(ROOT/n)==v for n,v in r['input_sha256'].items())
    assert r['budget']==p['new_evaluation_budget']==344 and p['noise_std_rad_s']==0
    unit=json.loads((OUT/'unit.json').read_text())
    assert all(owner['role']['gyro_alpha']==.025 for owner in unit['batch_owners'])
    return p


def freeze():
    assert not (OUT/'runner_contract.json').exists();p=json.loads((OUT/'proposal.json').read_text())
    inputs={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ('proposal.json','source_contract.json','admission_review.json','unit.json','supplemental_metadata_unit.json')}
    for group in ('original_development_references','covered_candidate_references'):
        for ref in p[group].values():assert sha(ROOT/ref['path'])==ref['sha256'];inputs[ref['path']]=ref['sha256']
    for module in (prior,prior.prior):inputs[str(Path(module.__file__).relative_to(ROOT))]=sha(Path(module.__file__))
    write(OUT/'runner_contract.json',dict(source_sha256=sha(Path(__file__)),input_sha256=inputs,budget=344,reused_rows=312,jobs=jobs(p),updates=0,
        interruption='All full/gyro/role prefix/last buffers/metadata IDs preserved; no resume or hidden baseline repeats.',
        order='Groups preserve global panel indices; originallegacy jobs finish beforecandidatelegacy. CPU historical calibration separate.'))
    verify();print('FROZEN344 queue,homogeneous<=20worlds;0eval run',flush=True)


def logs(directory,cases,arm):
    rows=json.loads((directory/'result.json').read_text())['runs'];total=0
    for row in rows:
        for field,columns in (('gyro_trace',adapter.roles.noise.COL),('role_trace',adapter.roles.COL)):
            file=directory/row[field]['path'];assert sha(file)==row[field]['sha256']
            with np.load(file,allow_pickle=False) as z:
                assert z['columns'].tolist()==columns;data=z['trace'];assert len(data)==row['physical_steps']==row[field]['rows']
                if field=='gyro_trace':adapter.roles.noise.check_log(data,np.zeros(len(data),np.float32),.025)
                else:
                    adapter.roles.check_log(data,0 if arm=='original' else 1)
                    with np.load(directory/row['complete_trace']['path'],allow_pickle=False) as full:
                        np.testing.assert_allclose(data[:,6],full['trace'][:,27],atol=1e-12,rtol=0)
                        np.testing.assert_allclose(data[:,7:9],full['trace'][:,29:31],atol=1e-12,rtol=0)
        total+=row['physical_steps']
    return total


def original_map(p,job,ledger):
    if job['panel']!='legacy':
        panel='controlled' if job['panel']=='controlled_remaining' else 'regular'
        file=ROOT/p['original_development_references'][job['law']+'/'+panel]['path']
        return {r['seed']:r for r in json.loads(file.read_text())['runs']}
    rows=[]
    for j in ledger:
        if j['panel']=='legacy' and j['arm']=='original' and j['law']==job['law']:
            rows.extend(json.loads((OUT/j['path']).read_text())['runs'])
    return {r['seed']:r for r in rows}


def run():
    p=verify();dest=OUT/'runs';dest.mkdir(exist_ok=False)
    ledger=[];finished=reserved=0;raw=None;directory=None;current=None
    try:
        for current in jobs(p):
            verify();raw=None;cases=current['cases'];key=f'{current["panel"]}/{current["arm"]}/{current["law"]}/batch_{current["batch"]}'
            directory=dest/key;directory.mkdir(parents=True,exist_ok=False);factory=rec.old.evaluator.raw_env
            def wrapped(group,mode):
                nonlocal raw
                assert group==cases;raw=adapter.instrument(group,mode,current['arm'],directory);wait=raw.step_wait;batches=0
                def observed():
                    nonlocal batches
                    result=wait();batches+=1
                    if batches%25==0:write(OUT/'progress.json',dict(status='running',job=key,reserved_evaluations=reserved,completed_evaluations=finished,
                        first_episodes_saved=len(raw._role_frozen),first_physics_steps=int(raw._complete_buffers[2].numpy()[:,0].sum())))
                    return result
                raw.step_wait=observed;return raw
            reserved+=len(cases);write(OUT/'progress.json',dict(status='starting',job=key,reserved_evaluations=reserved,completed_evaluations=finished))
            rec.old.evaluator.raw_env=wrapped
            try:result=rec.old.evaluator.evaluate(cases,'M3-route',classical=p['classical'][current['law']])
            finally:rec.old.evaluator.raw_env=factory
            result=json.loads(json.dumps(result,allow_nan=False));write(directory/'result.json',result);finished+=len(result['runs'])
            archive=original_map(p,current,ledger) if not(current['panel']=='legacy' and current['arm']=='original') else {r['seed']:r for r in result['runs']}
            assert all(c['seed'] in archive for c in cases)
            checked=prior.prior.check_files(directory,cases,archive);assert logs(directory,cases,current['arm'])==checked['physics_steps']
            ledger.append(dict(panel=current['panel'],arm=current['arm'],law=current['law'],batch=current['batch'],indices=current['indices'],
                path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),
                broad_contract_sha256=sha(directory/'broad_contract.json'),role_contract_sha256=sha(directory/'role_contract.json'),gyro_contract_sha256=sha(directory/'gyro_contract.json'),**checked))
            write(OUT/'completed_jobs.json',dict(completed_evaluations=finished,records=ledger));verify()
            print('COMPLETED',finished,key,'success',result['summary']['success_count'],'physical',result['physical'],'design',result['design'],flush=True)
        assert finished==reserved==344
        write(OUT/'completion.json',dict(verified=True,completed_new_evaluations=344,reused_records=312,comparison_records=656,
            registered_case_IDs=164,records=ledger,runner_contract_sha256=sha(OUT/'runner_contract.json'),
            new_first_physics_steps=sum(j['physics_steps'] for j in ledger),new_contact_records=sum(j['contacts'] for j in ledger),training_updates=0,
            scope='Registered development/regression reused+new records,not656 independent cases/trainingseeds. Firstepisode counts exclude autoreset surplus; smallbatch float differences not bitwise claims.'))
        write(OUT/'progress.json',dict(status='complete',completed_new_evaluations=344,reused_records=312))
    except BaseException as error:
        partial=prior.preserve(raw,directory,current['cases']) if raw is not None else None
        write(OUT/'interruption.json',dict(error=repr(error),finished_evaluations=finished,reserved_evaluations=reserved,current_job=current,checked_jobs=ledger,partial=partial,silently_resumable=False));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run'])
    globals()[parser.parse_args().command]()
