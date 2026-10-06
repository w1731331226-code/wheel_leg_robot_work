"""One admitted 40-episode classical queue; preserve interruptions, never resume silently."""
import ast
import inspect
import json
from pathlib import Path

import numpy as np
from mujoco_warp._src.support import contact_force_fn

import complete_contact_recorder as rec
from score_reference_learning import load_rows

OUT, ROOT, sha, write = rec.OUT, rec.ROOT, rec.sha, rec.atomic_json


def verify():
    p = json.loads((OUT/'proposal.json').read_text())
    c = json.loads((OUT/'source_contract.json').read_text())
    a = json.loads((OUT/'admission_review.json').read_text())
    r = json.loads((OUT/'runner_contract.json').read_text())
    assert c['verified'] and a['verified'] and a['source_admitted_for_next_queue']
    assert c['proposal_sha256'] == a['proposal_sha256'] == sha(OUT/'proposal.json')
    assert c['unit_sha256'] == a['unit_sha256'] == sha(OUT/'unit.json')
    assert a['source_contract_sha256'] == sha(OUT/'source_contract.json')
    assert a['supplemental_sha256'] == sha(OUT/'supplemental_counts_check.json')
    assert all(sha(ROOT/n) == v for n,v in c['source_sha256'].items())
    assert c['contact_api_sha256'] == sha(Path(inspect.getfile(contact_force_fn.func)))
    import mujoco_warp._src.forward as forward
    assert a['installed_forward_sha256'] == sha(Path(forward.__file__))
    previous = OUT/'first_admission/source_before_partial_access.py'
    assert a['prior_tested_source_sha256'] == sha(previous)
    def bodies(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old, current = bodies(previous), bodies(ROOT/'wheelleg_warp/complete_contact_recorder.py')
    assert all(old[n] == current[n] for n in ('contact_counts','contact_copy','state_copy','validate'))
    assert sha(ROOT/p['parent_proposal']) == p['parent_proposal_sha256']
    assert r['runner_source_sha256'] == sha(Path(__file__))
    assert all(sha(ROOT/n) == v for n,v in r['input_sha256'].items())
    assert len(p['cases'])*len(p['classical']) == r['budget'] == p['budget_first_episode_evaluations'] == 40
    assert c['columns'] == rec.old.COL and c['contact_columns'] == rec.CONTACT_COL
    return p


def check_files(directory, cases, archived):
    result = load_rows(directory/'result.json',cases)
    geometry = json.loads((directory/'geometry.json').read_text())
    assert geometry['case_ids'] == [c['seed'] for c in cases]
    total_steps = total_contacts = total_bytes = 0
    differences = []
    for w,row in enumerate(result['runs']):
        f = directory/row['complete_trace']['path']
        assert sha(f) == row['complete_trace']['sha256']
        with np.load(f,allow_pickle=False) as z:
            assert z['columns'].tolist() == rec.old.COL and z['contact_columns'].tolist() == rec.CONTACT_COL
            sizes = z['state_sizes'].tolist()
            data = tuple(z[n] for n in ('trace','pre','post','meta','contacts','contact_steps'))
            rec.validate(data)
            t,pre,post,meta,contacts,steps = data
            assert sizes == [17,16,6] and pre.shape[1] == sum(sizes)
            assert len(t) == row['physical_steps'] == row['complete_trace']['steps']
            assert len(contacts) == row['complete_trace']['contacts']
            assert row['complete_counts'] == dict(steps=len(t),contacts=len(contacts))
            assert np.all(contacts[:,0] == 1) and np.all(contacts[:,1] == w)
            assert np.all(contacts[:,2:4] == contacts[:,2:4].astype(int))
            assert np.all((contacts[:,2:4] >= 0) & (contacts[:,2:4] < len(geometry['geom_names'])))
            assert np.isin(contacts[:,5],[1,3,4,6]).all()
            # Controller/command logic reads q/v while active: consecutive physical states join exactly.
            np.testing.assert_array_equal(pre[1:,:sum(sizes[:2])],post[:-1,:sum(sizes[:2])])
            total_steps += len(t); total_contacts += len(contacts)
        total_bytes += f.stat().st_size
        prior = archived[row['seed']]
        changed = {key:dict(archived=prior[key],current=row[key]) for key in
                   ('success','reason','physical_safety_passed','design_joint_passed','terrain_passed','terrain_exit_passed') if prior[key] != row[key]}
        if changed:
            differences.append(dict(case=row['seed'],changed=changed))
    return dict(verified=True,evaluations=len(cases),physics_steps=total_steps,contacts=total_contacts,
                trace_bytes=total_bytes,geometry_sha256=sha(directory/'geometry.json'),original_label_differences=differences)


def preserve_partial(raw, directory, cases):
    """Keep both accumulated prefixes and the last buffer, even on validation failure."""
    arrays = raw._complete_buffers
    snapshot = [a.numpy() for a in arrays]
    names = ('trace','tracking','count','window','pre','post','meta','contacts')
    np.savez_compressed(directory/'interrupted_last_buffers.npz',**dict(zip(names,snapshot)))
    records = []
    for w, chunks in enumerate(raw._complete_chunks):
        if chunks:
            data = tuple(np.concatenate([x[i] for x in chunks]) for i in range(6))
            f = directory/f'interrupted_prefix_{cases[w]["seed"]}.npz'
            np.savez_compressed(f,**dict(zip(('trace','pre','post','meta','contacts','contact_steps'),data)))
        records.append(dict(case=cases[w]['seed'],physical_steps_attempted=int(snapshot[2][w,0]),
                            saved_complete=w in raw._complete_frozen,accumulated_chunks=len(chunks)))
    return dict(records=records,last_buffers_sha256=sha(directory/'interrupted_last_buffers.npz'),
                note='Last buffer may overlap saved prefix. Preserve separately; no silent concatenation or restart.')


def freeze():
    assert not (OUT/'runner_contract.json').exists()
    inputs = {str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in
              ('proposal.json','source_contract.json','unit.json','admission_review.json','supplemental_counts_check.json')}
    completion = rec.old.OUT/'completion.json'
    inputs[str(completion.relative_to(ROOT))] = sha(completion)
    jobs = json.loads(completion.read_text())['records']
    for label in ('B0','B1-route'):
        job = next(x for x in jobs if x['label'] == label)
        file = rec.old.OUT/job['path']
        assert sha(file) == job['sha256']
        inputs[str(file.relative_to(ROOT))] = sha(file)
    write(OUT/'runner_contract.json',dict(runner_source_sha256=sha(Path(__file__)),input_sha256=inputs,budget=40,
        updates=0,serialization='Canonical JSON before reading/checking; avoids native tuple versus JSON list mismatch.',
        interruption='Save consumed count, complete files, accumulated prefixes and latest buffers. No implicit resume.',
        worker='wheelleg-complete-contact-witness-v2.service; exclusive project-write.lock'))
    verify()
    print('FROZEN one40 classical queue; 0 episodes run',flush=True)


def run():
    p = verify()
    destination = OUT/'runs'
    destination.mkdir(exist_ok=False)
    ledger = []
    finished = attempted = 0
    raw = None
    directory = None
    try:
        for label,classical in p['classical'].items():
            verify()
            raw = None
            directory = destination/label
            directory.mkdir(exist_ok=False)
            factory = rec.old.evaluator.raw_env
            def wrapped(cases,mode):
                nonlocal raw
                raw = rec.instrument(factory,cases,mode,directory)
                wait = raw.step_wait
                batches = 0
                def observed_wait():
                    nonlocal batches
                    result = wait(); batches += 1
                    if batches % 25 == 0:
                        write(OUT/'progress.json',dict(status='running',job=label,attempted_evaluations=attempted,
                            completed_jobs_evaluations=finished,first_episodes_saved=len(raw._complete_frozen),
                            recorded_first_physics_steps=int(raw._complete_buffers[2].numpy()[:,0].sum())))
                    return result
                raw.step_wait = observed_wait
                return raw
            attempted += len(p['cases'])
            write(OUT/'progress.json',dict(status='starting',job=label,reserved_evaluations=attempted,completed_jobs_evaluations=finished))
            rec.old.evaluator.raw_env = wrapped
            try:
                result = rec.old.evaluator.evaluate(p['cases'],'M3-route',classical=classical)
            finally:
                rec.old.evaluator.raw_env = factory
            # Canonicalize only serialization, never the physics/controller/task data.
            result = json.loads(json.dumps(result,allow_nan=False))
            write(directory/'result.json',result)
            finished += len(result['runs'])
            parent = json.loads((rec.old.OUT/f'runs/{label}/result.json').read_text())
            checked = check_files(directory,p['cases'],{r['seed']:r for r in parent['runs']})
            ledger.append(dict(label=label,path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),**checked))
            write(OUT/'completed_jobs.json',dict(completed_evaluations=finished,records=ledger))
            verify()
            print('COMPLETED',label,finished,'success',result['summary']['success_count'],'steps',checked['physics_steps'],'contacts',checked['contacts'],flush=True)
        assert attempted == finished == 40
        write(OUT/'completion.json',dict(verified=True,completed_evaluations=finished,unique_development_cases=len(p['cases']),
            records=ledger,runner_contract_sha256=sha(OUT/'runner_contract.json'),training_updates=0,
            physical_steps=sum(x['physics_steps'] for x in ledger),contacts=sum(x['contacts'] for x in ledger),
            scope='40 replayed episodes on20 existing development cases,2 fixed laws; not independent generalization or40 trained seeds. Counts cover first episodes, not extra autoreset work.'))
        write(OUT/'progress.json',dict(status='complete',completed_evaluations=finished))
    except BaseException as error:
        partial = None
        if raw is not None and directory is not None:
            partial = preserve_partial(raw,directory,p['cases'])
        write(OUT/'interruption.json',dict(error=repr(error),finished_evaluations=finished,reserved_evaluations=attempted,
            checked_jobs=ledger,partial=partial,silently_resumable=False))
        raise


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run'])
    globals()[parser.parse_args().command]()
