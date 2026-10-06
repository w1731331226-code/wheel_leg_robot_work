"""One registered328 queue, with inherited full/gyro/role and phase evidence."""
import json
from pathlib import Path
import numpy as np
import mujoco_warp._src.forward as forward
import mujoco_warp._src.support as support
import phase_support_probe as probe
import run_floor_broad_qualification as broad_runner

OUT = probe.OUT
rec = probe.rec
ROOT, sha, write = rec.ROOT, rec.sha, rec.atomic_json
prior = broad_runner.prior


def jobs(p):
    result = []
    for panel, cases in p['panels'].items():
        groups = probe.broad.batches(cases)
        assert sorted(i for g in groups for i, _ in g) == list(range(len(cases)))
        for law in p['classical']:
            for batch, group in enumerate(groups):
                result.append(dict(panel=panel, law=law, batch=batch,
                    indices=[i for i, _ in group], cases=[c for _, c in group]))
    assert sum(len(j['cases']) for j in result) == 328
    return result


def verify():
    p = json.loads((OUT/'proposal.json').read_text())
    c = json.loads((OUT/'source_contract.json').read_text())
    a = json.loads((OUT/'admission_review.json').read_text())
    r = json.loads((OUT/'runner_contract.json').read_text())
    assert c['verified'] and a['verified'] and a['source_admitted_for_328_queue']
    assert a['conditional656_reuse_source_qualified']
    assert c['proposal_sha256'] == a['proposal_sha256'] == sha(OUT/'proposal.json')
    assert c['unit_sha256'] == a['unit_sha256'] == sha(OUT/'unit.json')
    assert a['source_contract_sha256'] == sha(OUT/'source_contract.json')
    assert a['reuse_unit_sha256'] == sha(OUT/'reuse_unit.json')
    assert a['installed_forward_sha256'] == sha(Path(forward.__file__))
    assert a['installed_support_sha256'] == sha(Path(support.__file__))
    assert all(sha(ROOT/n) == s for n, s in c['source_sha256'].items())
    assert r['source_sha256'] == sha(Path(__file__)) and r['jobs'] == jobs(p)
    assert all(sha(ROOT/n) == s for n, s in r['input_sha256'].items())
    assert r['budget'] == p['new_evaluation_budget'] == 328 and p['training_updates'] == 0
    assert p['noise_std_rad_s'] == 0 and r['reused_rows'] == 656
    return p


def preserve(raw, directory, cases):
    report = prior.preserve(raw, directory, cases)
    report['phase'] = probe.preserve_phase(raw, directory)
    report['ID_mapping'] = raw._broad_contract
    return report


def freeze():
    assert not (OUT/'runner_contract.json').exists()
    p = json.loads((OUT/'proposal.json').read_text())
    inputs = {str((OUT/n).relative_to(ROOT)): sha(OUT/n) for n in
        ('proposal.json', 'source_contract.json', 'admission_review.json', 'unit.json', 'reuse_unit.json')}
    inputs.update(json.loads((OUT/'reuse_unit.json').read_text())['input_sha256'])
    for module in (broad_runner, prior, prior.prior):
        inputs[str(Path(module.__file__).relative_to(ROOT))] = sha(Path(module.__file__))
    write(OUT/'runner_contract.json', dict(source_sha256=sha(Path(__file__)), input_sha256=inputs,
        budget=328, reused_rows=656, updates=0, jobs=jobs(p),
        interruption='Inherited full/gyro/role plus phase prefixes/last buffer and ID mapping; no implicit resume or budget increase.',
        worker='wheelleg-phase-support-qualification-v1.service; exclusive project-write.lock'))
    verify()
    print('FROZEN328 queue/656 qualified reused rows;0 evaluations run', flush=True)


def logs(directory, cases):
    total = broad_runner.logs(directory, cases, 'original')
    for row in json.loads((directory/'result.json').read_text())['runs']:
        file = directory/row['phase_trace']['path']; assert sha(file) == row['phase_trace']['sha256']
        with np.load(file, allow_pickle=False) as z:
            assert z['columns'].tolist() == probe.COL
            phase = z['trace']; probe.check_log(phase, 'phase_support')
            assert len(phase) == row['physical_steps'] == row['phase_trace']['rows']
        with np.load(directory/row['role_trace']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(phase[:, 4], z['trace'][:, 3])
            np.testing.assert_array_equal(phase[:, 5], z['trace'][:, 5])
        with np.load(directory/row['complete_trace']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(phase[:, :2], z['trace'][:, :2])
            np.testing.assert_array_equal(phase[:, 2], z['trace'][:, 43])
    return total


def original_map(p, job):
    if job['panel'] != 'legacy':
        refs = [p['original_development_references'][job['law']+'/'+job['panel']]]
    else:
        refs = p['currentGPU_original_legacy_references'][job['law']]
    rows = [r for ref in refs for r in json.loads((ROOT/ref['path']).read_text())['runs']]
    return {r['seed']: r for r in rows}


def run():
    p = verify(); destination = OUT/'runs'; destination.mkdir(exist_ok=False)
    ledger = []; finished = reserved = 0; raw = directory = current = None
    try:
        for current in jobs(p):
            verify(); raw = None; cases = current['cases']
            key = f'{current["panel"]}/{current["law"]}/batch_{current["batch"]}'
            directory = destination/key; directory.mkdir(parents=True, exist_ok=False)
            factory = rec.old.evaluator.raw_env
            def wrapped(group, mode):
                nonlocal raw
                assert group == cases
                raw = probe.instrument(group, mode, directory=directory)
                wait = raw.step_wait; batches = 0
                def observed():
                    nonlocal batches
                    result = wait(); batches += 1
                    if batches % 25 == 0:
                        write(OUT/'progress.json', dict(status='running', job=key,
                            reserved_evaluations=reserved, completed_evaluations=finished,
                            first_episodes_saved=len(raw._phase_frozen),
                            first_physics_steps=int(raw._complete_buffers[2].numpy()[:, 0].sum())))
                    return result
                raw.step_wait = observed
                return raw
            reserved += len(cases)
            write(OUT/'progress.json', dict(status='starting', job=key,
                reserved_evaluations=reserved, completed_evaluations=finished))
            rec.old.evaluator.raw_env = wrapped
            try:
                result = rec.old.evaluator.evaluate(cases, 'M3-route', classical=p['classical'][current['law']])
            finally:
                rec.old.evaluator.raw_env = factory
            result = json.loads(json.dumps(result, allow_nan=False))
            write(directory/'result.json', result); finished += len(result['runs'])
            checked = prior.prior.check_files(directory, cases, original_map(p, current))
            assert logs(directory, cases) == checked['physics_steps']
            ledger.append(dict(panel=current['panel'], law=current['law'], batch=current['batch'],
                indices=current['indices'], path=str((directory/'result.json').relative_to(OUT)),
                sha256=sha(directory/'result.json'), contracts={n: sha(directory/n) for n in
                ('broad_contract.json', 'role_contract.json', 'gyro_contract.json', 'phase_contract.json')}, **checked))
            write(OUT/'completed_jobs.json', dict(completed_evaluations=finished, records=ledger)); verify()
            print('COMPLETED', finished, key, 'success', result['summary']['success_count'],
                  'physical', result['physical'], 'design', result['design'], flush=True)
        assert finished == reserved == 328
        write(OUT/'completion.json', dict(verified=True, completed_new_evaluations=328,
            reused_records=656, comparison_records=984, registered_case_IDs=164, records=ledger,
            runner_contract_sha256=sha(OUT/'runner_contract.json'),
            new_first_physics_steps=sum(j['physics_steps'] for j in ledger),
            new_contact_records=sum(j['contacts'] for j in ledger), training_updates=0,
            scope='Existing development/regression IDs, not independent tests or trained seeds. First episode counts exclude surplus autoreset work. Paired gates not yet reviewed.'))
        write(OUT/'progress.json', dict(status='complete', completed_new_evaluations=328, reused_records=656))
    except BaseException as error:
        partial = preserve(raw, directory, current['cases']) if raw is not None else None
        write(OUT/'interruption.json', dict(error=repr(error), finished_evaluations=finished,
            reserved_evaluations=reserved, current_job=current, checked_jobs=ledger,
            partial=partial, silently_resumable=False))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['freeze', 'run'])
    globals()[parser.parse_args().command]()
