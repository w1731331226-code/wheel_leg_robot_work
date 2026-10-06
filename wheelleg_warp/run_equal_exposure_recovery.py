"""One225 first-episode fixed-policy assay, no learning or silent replay."""
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from mujoco_warp._src import support, forward, types, io
import equal_exposure_probe as probe
import run_phase_support_qualification as previous
from smoke_reward_training import weight_digest

OUT, ROOT, sha, write = probe.OUT, probe.rec.ROOT, probe.rec.sha, probe.rec.atomic_json


def jobs(p):
    controllers = [dict(label=k, classical=v, mode='diff3', arm='M3-route')
                   for k, v in p['controllers']['classical'].items()]
    controllers += [dict(label=m['label'], model=m, mode='virtual6', arm='V6-route')
                    for m in p['controllers']['models']]
    groups = probe.phase.broad.batches(p['cases'])
    assert sorted(i for group in groups for i, _ in group) == list(range(45))
    result = [dict(**control, batch=b, indices=[i for i, _ in group], cases=[c for _, c in group])
              for control in controllers for b, group in enumerate(groups)]
    assert len(result) == 15 and sum(len(j['cases']) for j in result) == 225
    return result


def verify():
    p = json.loads((OUT/'proposal.json').read_text()); c = json.loads((OUT/'source_contract.json').read_text())
    a = json.loads((OUT/'admission_review.json').read_text()); r = json.loads((OUT/'runner_contract.json').read_text())
    assert a['verified'] and c['verified'] and a['source_admitted_for_225_queue']
    assert c['proposal_sha256'] == a['proposal_sha256'] == sha(OUT/'proposal.json')
    assert c['unit_sha256'] == a['unit_sha256'] == sha(OUT/'unit.json')
    assert a['source_contract_sha256'] == sha(OUT/'source_contract.json')
    assert c['profile_sha256'] == a['profile_sha256'] == p['stimulus']['profile_sha256'] == sha(OUT/'profiles.npz')
    assert a['profile_manifest_sha256'] == sha(OUT/'profile_manifest.json')
    for name, module in [('support', support), ('forward', forward), ('types', types), ('io', io)]:
        assert a['installed_API_sha256'][name] == sha(Path(module.__file__))
    assert all(sha(ROOT/n) == s for n, s in c['source_sha256'].items())
    assert r['source_sha256'] == sha(__file__) and r['jobs'] == jobs(p)
    assert all(sha(ROOT/n) == s for n, s in r['input_sha256'].items())
    assert r['budget'] == p['budget']['new_first_episode_evaluations'] == 225
    assert p['budget']['training_updates'] == p['budget']['reused_evaluations'] == 0
    for m in p['controllers']['models']:
        prefix = Path(m['prefix'])
        assert sha(prefix.with_suffix('.zip')) == m['checkpoint']['checkpoint_sha256']
        assert sha(prefix.with_suffix('.pkl')) == m['checkpoint']['normalization_sha256']
    return p


def preserve(raw, directory, cases):
    report = previous.preserve(raw, directory, cases)
    report['force'] = probe.preserve_force(raw, directory)
    return report


def freeze():
    assert not (OUT/'runner_contract.json').exists()
    p = json.loads((OUT/'proposal.json').read_text())
    inputs = {str((OUT/n).relative_to(ROOT)): sha(OUT/n) for n in
        ('proposal.json', 'source_contract.json', 'admission_review.json', 'unit.json', 'profiles.npz', 'profile_manifest.json')}
    for module in (previous, previous.prior, previous.prior.prior, previous.broad_runner):
        inputs[str(Path(module.__file__).relative_to(ROOT))] = sha(Path(module.__file__))
    write(OUT/'runner_contract.json', dict(source_sha256=sha(__file__), input_sha256=inputs, budget=225,
        jobs=jobs(p), training_updates=0, reused_rows=0,
        interruption='Allfive stream prefix/last buffers and appliedforce, current job/consumption retained. No implicit resume, survivor selection or added budget.',
        worker='wheelleg-equal-exposure-recovery-v1.service; exclusive project-write.lock'))
    verify(); print('FROZEN225 (200pulse+25zero), all5 fixedcontrollers;0 evaluations run', flush=True)


def logs(directory, cases):
    total = previous.logs(directory, cases); values = probe.profiles()
    contract = json.loads((directory/'force_contract.json').read_text()); root = contract['root_body']
    for row in json.loads((directory/'result.json').read_text())['runs']:
        file = directory/row['force_trace']['path']; assert sha(file) == row['force_trace']['sha256']
        with np.load(file, allow_pickle=False) as z:
            assert z['columns'].tolist() == probe.COL; data = z['trace']
            index = -1 if row['profile'] == 'zero' else row['profile']
            probe.check_samples(data, index, root, values)
            assert len(data) == row['physical_steps'] == row['force_trace']['rows']
            delivered = int(np.count_nonzero((data[:, 1]-1 >= 5000) & (data[:, 1]-1 < 5200))) if index >= 0 else 0
            expected = 200 if index >= 0 else 0
            assert row['force_delivery'] == dict(profile=index, samples=delivered, expected_samples=expected, complete=delivered == expected)
        with np.load(directory/row['complete_trace']['path'], allow_pickle=False) as z:
            np.testing.assert_array_equal(data[:, :2], z['trace'][:, :2])
            np.testing.assert_allclose(data[:, 2], z['trace'][:, 2], rtol=0, atol=1e-12)
    return total


def run():
    p = verify(); destination = OUT/'runs'; destination.mkdir(exist_ok=False)
    ledger = []; finished = reserved = 0; raw = directory = current = None
    try:
        for current in jobs(p):
            verify(); raw = None; cases = current['cases']; model = normalization = before = None
            key = f'{current["label"]}/batch_{current["batch"]}'
            directory = destination/key; directory.mkdir(parents=True, exist_ok=False)
            if 'model' in current:
                prefix = Path(current['model']['prefix']); model = PPO.load(str(prefix.with_suffix('.zip')), device='cuda')
                normalization = str(prefix.with_suffix('.pkl'))
                before = (model.num_timesteps, model._n_updates, weight_digest(model))
            factory = probe.rec.old.evaluator.raw_env
            def wrapped(group, mode):
                nonlocal raw
                assert group == cases and mode == current['mode']
                raw = probe.instrument(group, mode, directory); wait = raw.step_wait; batches = 0
                def observed():
                    nonlocal batches
                    result = wait(); batches += 1
                    if batches % 25 == 0:
                        write(OUT/'progress.json', dict(status='running', job=key, reserved_evaluations=reserved,
                            completed_evaluations=finished, first_episodes_saved=len(raw._force_frozen),
                            first_physics_steps=int(raw._complete_buffers[2].numpy()[:, 0].sum())))
                    return result
                raw.step_wait = observed; return raw
            reserved += len(cases)
            write(OUT/'progress.json', dict(status='starting', job=key, reserved_evaluations=reserved, completed_evaluations=finished))
            probe.rec.old.evaluator.raw_env = wrapped
            try:
                result = probe.rec.old.evaluator.evaluate(cases, current['arm'], model=model,
                    normalization=normalization, classical=current.get('classical'))
            finally:
                probe.rec.old.evaluator.raw_env = factory
            if model is not None:
                assert before == (model.num_timesteps, model._n_updates, weight_digest(model))
            result = json.loads(json.dumps(result, allow_nan=False))
            write(directory/'result.json', result); finished += len(result['runs'])
            # Reuse schema/physical checks, without pretending this new flat assay has an old paired trajectory.
            checked = previous.prior.prior.check_files(directory, cases, {r['seed']: r for r in result['runs']})
            checked.pop('original_label_differences'); assert logs(directory, cases) == checked['physics_steps']
            ledger.append(dict(label=current['label'], batch=current['batch'], indices=current['indices'],
                path=str((directory/'result.json').relative_to(OUT)), sha256=sha(directory/'result.json'),
                model_immutable=model is None or before == (model.num_timesteps, model._n_updates, weight_digest(model)),
                normalization_immutable_checked_by_evaluator=True, original_bank_pair_not_applicable=True,
                contracts={n: sha(directory/n) for n in ('broad_contract.json', 'role_contract.json',
                    'gyro_contract.json', 'phase_contract.json', 'force_contract.json')}, **checked))
            write(OUT/'completed_jobs.json', dict(completed_evaluations=finished, records=ledger)); verify()
            print('COMPLETED', finished, key, 'success', result['summary']['success_count'], 'physical', result['physical'],
                  'design', result['design'], 'delivery', sum(r['force_delivery']['complete'] for r in result['runs']), flush=True)
        assert finished == reserved == 225
        write(OUT/'completion.json', dict(verified=True, completed_new_evaluations=225, pulse_evaluations=200,
            zero_evaluations=25, registered_case_IDs=45, records=ledger,
            runner_contract_sha256=sha(OUT/'runner_contract.json'), training_updates=0,
            new_first_physics_steps=sum(j['physics_steps'] for j in ledger), new_contact_records=sum(j['contacts'] for j in ledger),
            model_and_normalization_immutable=True,
            scope='Auxiliary flat fixed-stimulus transfer assay,45 shared operational cases/5 controllers; not independent generalization, matched new training or terrain-equivalent recovery. First counts omit surplus autoreset work.'))
        write(OUT/'progress.json', dict(status='complete', completed_new_evaluations=225))
    except BaseException as error:
        partial = preserve(raw, directory, current['cases']) if raw is not None else None
        write(OUT/'interruption.json', dict(error=repr(error), finished_evaluations=finished, reserved_evaluations=reserved,
            current_job=current, checked_jobs=ledger, partial=partial, silently_resumable=False))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['freeze', 'run'])
    globals()[parser.parse_args().command]()
