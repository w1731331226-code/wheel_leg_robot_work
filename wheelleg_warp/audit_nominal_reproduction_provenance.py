"""Locate saved baseline divergence without rerunning physics or changing gates."""
import json
from collections import Counter
import numpy as np
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

BASE = ROOT / 'wheelleg_warp/results/paper_recovery_20261004'
OUT = BASE / 'continuous_nominal_task_pair_v1'


def first_difference(old, new):
    assert old.ndim == new.ndim == 2 and old.shape[1] == new.shape[1] == 39
    assert np.isfinite(old).all() and np.isfinite(new).all()
    length = min(len(old), len(new))
    where = np.flatnonzero(np.any(old[:length] != new[:length], axis=1))
    if not len(where):
        return None
    i = int(where[0])
    fields = np.flatnonzero(old[i] != new[i])
    return dict(physical_step=i + 1, fields=fields.tolist(),
                groups=sorted({'qpos' if j < 17 else 'qvel' if j < 33 else 'effort' for j in fields}),
                old_values=old[i, fields].tolist(), new_values=new[i, fields].tolist(),
                max_abs_delta=float(abs(old[i] - new[i]).max()))


def run():
    target = OUT / 'round294_reproduction_provenance.json'
    assert not target.exists()
    # Tiny independent check: the earliest difference is preserved, even at 1 ULP.
    a = np.zeros((3, 39)); b = a.copy(); b[1, 20] = np.nextafter(0., 1.)
    assert first_difference(a, b)['physical_step'] == 2
    assert first_difference(a, a) is None
    p = json.loads((OUT / 'proposal.json').read_text())
    prior = BASE / 'execution_input_history_v1/study_contract.json'
    contract = json.loads(prior.read_text())
    declared = contract['source_sha256']
    assert len(declared) == 55
    assert all(sha(ROOT / path) == digest for path, digest in declared.items())
    common = set(declared) & set(p['source_sha256'])
    assert len(common) == 13
    assert all(declared[path] == p['source_sha256'][path] for path in common)
    c = json.loads((OUT / 'completion.json').read_text())
    records = []
    for archived in p['old_archived_baseline']:
        entry = next(r for r in c['records'] if r['arm'] == 'old_B0' and
                     (r['panel'], r['batch']) == (archived['panel'], archived['batch']))
        old_file, new_file = ROOT / archived['path'], OUT / entry['path']
        assert sha(old_file) == archived['sha256'] and sha(new_file) == entry['sha256']
        old_rows = json.loads(old_file.read_text())['runs']
        new_rows = json.loads(new_file.read_text())['runs']
        assert len(old_rows) == len(new_rows) == len(entry['indices'])
        for old, new in zip(old_rows, new_rows):
            assert old['seed'] == new['seed'] and old['scenario'] == new['scenario']
            paths = [f.parent / row['complete_trace']['path'] for f, row in ((old_file, old), (new_file, new))]
            for file, row in zip(paths, (old, new)):
                assert sha(file) == row['complete_trace']['sha256']
            with np.load(paths[0], allow_pickle=False) as x, np.load(paths[1], allow_pickle=False) as y:
                assert len(x['pre']) == len(x['post']) == old['physical_steps']
                assert len(y['pre']) == len(y['post']) == new['physical_steps']
                pre = first_difference(x['pre'], y['pre'])
                post = first_difference(x['post'], y['post'])
                initial_equal = bool(np.array_equal(x['pre'][0], y['pre'][0]))
                prefix_equal = pre is None and post is None
            records.append(dict(seed=new['seed'], panel=entry['panel'], batch=entry['batch'],
                old_dense_sha256=old['complete_trace']['sha256'], new_dense_sha256=new['complete_trace']['sha256'],
                initial_recorded_qpos_qvel_ctrl_equal=initial_equal,
                first_pre_difference=pre, first_post_difference=post,
                common_prefix_equal=prefix_equal,
                complete_dense_equal=prefix_equal and old['physical_steps'] == new['physical_steps'],
                old_steps=old['physical_steps'], new_steps=new['physical_steps']))
        print('REPRODUCTION', entry['panel'], entry['batch'], len(records), flush=True)
    assert len(records) == 164 and len({r['seed'] for r in records}) == 164
    summaries = {}
    for panel in ('regular', 'controlled', 'legacy'):
        rows = [r for r in records if r['panel'] == panel]
        summaries[panel] = dict(cases=len(rows), initial_equal=sum(r['initial_recorded_qpos_qvel_ctrl_equal'] for r in rows),
            dense_equal=sum(r['complete_dense_equal'] for r in rows),
            first_pre_step_histogram=dict(Counter(str(r['first_pre_difference']['physical_step']) if r['first_pre_difference'] else 'none' for r in rows)),
            first_post_step_histogram=dict(Counter(str(r['first_post_difference']['physical_step']) if r['first_post_difference'] else 'none' for r in rows)))
    atomic_json(target, dict(verified=True, round=294, records=records, summary=summaries,
        source_sha256=sha(__file__), archived_contract_sha256=sha(prior),
        task_proposal_sha256=sha(OUT / 'proposal.json'), task_completion_sha256=sha(OUT / 'completion.json'),
        archived_declared_sources_checked=55, common_source_hashes_equal=13,
        unique_cause_established=False, new_simulation=0, new_optimizer=0, new_training_samples=0,
        limits='Only recorded qpos/qvel/ctrl and post effort are compared. Hidden solver warm starts, contact ordering, compiled kernels and historical runtime versions are not proven equal by source hashes. Initial-state equality does not prove identical simulator state. No numerical tolerance or original success gate changed.',
        next='295 direction review must include these onset records and the closed information branch. Isolate runtime/solver provenance before causal method comparisons; do not restart the full164 baseline queue or claim source equality proves deterministic replay.'))
    print('PASS294 full164 source/initial/divergence audit', summaries, flush=True)


if __name__ == '__main__':
    run()
