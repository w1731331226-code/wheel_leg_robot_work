"""Recalculate archived reset comparisons and continuous rejection; no simulation."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def errors(predicted, actual, nq, nv):
    return dict(zip(('qpos', 'qvel', 'command'), (
        float(abs(predicted[:, :, s] - actual[:, :, s]).max())
        for s in (slice(0, nq), slice(nq, nq + nv), slice(-12, -6)))))


def check():
    previous = OUT.parent / 'thirteenth_round_feedback_checked_20261002'
    state = np.load(OUT / 'current_pre.npz', allow_pickle=False)
    nq, nv = state['q'].shape[1], state['v'].shape[1]
    frozen = json.loads((OUT / 'reset_check.json').read_text())
    predictions = np.load(OUT / 'reset_predictions.npz', allow_pickle=False)
    truth = np.load(previous / 'failed_pair.npz', allow_pickle=False)['actual']
    gates = dict(qpos=2e-6, qvel=1e-3, command=1e-5)
    assert len(frozen['rows']) == 33
    for row, prediction in zip(frozen['rows'], [predictions['fresh'], *predictions['history']]):
        measured = errors(prediction, truth, nq, nv)
        assert all(row[k] == v for k, v in measured.items())
        assert row['original_pass'] == all(measured[k] <= gates[k] for k in gates)
    counts = {mode: sum(r['original_pass'] for r in frozen['rows'] if r['mode'] == mode)
              for mode in ('fresh', 'reused', 'public_reset')}
    assert counts == dict(fresh=1, reused=13, public_reset=16)
    run = json.loads((OUT / 'verification.json').read_text())
    assert run['original_pairing_gates'] == gates
    assert not run['completed'] and not run['default_promoted'] and not run['learning_performed']
    assert len(run['decisions']) == len(run['pairing_errors']) == 96
    assert all(all(row[k] <= gates[k] for k in gates) for row in run['pairing_errors'][:-1])
    pair = np.load(OUT / 'failed_pair.npz', allow_pickle=False)
    measured = errors(pair['predicted'], pair['actual'], nq, nv)
    assert run['failure']['step'] == 95 and measured['command'] > gates['command']
    assert all(run['failure'][k] == v for k, v in measured.items())
    actual = np.load(OUT / 'actual.npz', allow_pickle=False)
    requests = actual['requests']
    assert actual['trace'].shape[0] == 960 and requests.shape == (96, 2, 6)
    np.testing.assert_array_equal(requests, [r['request'] for r in run['decisions']])
    delta = np.diff(np.concatenate([np.zeros_like(requests[:1]), requests]), axis=0)
    assert abs(requests).max() <= 1 and abs(delta).sum(axis=2).max() <= .1 + 1e-12
    for record in (frozen, run):
        for field in ('input_sha256', 'source_sha256'):
            for name, digest in record.get(field, {}).items():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    print('PASS: frozen reset 16/16; continuous window 96 rejected; original gates/domain retained; no admission.')


if __name__ == '__main__':
    check()
