"""CPU receipt of the sealed PPO-v1 Cartesian streams and actual wheel/Nom evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import review_cartesian_light_map as frozen

ROOT, BASE = frozen.ROOT, frozen.BASE
RUNTIME = BASE / 'cartesian_ppo_interface_v1/runtime'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def envelope(cart, guard, nominal, maxima):
    """Nom recorder precedes candidate/guard and physics: all six speeds are contemporaneous."""
    assert nominal.shape == (len(cart), 5) and np.isfinite(nominal).all()
    np.testing.assert_array_equal(nominal[:, 0], 1)
    assert np.all(np.isin(nominal[:, 3], (0, 1)))
    np.testing.assert_array_equal(nominal[:, 4], 0)
    speed = np.column_stack((guard[:, 54:58], nominal[:, 1:3]))
    bound = frozen.limits(speed)
    error = float(abs(bound - cart[:, 27:33]).max()) if len(cart) else 0.
    maxima['six_speed_limits'] = max(maxima.get('six_speed_limits', 0.), error)
    np.testing.assert_allclose(cart[:, 27:33], bound, rtol=0, atol=frozen.TOL['hip_limits'])
    # Accepted mode1 projects the current residual; an old base-invalid flag does not force lambda0.
    expected = frozen.projection(guard[:, 2:8], cart[:, 15:21], bound)
    np.testing.assert_allclose(cart[:, 14], expected, rtol=0, atol=frozen.TOL['projection'])
    flagged = nominal[:, 3] == 1
    nonzero = np.any(cart[:, 15:21] != 0, axis=1)
    eligible = flagged & nonzero & (expected > frozen.TOL['projection'])
    assert np.all(cart[eligible, 14] > 0), 'Historical Nom flag wrongly rejected a feasible residual'
    assert np.all(abs(cart[:, 7:13]) <= 1 + 1e-12)
    assert np.all(abs(cart[:, 9] + cart[:, 10]) <= .1 + 1e-12)
    np.testing.assert_array_equal(cart[:, 11] + cart[:, 12], 0)
    np.testing.assert_array_equal(np.clip(guard[:, 2:8] + cart[:, 14, None] * cart[:, 15:21], -bound, bound).astype(np.float32), cart[:, 21:27].astype(np.float32))
    np.testing.assert_array_equal(np.clip(guard[:, 14:20] + guard[:, 27, None] * guard[:, 8:14], -bound, bound).astype(np.float32), guard[:, 20:26].astype(np.float32))
    flag_error = float(abs(cart[flagged, 14] - expected[flagged]).max()) if flagged.any() else 0.
    maxima['flag1_projection'] = max(maxima.get('flag1_projection', 0.), flag_error)
    return dict(flag1=int(flagged.sum()), flag1_nonzero=int((flagged & nonzero).sum()),
                flag1_feasible_accepted=int(eligible.sum()), projection_errors=0)


def packet(record, mode, latent, seen, reference, maxima):
    last, latch, count, withdrawn = frozen.packet(record, mode, latent, seen, reference, maxima)
    nominal = record['nominal_inputs'];keep = record['trace'][:, :, 0] == 1
    assert nominal.shape == (40, 10, 5) and np.isfinite(nominal).all()
    np.testing.assert_array_equal(nominal[:, :, 0] == 1, keep)
    exposure = envelope(record['trace'][keep], record['guard'][keep], nominal[keep], maxima)
    return last, latch, count, withdrawn, exposure


def self_check():
    frozen.self_check()
    cart = np.zeros((1, 38));guard = np.zeros((1, 62));nominal = np.zeros((1, 5))
    nominal[:, 0] = nominal[:, 3] = 1;cart[:, 14] = 1;cart[:, 27:33] = frozen.limits(np.zeros((1, 6)))
    cart[:, 15:21] = .01;guard[:, 8:14] = .01;guard[:, 26:28] = 1
    cart[:, 21:27] = .01;guard[:, 20:26] = .01
    assert envelope(cart, guard, nominal, {})['flag1_feasible_accepted'] == 1
    for field in ('wheel_speed', 'bounds', 'flag', 'projection'):
        c, n = cart.copy(), nominal.copy()
        if field == 'wheel_speed':n[:, 1] = 1000
        elif field == 'bounds':c[:, 31] -= .1
        elif field == 'flag':n[:, 3] = 2
        else:c[:, 14] = 0
        try:envelope(c, guard, n, {})
        except AssertionError:pass
        else:raise AssertionError('Corrupt evidence accepted: ' + field)


def review(runtime_path):
    runtime = Path(runtime_path).resolve()
    if runtime != RUNTIME.resolve():
        raise ValueError('Only the sealed registered PPO-v1 training runtime is in scope')
    self_check();out = runtime.parent
    receipt = json.loads((out / 'execution_receipt.json').read_text())
    admission = json.loads((out / 'source_admission.json').read_text())
    proposal = json.loads((out / 'proposal.json').read_text())
    assert receipt['completion_sha256'] == sha(runtime / 'completion.json')
    assert receipt['source_admission_sha256'] == sha(out / 'source_admission.json')
    assert receipt['proposal_sha256'] == sha(out / 'proposal.json')
    assert len(admission['source_sha256']) == 114
    assert all(sha(ROOT / name) == digest for name,digest in admission['source_sha256'].items())
    assert all(sha(name) == digest for name,digest in admission['runtime']['installed_sha256'].items())
    assert np.__version__ == admission['runtime']['versions']['numpy']
    assert proposal['bank_seed'] == 33531 and proposal['worlds'] == 10
    assert all(len(rows) == 10 for rows in proposal['training_banks']['33531'].values())
    completion = json.loads((runtime / 'completion.json').read_text());reports = {}
    assert completion['optimizer_updates'] == 0 and not (runtime / 'failure.json').exists()
    def load(path):
        assert receipt['outputs_sha256'][str(path.relative_to(ROOT))] == sha(path)
        with np.load(path, allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    fields = ('trace','guard','parking','actor','action','reference','latent','shadow','terminal_packet',
              'done','public_after','latent_after','shadow_after','reference_after','nominal_inputs')
    for mode,arm in enumerate(('CartLive3','CartReset3')):
        directory = runtime / arm;initial = load(directory / 'initial.npz')
        zero_common = load(directory / 'zero_prefix.npz');zero = load(directory / 'zero_before_autoreset.npz')
        zero.update(actor=zero_common['actor'], action=np.zeros((10, 6), np.float32))
        reference = frozen.geometry(initial['public'][:, 12:16].reshape(10, 2, 2))[0].reshape(10, 4)
        maxima = {};latent = np.zeros((10, 3));seen = np.zeros(10, bool)
        _, _, prefix, _, zero_flags = packet(zero, mode, latent, seen, reference, maxima)
        assert zero_flags['flag1'] == 0
        stored_reference = zero['reference'].copy();total = withdrawn = resets = packets = 0
        exposure = dict(flag1=0,flag1_nonzero=0,flag1_feasible_accepted=0,projection_errors=0)
        metadata_path = directory / 'episodes.json'
        assert receipt['outputs_sha256'][str(metadata_path.relative_to(ROOT))] == sha(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        for block in metadata['blocks']:
            path = directory / block['path'];assert sha(path) == block['sha256']
            assert receipt['outputs_sha256'][str(path.relative_to(ROOT))] == block['sha256']
            with np.load(path, allow_pickle=False) as z:batch = {name:z[name] for name in fields}
            assert all(len(values) == block['packets'] for values in batch.values())
            for k in range(block['packets']):
                r = {name:values[k] for name,values in batch.items()}
                np.testing.assert_array_equal(r['reference'], stored_reference)
                latent, seen, count, park_count, flags = packet(r, mode, latent, seen, reference, maxima)
                total += count;withdrawn += park_count;done = r['done'];resets += int(done.sum());packets += 1
                for name in exposure:exposure[name] += flags[name]
                for name in ('latent','shadow','reference'):
                    np.testing.assert_array_equal(r[name + '_after'][~done], r[name][~done])
                np.testing.assert_array_equal(r['latent_after'][done], 0)
                np.testing.assert_array_equal(r['shadow_after'][done, 16:22], 0)
                np.testing.assert_array_equal(r['public_after'][done, 32:38], 0)
                np.testing.assert_array_equal(r['public_after'][~done, 32:38], frozen.pairs(latent[~done]).astype(np.float32))
                if done.any():
                    reset_reference = frozen.geometry(r['public_after'][done, 12:16].reshape(-1, 2, 2))[0].reshape(-1, 4)
                    frozen.close('geometry', r['reference_after'][done], reset_reference, maxima)
                    reference[done] = reset_reference;latent[done] = 0;seen[done] = False
                stored_reference = r['reference_after'].copy()
        final = load(directory / 'final_cartesian.npz')
        np.testing.assert_array_equal(final['latent'], latent);np.testing.assert_array_equal(final['shadow'], r['shadow_after'])
        np.testing.assert_array_equal(final['reference'], stored_reference);frozen.close('geometry',final['reference'],reference,maxima)
        assert packets == completion['reports'][arm]['packets'] == 1200
        assert prefix + total == receipt['valid_guard_steps_including_zero'][arm]
        reports[arm] = dict(prefix_substeps=prefix,rollout_substeps=total,withdrawn_substeps=withdrawn,
            reset_rows=resets,nominal_flags=exposure,max_abs_errors=maxima)
    assert sum(r['prefix_substeps'] + r['rollout_substeps'] for r in reports.values()) == 959707
    return dict(verified=True,reports=reports,tolerances={**frozen.TOL,'six_speed_limits':frozen.TOL['hip_limits']},
        source_manifest_sha256=sha(out / 'source_admission.json'),frozen_mapping_sha256=sha(Path(frozen.__file__)),
        source_files_checked=114,metadata_scope='Only sealed PPO-v1 registered33531 training records; no heldout/future final data.',
        scope='Cold-zero and all recorded Cart mappings, complete six-speed limits, actual Nom flags, parking/latent and Cart3 actual post/reset/final.',
        nominal_flag_condition='diag13=1 is historical invalid-base exposure, not a residual rejection gate; acceptedmode1 lambda is reconstructed with all six actual speed bounds.',
        flag1_acceptance_count_condition='diag13==1, nonzero motor residual and reconstructed lambda>the frozen1e-7 projection tolerance.',
        limits=['Nom flag producer correctness is source/route-bound; unclipped Nom base torque is not recorded per substep for recomputing diag13.',
                'Command/history/post-bank/RMS/PPO/GAE qualification belongs to the independent aggregate receiver, not this helper.'],
        new_physics_steps=0,new_FD=0,new_model_forward_rows=0,new_learning_samples=0,full_interface_qualified=False)


if __name__ == '__main__':
    print(json.dumps(review(RUNTIME),ensure_ascii=False))
