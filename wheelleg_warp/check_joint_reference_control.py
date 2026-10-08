"""100 saved-fixture GPU control queries; no environment or physics graph."""
import ast
import json
from pathlib import Path
import numpy as np
import warp as wp
import reference_role_control as old
import joint_reference_control as candidate
import joint_reference_prepare as prep
from joint_reference_mapping import select, LOW
from native.controller import D, sim
import model_lqr as ml
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'
FIXTURE = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_map_v1/queries_old.npz'


def execute(kernel, data, ids, ref, targets):
    n = len(data['q'])
    def array(value, dtype=D):
        return wp.array(value, dtype=dtype, device='cuda:0')
    args = [array(data['q'], wp.float32), array(data['v'], wp.float32), array(data['sensor'], wp.float32),
            array(targets, wp.float32), array(data['command']), array(data['active'], wp.int32),
            array(data['memory_before']), array(ids, wp.int32), array(data['heights']),
            array(data['gains']), array(data['feed']), array(data['angles']), array(ref),
            array([.4, 2., .24, .3]), array(np.zeros((n, 6)), wp.float32),
            array(np.zeros_like(data['diag'])), 0, 0, array(np.ones((n, 6))),
            array(data['nominal_correction'])]
    before = {str(i): a.numpy().copy() for i, a in enumerate(args) if isinstance(a, wp.array)}
    wp.launch(kernel, n, args, block_dim=32)
    after = {str(i): a.numpy().copy() for i, a in enumerate(args) if isinstance(a, wp.array)}
    for key in before:
        if int(key) not in (6, 14, 15):
            np.testing.assert_array_equal(after[key], before[key])
    for key in ('6', '14', '15'):
        assert np.isfinite(after[key]).all()
    return dict(memory_after=after['6'], ctrl=after['14'], diag=after['15'],
                reference=after['12'], targets=after['3'], memory_before=before['6'])


def run():
    assert not any((OUT / n).exists() for n in ('control_started.json', 'control_completion.json', 'control_failure.json'))
    source = Path(old.__file__).read_text(); patched = Path(candidate.__file__).read_text()
    start = patched.index('    # Change tracking targets only;')
    end = patched.index('    height=length*wp.cos(th);', start)
    restored = patched[:start] + patched[end:]
    assert restored == source and ast.dump(ast.parse(restored)) == ast.dump(ast.parse(source))
    derivation = json.loads((OUT / 'derivation.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in derivation['source_sha256'].items())
    previous = json.loads((FIXTURE.parent / 'query_completion.json').read_text())
    assert any(r['arm'] == 'old' and r['query_sha256'] == sha(FIXTURE) for r in previous['records'])
    with np.load(FIXTURE, allow_pickle=False) as z:
        data = {name: z[name].copy() for name in z.files}
    n = len(data['q']); assert n == 25
    np.testing.assert_array_equal(data['active'], 1)
    np.testing.assert_array_equal(data['targets'], 0)
    np.testing.assert_array_equal(data['base_reference'][:, 3], LOW)
    model, _ = sim.load_model(ml.XML, True)
    names = ('alphaL', 'betaL', 'alphaR', 'betaR')
    ids = [int(model.jnt_qposadr[model.joint(name).id]) for name in names]
    ids += [int(model.jnt_dofadr[model.joint(name).id]) for name in (*names, 'wheel1', 'wheel2')]
    ids += [int(model.sensor('body_gyro').adr[0])]
    files = [Path(__file__), Path(candidate.__file__), Path(prep.__file__), Path(old.__file__)]
    atomic_json(OUT / 'control_started.json', dict(round=299, static_controller_query_budget=100,
        fixture_sha256=sha(FIXTURE), derivation_sha256=sha(OUT / 'derivation.json'),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
        reversible_single_tracking_patch=True, environment_constructed=False,
        physics_graph_created=False, baseline_constructor_FD_calls=0))
    count = 0
    try:
        prep.unit('cuda:0')
        ref = data['base_reference']
        private = np.column_stack((ref, np.zeros((n, 2))))
        for label, kernel, references, targets in [
            ('old', old.control_physical_nominal, ref, data['targets']),
            ('zero', candidate.control_physical_nominal, private, data['targets'])]:
            count += n
            result = execute(kernel, data, ids, references, targets)
            np.savez_compressed(OUT / f'control_{label}.npz', **result)
            for name in ('ctrl', 'diag', 'memory_after'):
                np.testing.assert_array_equal(result[name], data[name])
            print('QUERY', label, count, 'exact saved original output', flush=True)
        am = np.where(np.arange(n) % 2 == 0, 1., -1.)
        ad = -am
        request = prep.validate(np.column_stack((am, ad, np.zeros(n))))
        base = wp.array(ref, dtype=D, device='cuda:0')
        action = wp.array(request, dtype=D, device='cuda:0')
        active = wp.array(data['active'], dtype=int, device='cuda:0')
        filtered = wp.zeros((n, 2), dtype=D, device='cuda:0')
        packet = wp.zeros((n, 18), dtype=D, device='cuda:0')
        for _ in range(40):
            wp.launch(prep.prepare, n, [base, action, active, filtered, packet])
        private = packet.numpy().copy()
        np.testing.assert_array_equal(private[:, :16], ref)
        np.testing.assert_allclose(private[:, 16:], request[:, :2]*.4, rtol=0, atol=3e-16)
        d0 = (data['diag'][:, 21] - data['diag'][:, 22]) / 2
        count += n
        result = execute(candidate.control_physical_nominal, data, ids, private, data['targets'])
        np.savez_compressed(OUT / 'control_reference.npz', **result, request=request, filtered=filtered.numpy())
        m, d, _ = select(ref[:, 2], d0, np.column_stack((private[:, 16:], np.zeros(n))))
        np.testing.assert_allclose(result['diag'][:, 21], m+d, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['diag'][:, 22], m-d, rtol=0, atol=1e-12)
        print('QUERY reference', count, 'targets match independent mapping', flush=True)
        targets = data['targets'].copy(); targets[:, 4] = .5; targets[:, 5] = -.5
        count += n
        result = execute(candidate.control_physical_nominal, data, ids,
                         np.column_stack((ref, np.zeros((n, 2)))), targets)
        np.savez_compressed(OUT / 'control_yaw.npz', **result)
        np.testing.assert_allclose(result['memory_after'][:, 20:22], [[.01, -.01]]*n, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result['diag'][:, 21:23], data['diag'][:, 21:23])
        assert count == 100
        outputs = {str(p.relative_to(OUT)): sha(p) for p in OUT.glob('control_*.npz')}
        atomic_json(OUT / 'control_completion.json', dict(verified=True, round=299,
            individual_static_controller_queries=count, outputs_sha256=outputs,
            zero_ctrl_diag_memory_exact=True, old_saved_output_reproduced_exact=True,
            reference_targets_match_CPU_mapping=True, original_reference_and_radial_anchor_preserved=True,
            reference_normalized_slew_per_substep=.01, inactive_filter_and_invalid_actions_checked=True,
            yaw_original_filter_path_checked=True, new_simulation=0, new_training_samples=0,
            optimization_calls=0, transitionFD_calls=0, model_compile_only=True,
            full_rollout_admitted=False, formal_PPO_admitted=False,
            limits='25 existing nominal fixtures/static calls; new reference target geometry and zero outputs only. No actual contact dynamics, full trajectory identity, learned advantage, CPU-GPU task pairing or fair comparator implementation.300deepreview required.'))
        print('PASS299 100GPU static queries; no physics/FD/PPO', flush=True)
    except BaseException as error:
        atomic_json(OUT / 'control_failure.json', dict(error=repr(error), attempted_controller_queries=count,
            implicit_retry=False, new_simulation=0))
        raise


if __name__ == '__main__':
    run()
