"""Independent saved 70-query review; no controller construction or rerun."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

sys.path.insert(0, str(ROOT / 'wheelleg_ppo/tools'))
import model_lqr as ml
import wheelleg_sim as sim

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/motion_balanced_nominal_v1'


def run():
    assert not (OUT / 'direction_review.json').exists()
    p = json.loads((OUT / 'proposal.json').read_text())
    c = json.loads((OUT / 'completion.json').read_text())
    k = json.loads((OUT / 'query_contract.json').read_text())
    assert c['verified'] and c['individual_static_queries'] == 70
    assert c['source_contract_sha256'] == sha(OUT / 'query_contract.json')
    assert k['proposal_sha256'] == sha(OUT / 'proposal.json')
    assert all(sha(ROOT / n) == h for n, h in k['source_sha256'].items())
    original = (ROOT / 'wheelleg_warp/reference_role_control.py').read_text()
    source = (ROOT / 'wheelleg_warp/motion_balanced_control.py').read_text()
    # Remove only the two specifically allowed branches; every other AST node must match.
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'control_step')
    branches = [n for n in fn.body if isinstance(n, ast.If) and
                ast.unparse(n.test) == 'reference.shape[1] == 21 and reference[w, 20] == D(1) and (cmd != D(0))']
    assert len(branches) == 2
    assert [[ast.unparse(x) for x in n.body] for n in branches] == [
        ['theta_eq = reference[w, 19]'],
        ['wheel = reference[w, 16]', 'hub = reference[w, 17]', 'support = reference[w, 18]']]
    fn.body = [n for n in fn.body if n not in branches]
    assert ast.dump(tree) == ast.dump(ast.parse(original))
    model, _ = sim.load_model(ml.XML, True)
    dofs = [model.jnt_dofadr[model.joint(n).id] for n in
            ('alphaL', 'betaL', 'alphaR', 'betaR', 'wheel1', 'wheel2')]
    rows, arrays = [], {}
    for j in range(3):
        for arm in ('old', 'candidate'):
            path = OUT / f'paired_integral_{j}_{arm}.npz'
            with np.load(path, allow_pickle=False) as z:
                x = {n: z[n].copy() for n in z.files}
            arrays[j, arm] = x
            records = [r for r in c['records'] if r['path'] == path.name]
            assert len(records) == (15 if j == 0 else 10)
            assert all(r['sha256'] == sha(path) for r in records)
            np.testing.assert_array_equal(x['q'], x['original_q'].astype(np.float32))
            np.testing.assert_array_equal(x['v'], x['original_v'].astype(np.float32))
            np.testing.assert_array_equal(x['ctrl_minus_required'], x['ctrl'].astype(float)-x['required_ctrl'])
            np.testing.assert_array_equal(x['clock'], 0.)
            np.testing.assert_array_equal(x['targets'], 0.)
            np.testing.assert_array_equal(x['nominal_correction'], 0.)
            for r in records:
                i = r['world']
                assert x['active'][i] == 1 and r['height'] == k['references'][i]['height']
                assert x['command'][i] == r['speed'] == k['references'][i]['speed']
                assert r['state11'] == x['memory_before'][i, 11]
                np.testing.assert_array_equal(x['diag'][i, 14], 0.)
                speeds = x['v'][i, dofs].astype(float)
                clipped = np.array([sim.hw.torque_limit(float(t), float(v), n < 4, 0., .0005)[0]
                                    for n, (t, v) in enumerate(zip(x['ctrl'][i], speeds))])
                np.testing.assert_allclose(clipped, x['ctrl'][i], rtol=0, atol=1e-6)
                for key, value in (
                        ('maximum_command_gap_Nm', abs(x['ctrl_minus_required'][i]).max()),
                        ('wheel_gap_Nm', abs(x['ctrl_minus_required'][i, -2:]).max()),
                        ('raw_nominal_gap_Nm', abs(x['diag'][i, 15:21]-x['required_ctrl'][i]).max())):
                    assert r[key] == float(value)
                rows.append(r)
        old, new = arrays[j, 'old'], arrays[j, 'candidate']
        for name in ('q', 'v', 'sensor', 'command', 'environment_state', 'clock', 'gains',
                     'feed', 'angles', 'heights', 'active', 'targets', 'nominal_correction',
                     'memory_before', 'base_reference', 'required_ctrl', 'phase'):
            np.testing.assert_array_equal(old[name], new[name])
        np.testing.assert_array_equal(old['reference'], new['reference'][:, :16])
        assert np.all(new['reference'][:10, 20] == 1.)
    for name in ('ctrl', 'diag', 'memory_after', 'role', 'phase'):
        np.testing.assert_array_equal(arrays[0, 'old'][name][10:],
                                      arrays[0, 'candidate'][name][10:])
    np.testing.assert_array_equal(arrays[0, 'candidate']['reference'][10:, 16:], 0.)
    summary = {}
    for arm in ('old', 'candidate'):
        moving = [r for r in rows if r['arm'] == arm and r['speed'] != 0.]
        summary[arm] = dict(max_gap_Nm=max(r['maximum_command_gap_Nm'] for r in moving),
                            max_wheel_gap_Nm=max(r['wheel_gap_Nm'] for r in moving))
    assert len(rows) == 70 and len(k['baseline_transitionFD_calls']) == 10
    atomic_json(OUT / 'direction_review.json', dict(
        verified=True, round=270, reviewer_sha256=sha(__file__),
        completion_sha256=sha(OUT / 'completion.json'), source_AST_only_two_branches=True,
        unchanged_pair_inputs=True, zero_speed_exact=True, physical_torque_bounds_passed=True,
        summary=summary, additional_queries=0, integration_steps=0, transitionFD_calls=0,
        new_training_samples=0, production_admitted=False,
        decision='Close static screen; proceed only finite nominal dynamic CPU/GPU qualification. Do not repeat static point sweeps or launch PPO.',
        rationale='Feed/reference consistency resolves the recorded static mismatch without changing gains or weakening baseline. Meaningful engineering progress, but static residual improvement is neither stability nor novel RL contribution.',
        required_next='Same nominal ten moving references, two arms, two physical backends, fixed commands and zero Actor, 2000 steps per trajectory: 40 trajectories/80000 physical steps maximum. Preserve all failures and internal/filter/guard/torque/contact traces. No tuning or automatic retries.',
        limits='Exact nominal height/speed nodes only. Full task requires a separately qualified continuous command/reference bridge, start/stop/asymmetry/jump retention and164 tasks. No new formal5 training/freshOOD/novelty/manuscript qualification yet.'))
    print('PASS270 independent70/source/bounds/zero-speed; close static screen; finite dynamics next', flush=True)


if __name__ == '__main__':
    run()
