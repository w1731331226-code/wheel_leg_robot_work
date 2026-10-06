"""Inspect the saved broad-qualification failure; never integrate or train."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'wheelleg_warp'), str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.terrain import HeightTerrainScenario, model
from review_yaw_sector import sha
from dashboard.live_env import atomic_json

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/floor_broad_qualification_v1'
NAMES = ('alphaL', 'betaL', 'alphaR', 'betaR')


def intervals(mask):
    mask = np.asarray(mask)
    assert mask.ndim == 1 and mask.dtype == bool
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1))


def run():
    assert intervals(np.array([True, False, True, True])) == [(0, 0), (2, 3)]
    assert intervals(np.zeros(3, dtype=bool)) == []
    proposal = json.loads((OUT/'proposal.json').read_text())
    completion = json.loads((OUT/'completion.json').read_text())
    paired = json.loads((OUT/'round188_broad_pair_review.json').read_text())
    contract = json.loads((OUT/'source_contract.json').read_text())
    assert paired['verified'] and not paired['overall_engineering_qualification']
    assert sha(OUT/'completion.json') == paired['completion_sha256']
    assert all(sha(ROOT/p) == s for p, s in contract['source_sha256'].items())
    inputs = {str((OUT/n).relative_to(ROOT)): sha(OUT/n) for n in
              ('proposal.json', 'completion.json', 'source_contract.json', 'round188_broad_pair_review.json')}
    records = {}
    for law in ('B0', 'B1-route'):
        reference = proposal['original_development_references'][law+'/regular']
        oldfile = ROOT/reference['path']
        assert sha(oldfile) == reference['sha256']
        old = next(r for r in json.loads(oldfile.read_text())['runs'] if r['seed'] == 6300085)
        inputs[str(oldfile.relative_to(ROOT))] = sha(oldfile)
        jobs = [j for j in completion['records'] if j['panel'] == 'regular'
                and j['arm'] == 'floor_only' and j['law'] == law and 85 in j['indices']]
        assert len(jobs) == 1
        result = OUT/jobs[0]['path']; assert sha(result) == jobs[0]['sha256']
        row = next(r for r in json.loads(result.read_text())['runs'] if r['seed'] == 6300085)
        assert row['scenario'] == old['scenario'] and old['success'] and not row['success']
        inputs[str(result.relative_to(ROOT))] = sha(result)
        paths = [result.parent/row[k]['path'] for k in ('complete_trace', 'role_trace')]
        for path, kind in zip(paths, ('complete_trace', 'role_trace')):
            assert sha(path) == row[kind]['sha256']; inputs[str(path.relative_to(ROOT))] = sha(path)
        with np.load(paths[0], allow_pickle=False) as z:
            trace, pre, post = z['trace'], z['pre'], z['post']
            columns = {str(k): i for i, k in enumerate(z['columns'])}
            sizes = z['state_sizes']
        with np.load(paths[1], allow_pickle=False) as z:
            role = z['trace']; rc = {str(k): i for i, k in enumerate(z['columns'])}
        # CPU compilation only supplies authoritative joint indices/ranges; no forward/step.
        cpu = model(HeightTerrainScenario(**row['scenario']))
        assert sizes.tolist() == [cpu.nq, cpu.nv, cpu.nu]
        ids = [int(cpu.joint(n).qposadr[0]) for n in NAMES]
        n = row['physical_steps']; assert len(trace) == len(role) == len(post) == n
        assert np.isfinite(pre).all() and np.isfinite(post).all() and np.isfinite(role).all()
        np.testing.assert_array_equal(trace[:, columns['step']], np.arange(1, n+1))
        np.testing.assert_array_equal(role[:, rc['step']], trace[:, columns['step']])
        np.testing.assert_array_equal(pre[1:, :cpu.nq+cpu.nv], post[:-1, :cpu.nq+cpu.nv])
        joint_margins = 1.4-np.abs(post[:, ids]); margins = joint_margins.min(axis=1)
        np.testing.assert_allclose(margins, trace[:, columns['design_margin']], atol=1e-12, rtol=0)
        assert margins.min() == row['min_active_design_margin_rad']
        k, j = np.unravel_index(joint_margins.argmin(), joint_margins.shape)
        bad = margins < 0; spans = intervals(bad); assert spans
        np.testing.assert_array_equal(role[:, rc['guard_lower']], .115)
        assert not role[:, [rc['guard_requested'], rc['guard_applied']]].any()
        target_load = trace[:, columns['left_target_normal_N']] + trace[:, columns['right_target_normal_N']]
        loaded = np.flatnonzero(target_load > 1e-6); assert len(loaded)
        def event(i):
            values = {name: float(trace[i, columns[name]]) for name in
                      ('post_s', 'body_x', 'speed_command', 'post_body_vx', 'left_leg_m',
                       'right_leg_m', 'left_reference', 'right_reference', 'roll_offset_selected')}
            values.update(step=int(i+1), active_joint_rad=dict(zip(NAMES, post[i, ids].tolist())),
                          commanded_hip_Nm=pre[i, cpu.nq+cpu.nv:cpu.nq+cpu.nv+4].tolist(),
                          target_normal_N=float(target_load[i]))
            return values
        records[law] = dict(
            scenario=row['scenario'], joint_qpos_indices=dict(zip(NAMES, ids)),
            physical_joint_ranges_rad={name: cpu.jnt_range[cpu.joint(name).id].tolist() for name in NAMES},
            original_summary_only={name: old[name] for name in ('success', 'physical_safety_passed',
                'design_joint_passed', 'min_active_design_margin_rad', 'min_leg_m', 'max_applied_radial_guard_N')},
            candidate_physical_pass=row['physical_safety_passed'], candidate_design_pass=row['design_joint_passed'],
            worst_joint=NAMES[j], worst_margin_rad=float(margins[k]),
            worst_excess_deg=float(np.rad2deg(-margins[k])),
            excess_in_float32_ULPs=float(-margins[k]/np.spacing(np.float32(1.4))),
            violating_substeps=int(bad.sum()), sampled_violation_duration_s=float(bad.sum()*.0005),
            per_joint_violating_substeps=dict(zip(NAMES, (joint_margins < 0).sum(axis=0).tolist())),
            spans=[dict(first_step=int(a+1), last_step=int(b+1), first_post_s=float((a+1)*.0005),
                        last_post_s=float((b+1)*.0005), samples=int(b-a+1)) for a, b in spans],
            arrival_s=row['arrival_s'], first_target_load_post_s=float((loaded[0]+1)*.0005),
            last_target_load_post_s=float((loaded[-1]+1)*.0005),
            target_load_during_violation_max_N=float(target_load[bad].max()),
            before_first=event(spans[0][0]-1), first=event(spans[0][0]), worst=event(k),
            recovered=event(spans[-1][1]+1), guard_requested_applied_max_N=[0., 0.])
    report = dict(verified=True, round=189, records=records, input_sha256=inputs,
        source_sha256=sha(__file__), new_physics_evaluations=0, training_updates=0,
        contract_facts='Design1.4 is evaluated on every post-integration active joint. Physical1.5 is distinct. '
            'project_leg_angle bounds a static target pose, not actual q/qdot invariance. Radial guard uses '
            'leg length/rate and torque headroom, not an explicit active-joint1.4 dynamic boundary.',
        decision='Fixedfloor remains rejected. Preserve all gates and original baseline. A static feasible '
            'reference, radial floor and torque clipping do not certify dynamic design compliance.',
        limits='Original comparator has archived summaries, not a dense trace here: no paired timing or '
            'unique causal attribution to guard is proved. ULP comparison rejects rounding-only dismissal, '
            'not all solver/discretization effects. Target-normal log is solver pre-integration; q is post. '
            'Saved samples bound sampled violation, not exact continuous crossing time. No new algorithm '
            'or learning necessity follows from this engineering counterexample.',
        next='190 direction review and verified redundancy cleanup; decide the bounded method-contract '
            'step before any new learning. Full contribution/strong comparisons/formal5seeds/independent '
            'generalization/statistics/end-to-end PPO timing/manuscript remain open.')
    atomic_json(OUT/'round189_design_crossing.json', report)
    for law, r in records.items():
        print(law, r['worst_joint'], r['worst_margin_rad'], r['spans'],
              'arrival', r['arrival_s'], 'last target load', r['last_target_load_post_s'], flush=True)
    print('PASS source/trace/recomputed margin/timing/role checks; 0 new evaluations/training', flush=True)


if __name__ == '__main__':
    run()
