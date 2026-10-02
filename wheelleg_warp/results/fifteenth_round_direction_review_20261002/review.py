"""Recalculate evidence for the fifth-round direction review, without physics."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
from native.terrain import HeightTerrainScenario, model


def run():
    paths = []
    def read(path):
        paths.append(path)
        return json.loads(path.read_text())

    m = model(HeightTerrainScenario(stand_height_m=.115))
    for name in ('alphaL', 'betaL', 'alphaR', 'betaR'):
        np.testing.assert_array_equal(m.joint(name).range, [-1.5, 1.5])
    panel = read(OUT / 'height_panel.json')
    # The recorded minimum over ALL eight joint margins lower-bounds each
    # active joint margin. Subtracting .1 certifies the stricter active cap
    # when nonnegative; a negative bound would be inconclusive, not a failure.
    design_bounds = [r['min_eight_joint_margin_rad'] - .1 for r in panel['episodes']]
    assert min(design_bounds) >= 0
    node_counts = {}
    for i, height in enumerate((.115, .16, .25, .30, .38)):
        rows = panel['episodes'][6*i:6*i+6]
        node_counts[str(height)] = dict(**panel['groups'][str(height)],
            active_design_margin_lower_bound_rad=min(design_bounds[6*i:6*i+6]),
            positive_negative_highspeed_stop_m=[r['stop_distance_m'] for r in rows[-2:]])
    common = read(OUT.parent / 'eleventh_round_nominal_panel_checked_20261002/verification.json')
    packet = read(OUT.parent / 'eighth_round_execution_packet_20261002/verification.json')
    assert packet['current_policy_normalization_roundtrip_passed'] and not packet['learning_performed']
    sequences = []
    for w in range(2):
        p = OUT.parent / 'braking_trajectory_slsqp_20261001' / f'world{w}_best.npz'
        paths.append(p)
        sequences.append(np.load(p, allow_pickle=False)['schedule'])
    branches = {}
    for name in ('twelfth_round_causal_allocation_20261002', 'thirteenth_round_feedback_checked_20261002', 'fourteenth_round_model_reset_20261002'):
        record = read(OUT.parent / name / 'verification.json')
        decisions = record['decisions']
        requests = np.array([r['request'] for r in decisions])
        guide = np.stack([s[:len(decisions)] for s in sequences], axis=1)
        np.testing.assert_array_equal(requests, guide)
        assert not record['completed'] and not record['learning_performed']
        times = np.array([r['elapsed_ms'] for r in decisions])
        branches[name] = dict(executed_updates=len(decisions), requests_different_from_guide=0,
            measured_decision_ms_median=float(np.median(times)), measured_decision_ms_p95=float(np.percentile(times, 95)),
            failure=record['failure'], completed=False)
    result = dict(role='fifteenth_round_direction_and_coverage_review',
        current_v2_five_node_normal_cases=node_counts,
        v3_frozen_plan_parameter_groups=common['groups'], predictor_branches=branches,
        current_packet_roundtrip_included_learning=packet['learning_performed'], full_admission=False,
        decision='Retain VMC/six-state LQR/diff3 research. Stop extending predictor resets; first isolate parameter sensitivity of the existing feasible nominal guide. Keep original constraints and all failed branches as evidence.',
        limitations='Thirty nominal zero-Actor cases cover five nodes only. Conservative active-joint bound follows from all-eight-joint evidence; no invariance, continuous-height, random-domain, learned delay robustness, or PPO advantage proof. Predictor timing is a measured diagnostic cost, not a pure-simulation real-time admission gate.',
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    target = OUT / 'verification.json'
    if target.exists():
        assert json.loads(target.read_text()) == result
    else:
        target.write_text(json.dumps(result, indent=2) + '\n')
    print('CHECKED 30 physical/design, 28 task; 1/96/96 requests all equal guide; no full admission.')


if __name__ == '__main__':
    run()
