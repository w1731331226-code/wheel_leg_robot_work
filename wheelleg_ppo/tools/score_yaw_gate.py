"""Offline v2 gate scoring. Consumes completed logs; never runs a simulator.

Input: {protocol, gate: {method: {training_seed: [raw metric rows]}},
regression: {B0: {fixed: [original 28 rows]}, M3: {training_seed: [rows]}}}.
Classical gate methods B0/B1 use key 'fixed'; RL keys are '1609'..'1611'.
Gate rows identify scenarios by 'seed'; regression rows by 'name'.
This checks numeric results and case identity, not checkpoint provenance.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

from pretrain_yaw import ROOT, yaw_score

REFERENCES = ('B0', 'B1', 'B2', 'B2-V', 'M1', 'N3+', 'N3-')
PAIRED = ('B2-V', 'M1', 'N3+', 'N3-')
SEEDS = ('1609', '1610', '1611')


def at_least(value, threshold):
    # Tolerance only for floating-point comparison, never added to a denominator.
    return value >= threshold or math.isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


def batches(data, expected_methods, cases, key):
    if set(data) != set(expected_methods):
        raise ValueError('Missing or extra methods')
    expected = {case[key]: case['scenario'] for case in cases}
    if len(expected) != len(cases):
        raise ValueError('Duplicate case IDs in reference data')
    result = {}
    for method, runs in data.items():
        seeds = ('fixed',) if method in ('B0', 'B1') else SEEDS
        if set(runs) != set(seeds):
            raise ValueError('Missing/extra training seeds, or fabricated classical repeats')
        result[method] = {}
        for seed, rows in runs.items():
            indexed = {row[key]: row for row in rows}
            if len(rows) != len(expected) or len(indexed) != len(rows) or set(indexed) != set(expected):
                raise ValueError('Missing, duplicate or foreign scenarios')
            for case_id, row in indexed.items():
                if row['scenario'] != expected[case_id] or type(row['success']) is not bool:
                    raise ValueError('Changed scenario or invalid success flag')
                if row['reason'] == 'completed':
                    for field in ('rms_deg', 'peak_deg'):
                        if len(row[field]) != 3 or any(not math.isfinite(x) or x < 0 for x in row[field]):
                            raise ValueError('Invalid attitude metrics')
                    for field in ('velocity_rmse', 'arrival_s', 'stop_distance_m', 'tail_speed_m_s'):
                        if not math.isfinite(row[field]) or row[field] < 0:
                            raise ValueError('Invalid completed-trajectory metrics')
                    yaw_score(row)  # Also validates duration and full stop observation.
                    if row['success'] and not (
                        max(row['peak_deg']) <= 5 and row['velocity_rmse'] <= .2 * abs(row['scenario']['speed'])
                        and row['stop_distance_m'] <= .6 and row['tail_speed_m_s'] <= .03
                    ):
                        raise ValueError('Success flag contradicts hard task thresholds')
                elif row['success']:
                    raise ValueError('Early termination cannot be successful')
            result[method][seed] = [indexed[case_id] for case_id in expected]
    return result


def assess(payload, gate_cases, regression_cases):
    if payload.get('protocol') != 'yaw-precision-v2' or len(gate_cases) != 64 or len(regression_cases) != 28:
        raise ValueError('Expected v2 with 64 gate and 28 regression cases')
    gate = batches(payload['gate'], (*REFERENCES, 'M3'), gate_cases, 'seed')
    regression = batches(payload['regression'], ('B0', 'M3'), regression_cases, 'name')
    failures = []
    scores = {}
    for method, runs in gate.items():
        scores[method] = {}
        for seed, rows in runs.items():
            values = [yaw_score(row) for row in rows]
            scores[method][seed] = mean(values) if all(v is not None for v in values) else None
            failures.extend(f'incomplete gate: {method}/{seed}/{r["seed"]}' for r in rows if r['reason'] != 'completed')
    failures.extend(f'incomplete regression: {method}/{seed}/{r["name"]}'
                    for method, runs in regression.items() for seed, rows in runs.items()
                    for r in rows if r['reason'] != 'completed')
    report = dict(protocol='yaw-precision-v2', passed=False, per_seed_scores=scores,
                  success_counts={method:{seed:sum(r['success'] for r in rows) for seed,rows in runs.items()}
                                  for method,runs in gate.items()},
                  gate_episodes_per_run=64, regression_episodes_per_run=28,
                  failures=failures, no_simulation_executed=True)
    if failures:
        return report  # Never rank a truncated low-error trajectory against a full one.

    q = {method: mean(runs.values()) for method, runs in scores.items()}
    reference = min(REFERENCES, key=lambda method: q[method])
    delta = q[reference] - q['M3']
    relative = delta / q[reference] if q[reference] > 0 else None
    if relative is None or not at_least(relative, .15) or not at_least(delta, .05):
        failures.append('effect: requires >=15% and >=0.05 degrees against best reference')
    for seed in SEEDS:
        for method in PAIRED:
            if not scores['M3'][seed] < scores[method][seed]:
                failures.append(f'paired direction: {seed}/{method}')
        candidate = gate['M3'][seed]
        best_count = max(sum(r['success'] for r in gate[method]['fixed' if method in ('B0','B1') else seed])
                         for method in REFERENCES)
        if sum(r['success'] for r in candidate) < best_count:
            failures.append(f'success count: {seed}')
        for row, base in zip(candidate, gate['B0']['fixed']):
            label = f'{seed}/{row["seed"]}'
            if base['success'] and not row['success']:
                failures.append('lost B0 success: '+label)
            if not at_least(1.05 * base['velocity_rmse'] + .005, row['velocity_rmse']):
                failures.append('velocity regression: '+label)
            if not at_least(1.05 * base['arrival_s'] + .05, row['arrival_s']):
                failures.append('arrival regression: '+label)
        for row, base in zip(regression['M3'][seed], regression['B0']['fixed']):
            label = f'{seed}/{row["name"]}'
            if not base['success']:
                raise ValueError('The original 28-case B0 regression must be successful')
            if not row['success']:
                failures.append('nominal success: '+label)
            if not at_least(1.05 * base['velocity_rmse'] + .005, row['velocity_rmse']):
                failures.append('nominal velocity: '+label)
            if any(not at_least(base['peak_deg'][i] + .1, row['peak_deg'][i]) for i in (0,1)):
                failures.append('nominal roll/pitch: '+label)
    report.update(passed=not failures, method_scores=q, best_reference=reference,
                  absolute_improvement_deg=delta, relative_improvement=relative)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Explicitly supplied result logs; no automatic search')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT/'tools/results/yaw_precision_v2_2026-09-21/protocol_manifest.json').read_text())
    baseline = json.loads((ROOT/'tools/results/fixes_2026-09-17/final/baseline.json').read_text())
    payload = json.loads(args.input.read_text())
    if payload['regression']['B0']['fixed'] != baseline['runs']:
        raise ValueError('Use the original unmodified 28-case B0 regression record')
    report = assess(payload, manifest['sets']['gate'], baseline['runs'])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print('PASS' if report['passed'] else 'NOT PASSED')
