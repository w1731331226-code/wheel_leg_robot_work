"""Frozen selection-only B0/B1 evaluation and yaw-precision scoring."""
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np
from ppo_env import Scenario, WheelLegEnv

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'tools/results/yaw_precision_v1_2026-09-20'


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def verify_sources(frozen=FROZEN):
    manifest = json.loads((frozen / 'protocol_manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    return manifest


def yaw_score(row):
    if row['reason'] != 'completed':
        return None
    s = row['scenario']
    ref = 3.5 + 1.5 * (s['center'] + abs(s['offset']) / 2 + .75) / abs(s['speed'])
    duration = row['physical_steps'] * .0005
    rms = row['rms_deg'][2]
    if not (duration > 0 and math.isfinite(rms) and rms >= 0
            and math.isclose(duration, row['duration_s'], abs_tol=1e-8)
            and row['arrival_s'] is not None and duration >= row['arrival_s'] + 2 - 1e-8):
        raise ValueError('Invalid complete trajectory or physics timestep')
    return rms * math.sqrt(duration / ref)


def summarize(rows, expected_seeds):
    if [r['seed'] for r in rows] != list(expected_seeds):
        raise ValueError('Missing, duplicated or reordered scenarios')
    scores = [yaw_score(r) for r in rows]
    return dict(total=len(rows), success_count=sum(bool(r['success']) for r in rows),
                complete=all(x is not None for x in scores),
                mean_yaw_score_deg=float(np.mean(scores)) if all(x is not None for x in scores) else None)


def selection_key(summary, steps=0):
    if not summary['complete']:
        return (math.inf, math.inf, math.inf)
    return (summary['total'] - summary['success_count'], summary['mean_yaw_score_deg'], steps)


def b1_action(obs, candidate):
    kp, kd, gain = candidate['kp'], candidate['kd'], candidate['roll_gain']
    schedule = 1 + gain * min(abs(float(obs[0])) / math.radians(2), 1.)
    torque = -schedule * (kp * float(obs[2]) + kd * float(obs[5]))
    return np.array([0., 0., np.clip(torque / .3, -1., 1.)], dtype=np.float32)


def evaluate_case(item):
    candidate, case = item
    with contextlib.redirect_stdout(io.StringIO()):
        env = WheelLegEnv(scenario=Scenario(**case['scenario']))
        obs, _ = env.reset(seed=0)
        while not env.done:
            obs, _, _, _, _ = env.step(b1_action(obs, candidate))
        row = dict(seed=case['seed'], scenario=case['scenario'], **env.metrics())
        env.close()
    return row


def evaluate_selection(output):
    manifest = verify_sources()
    config = json.loads((FROZEN / 'training_config.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [Path(__file__), FROZEN/'training_config.json', FROZEN/'protocol_manifest.json']}
    write(output/'source_sha256.json', sources)
    cases = manifest['sets']['selection']
    seeds = [r['seed'] for r in cases]
    summaries = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for candidate in config['b1_candidates']:
            rows = list(pool.map(evaluate_case, [(candidate, case) for case in cases]))
            summary = dict(candidate=candidate, **summarize(rows, seeds))
            summaries.append(summary)
            write(output/(candidate['name']+'.json'), dict(summary=summary, runs=rows))
            print(candidate['name'], summary['success_count'], summary['mean_yaw_score_deg'], flush=True)
    best = min(summaries, key=selection_key)
    complete = [s for s in summaries if s['complete']]
    # Best B1 follows the frozen lexicographic selection, including zero residual.
    b0 = summaries[0]
    q = min(b0['mean_yaw_score_deg'], best['mean_yaw_score_deg']) if b0['complete'] and complete else None
    verify_sources()
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in sources.items())
    write(output/'summary.json', dict(candidates=summaries, selected_b1=best if complete else None,
          classical_reference_score_deg=q, absolute_headroom_pass=q is not None and q >= .05,
          gate_evaluated=False, test_set_evaluated=False))


def self_check():
    base = dict(seed=1, scenario=dict(center=2., offset=0., speed=1.), reason='completed',
                physical_steps=12000, duration_s=6., arrival_s=4., rms_deg=[0., 0., 1.], success=True)
    assert math.isclose(yaw_score(base), math.sqrt(6/7.625))
    assert yaw_score(dict(base, reason='fall')) is None
    assert yaw_score(dict(base, success=False)) == yaw_score(base)
    summary = summarize([base], [1])
    assert selection_key(summary, 10) < selection_key(summary, 20)
    assert selection_key(summary) < selection_key(dict(summary, success_count=0, mean_yaw_score_deg=0.))
    assert not summarize([dict(base, reason='timeout')], [1])['complete']
    for bad in ([1, 2], [2]):
        try: summarize([base], bad)
        except ValueError: pass
        else: raise AssertionError('missing/mismatched scenarios accepted')
    try: yaw_score(dict(base, duration_s=5.))
    except ValueError: pass
    else: raise AssertionError('inconsistent time accepted')
    obs = np.zeros(32); obs[2] = .1; obs[5] = .2
    c = dict(kp=.4, kd=.3, roll_gain=1)
    assert b1_action(obs,c)[2] < 0
    np.testing.assert_array_equal(b1_action(obs,dict(kp=0,kd=0,roll_gain=0)), np.zeros(3))
    assert np.max(np.abs(b1_action(obs*100,c))) <= 1
    print('PASS: score, failure exclusion, checkpoint order, data completeness, B1 sign/bounds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.output:
        evaluate_selection(args.output)
    else:
        parser.error('Use --self-check or --output NEW_DIRECTORY')
