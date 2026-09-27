"""Audit leg/wheel residual headroom in the two unmodified 160-world runs."""
from pathlib import Path
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'wheelleg_warp'), str(ROOT / 'wheelleg_ppo/tools')]
import numpy as np
from trace_failure_chain import COL
from dashboard.live_env import atomic_json as write


def group_lambdas(base, request, bound):
    ratio = np.full(request.shape, np.inf)
    np.divide(bound - base, request, out=ratio, where=request > 1e-9)
    np.divide(-bound - base, request, out=ratio, where=request < -1e-9)
    return tuple(np.clip(ratio[:, part].min(axis=1), 0., 1.) for part in (slice(0, 4), slice(4, 6)))


def check():
    base = np.zeros((1, 6)); request = np.full((1, 6), .5)
    bound = np.array([[1, 1, 1, 1, 0, 0]], dtype=float)
    leg, wheel = group_lambdas(base, request, bound)
    assert leg[0] == 1 and wheel[0] == 0
    assert group_lambdas(base, request, np.ones((1, 6)))[0][0] == 1
    print('group headroom check passed')


def analyze(directory):
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert protocol['worlds'] == 160 and not protocol['training'] and not protocol['holdout_evaluated']
    panel = json.loads((ROOT / 'wheelleg_warp/results/contract_v2_baseline_checked_20260923/protocol.json').read_text())
    speed_by_seed = {row['scenario']['terrain_seed']: row['scenario']['speed'] for row in panel['panels']}
    results = []
    for run_id in (1, 8):
        run = json.loads((directory / f'run_{run_id:02}.json').read_text())
        assert run['component'] is None and run['delta'] == 0 and len(run['outcomes']) == 160
        for seed, data in run['selected'].items():
            with np.load(directory / f'run_{run_id:02}_{seed}.npz', allow_pickle=False) as file:
                assert file['columns'].tolist() == COL
                trace = file['trace']; times = file['policy_time']; progress = file['wheel_progress']
            get = lambda name: trace[:, COL.index(name)]
            contact = data['first_target_contact_s']
            window = (get('end_s') >= contact) & (get('end_s') < contact + .3)
            assert window.sum() == 600 and np.isfinite(trace).all()
            base = np.column_stack([get(f'base_{j}') for j in range(6)])[window]
            request = np.column_stack([get(f'requested_{j}') for j in range(6)])[window]
            bound = np.column_stack([get(f'bound_{j}') for j in range(6)])[window]
            leg, wheel = group_lambdas(base, request, bound)
            zero = (get('lambda')[window] < 1e-6) & (np.linalg.norm(request[:, :4], axis=1) > 1e-6)
            assert zero.any() and np.all(leg[zero] >= 1. - 1e-9)
            sides = []
            for name in ('left_target_contacts', 'right_target_contacts'):
                indexes = np.flatnonzero(get(name) > 0)
                assert len(indexes)
                sides.append(float(get('end_s')[indexes[0]]))
            i = np.searchsorted(times, min(sides)) - 1
            assert i > 0 and np.all(np.isfinite(progress[i]))
            speed = (progress[i] - progress[i - 1]) / (times[i] - times[i - 1])
            direction = np.sign(speed_by_seed[int(seed)])
            common_margin = np.min(bound[:, 4:] - direction * base[:, 4:], axis=1)
            results.append(dict(run=run_id, seed=int(seed), success=data['info']['success'],
                first_left_contact_s=sides[0], first_right_contact_s=sides[1],
                contact_gap_ms=abs(sides[1] - sides[0]) * 1000,
                precontact_wheel_progress_m=progress[i].tolist(), precontact_wheel_speed_m_s=speed.tolist(),
                entry_lambda_zero_fraction=float((get('lambda')[window] < 1e-6).mean()),
                zero_lambda_with_leg_request_steps=int(zero.sum()),
                leg_full_feasible_at_zero_lambda_fraction=float((leg[zero] >= 1. - 1e-9).mean()),
                wheel_zero_feasible_at_zero_lambda_fraction=float((wheel[zero] < 1e-6).mean()),
                forward_common_wheel_margin_median_Nm=float(np.median(common_margin)),
                base_infeasible_fraction=float((get('base_infeasible')[window] > 0).mean())))
    a = json.loads((directory / 'run_01.json').read_text())['outcomes']
    b = json.loads((directory / 'run_08.json').read_text())['outcomes']
    flips = [dict(seed=x['seed'], group=x['group'], first_success=x['info']['success'],
                  last_success=y['info']['success']) for x, y in zip(a, b)
             if x['info']['success'] != y['info']['success']]
    assert all(x['seed'] == y['seed'] for x, y in zip(a, b))
    write(directory / 'analysis.json', dict(baseline_rows=results, baseline_outcome_flips=flips,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Offline command-space headroom only; independent leg actuation was not simulated.'))
    print('audited', len(results), 'baseline cases;', len(flips), 'unchanged-world flips')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--input', type=Path)
    parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    if args.check: check()
    elif args.input is None: parser.error('--input is required unless --check is used')
    elif (args.input / 'analysis.json').exists(): parser.error('Refusing to overwrite analysis.json')
    else: analyze(args.input)
