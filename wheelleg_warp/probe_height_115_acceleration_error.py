"""首步加速度整场景留出审计；离散速度差分，不作连续时间安全保证。"""
import json
from pathlib import Path
import numpy as np
from probe_height_115_action_predict_loow import DT, ROOT, fit_gain, sha


def run():
    folder = ROOT / 'wheelleg_warp/results'
    source = folder / 'height_115_action_predict_1nm_single_graph_20260929'
    output = folder / 'height_115_acceleration_error_20260929'
    assert not output.exists()
    fit = json.loads((source / 'verification.json').read_text())
    assert sha(source / 'heldout_inputs.npz') == fit['heldout_inputs_sha256']
    data = np.load(source / 'heldout_inputs.npz')
    samples = [dict(world=i//2, q0=data['q0'][i], v0=data['v0'][i],
                    vprev=data['vprev'][i], coeff=data['actions'][i],
                    eps=np.array([data['actions'][i, 1+2*k, k] for k in range(3)]),
                    first_v=data['first_step_active_v'][i]) for i in range(12)]

    def errors(sample, gain):
        measured = (sample['first_v'] - sample['v0']) / DT
        previous = (sample['v0'] - sample['vprev']) / DT
        drift = previous - measured[0]
        response = sample['coeff'] @ gain.T - (measured - measured[0])
        total = previous + sample['coeff'] @ gain.T - measured
        assert np.allclose(total, drift + response, atol=1e-10, rtol=0)
        assert np.array_equal(response[0], np.zeros(4))
        return total, drift, response

    folds = []
    for world in range(6):
        train = set(range(6)) - {world}
        calibration = []
        for inner in sorted(train):
            gain = fit_gain(samples, train - {inner})
            calibration.extend(errors(s, gain)[0] for s in samples if s['world'] == inner)
        calibration = np.concatenate(calibration)
        # Both signs matter: upper/lower joint barriers use opposite acceleration signs.
        upper = np.maximum(0, calibration.max(axis=0))
        lower = np.maximum(0, -calibration.min(axis=0))
        gain = fit_gain(samples, train)
        assert np.allclose(gain, fit['folds'][world]['gain'], atol=1e-12, rtol=0)
        rows = []
        for i in (2*world, 2*world+1):
            total, drift, response = errors(samples[i], gain)
            excess = np.maximum(total-upper, -total-lower)
            rows.append(dict(lag_ms=(15,5)[i%2], zero_action_drift_rad_s2=drift.tolist(),
                max_abs_response_error_rad_s2=float(abs(response).max()),
                max_abs_total_error_rad_s2=float(abs(total).max()),
                uncovered_actions=int(np.any(excess > 1e-9, axis=1).sum()),
                maximum_bound_excess_rad_s2=float(max(0,excess.max())),
                signed_total_errors_rad_s2=total.tolist()))
        folds.append(dict(world=world, positive_error_bound_rad_s2=upper.tolist(),
                          negative_error_bound_rad_s2=lower.tolist(), rows=rows))

    live_path = folder / 'height_115_live_braking_20ms_20260929/verification.json'
    live = json.loads(live_path.read_text())
    e = np.array([a['active_acceleration_prediction_error'] for row in live['rows']
                  for a in row['acceleration_errors'] if a['arm']==2])
    upper = np.array(folds[5]['positive_error_bound_rad_s2'])
    lower = np.array(folds[5]['negative_error_bound_rad_s2'])
    excess = np.maximum(e-upper, -e-lower)
    report = dict(role='public_development_one_step_acceleration_error_audit',
        independent_new_data=False, hardware_or_continuous_time_bound=False,
        total_heldout_actions=108,
        uncovered_heldout_actions=sum(r['uncovered_actions'] for f in folds for r in f['rows']),
        folds=folds,
        live_reverse_receding=dict(scored_steps=len(e),
            uncovered_steps=int(np.any(excess>1e-9, axis=1).sum()),
            maximum_bound_excess_rad_s2=float(max(0,excess.max())),
            warning='Rolling trace is a retrospective transfer check; no tuning or new safety claim.'),
        limitations=['Only first-step active joint accelerations available in calibration archive.',
            'No per-step q/v for discrete radial rate acceleration or delayed action replay.',
            'Nested empirical maxima do not guarantee unseen states, action sequences or contacts.'],
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            (source/'verification.json',source/'heldout_inputs.npz',live_path)},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_action_predict_loow.py')})
    output.mkdir()
    (output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('folds','input_sha256','source_sha256')},ensure_ascii=False))
    for f in folds:
        print(f['world'], [(r['lag_ms'],r['uncovered_actions'],
              r['max_abs_total_error_rad_s2'],max(abs(x) for x in r['zero_action_drift_rad_s2'])) for r in f['rows']])


if __name__ == '__main__':
    run()
