"""Immutable fixed-transfer assay gates, including applied exposure and zero controls."""
import hashlib
import json
import numpy as np
import run_equal_exposure_recovery as runner
from review_floor_broad_qualification import ordered, wrap
from review_nom_yaw_filter import compare, unit, full_flags
from review_reference_role import candidate_names
from score_reference_learning import load_rows, yaw_gate

OUT, ROOT, sha, write = runner.OUT, runner.ROOT, runner.sha, runner.write


def waveform_digest(submitted):
    return hashlib.sha256(np.ascontiguousarray(submitted, dtype=np.float32).tobytes()).hexdigest()


def run():
    unit()
    assert waveform_digest(np.zeros((200, 6))) == waveform_digest(np.zeros((200, 6), np.float32))
    x = np.zeros((200, 6)); x[0, 5] = 1
    assert waveform_digest(x) != waveform_digest(np.zeros((200, 6)))
    p = runner.verify(); c = json.loads((OUT/'completion.json').read_text())
    previous = json.loads((OUT/'round197_data_review.json').read_text())
    assert c['verified'] and previous['verified'] and c['completed_new_evaluations'] == 225
    assert previous['completion_sha256'] == sha(OUT/'completion.json')
    inputs = {}; collected = {}; delivery = []; exposure = {}
    values = runner.probe.profiles()
    for job in c['records']:
        file = OUT/job['path']; assert sha(file) == job['sha256']
        inputs[str(file.relative_to(ROOT))] = job['sha256']
        cases = [p['cases'][i] for i in job['indices']]
        rows = load_rows(file, cases)['runs']
        group = collected.setdefault(job['label'], dict(rows=[], indices=[]))
        group['rows'] += rows; group['indices'] += job['indices']
        for row in rows:
            index = -1 if row['profile'] == 'zero' else row['profile']
            force_file = file.parent/row['force_trace']['path']; assert sha(force_file) == row['force_trace']['sha256']
            inputs[str(force_file.relative_to(ROOT))] = row['force_trace']['sha256']
            with np.load(force_file, allow_pickle=False) as z:
                assert z['columns'].tolist() == runner.probe.COL; data = z['trace']
                runner.probe.check_samples(data, index, int(data[0, 4]), values)
                np.testing.assert_array_equal(data[:, 1], np.arange(1, len(data)+1))
            pulse = (data[:, 1]-1 >= 5000) & (data[:, 1]-1 < 5200)
            complete = int(pulse.sum()) == 200 if index >= 0 else not data[:, 6:12].any()
            assert row['force_delivery']['complete'] == complete
            digest = waveform_digest(data[pulse, 6:12]) if index >= 0 and complete else None
            expected = np.zeros((200, 6), np.float32)
            if index >= 0: expected[:, 5] = values[index].astype(np.float32)
            if digest is not None: assert digest == waveform_digest(expected)
            if index >= 0:
                exposure.setdefault(index, []).append(dict(controller=job['label'], case=row['seed'],
                    height_m=row['target_leg_m'], complete=complete, submitted_waveform_sha256=digest))
            delivery.append(dict(controller=job['label'], case=row['seed'], profile=index,
                expected=200 if index >= 0 else 0, delivered=int(pulse.sum()) if index >= 0 else 0,
                complete=complete, submitted_waveform_sha256=digest))
    data = {label: ordered(v['rows'], v['indices'], 45) for label, v in collected.items()}
    assert len(data) == 5
    split = {label: {kind: wrap([r for r in rows if (r['profile'] == 'zero') == (kind == 'zero')])
                    for kind in ('zero', 'pulse')} for label, rows in data.items()}
    same_exposure = all(len(rows) == 25 and all(r['complete'] for r in rows) and
                        len({r['submitted_waveform_sha256'] for r in rows}) == 1 for rows in exposure.values())
    assert len(exposure) == 8
    context = {label: dict(success=g['zero']['summary']['success_count'], count=5,
        mean_yaw_score_deg=g['zero']['summary']['mean_yaw_score_deg'],
        flags=[dict(case=r['seed'], flags=[k for k, v in full_flags(r).items() if v])
               for r in g['zero']['runs'] if not r['success']], passed=all(r['success'] for r in g['zero']['runs']))
               for label, g in split.items()}
    pairs = {}; per_model = {}; paired_zero_pulse = {}
    models = [m['label'] for m in p['controllers']['models']]
    for label in models:
        preservation = []
        for reference in ('B0', 'B1-route'):
            v = candidate_names(compare(split[reference]['pulse'], split[label]['pulse']))
            pairs[label+'/vs_'+reference] = v
            preservation.append(not v['lost'] and not v['new_axis_or_task_failures'] and v['physical_design_all_pass'])
        per_model[label] = dict(classical_union_success_and_components_preserved=all(preservation),
            pulse_success=split[label]['pulse']['summary']['success_count'],
            mean_yaw_score_deg=split[label]['pulse']['summary']['mean_yaw_score_deg'])
    for label, groups in split.items():
        zero = {r['target_leg_m']: r for r in groups['zero']['runs']}
        changes = []
        for row in groups['pulse']['runs']:
            z = zero[row['target_leg_m']]
            changes.append(dict(case=row['seed'], profile=row['profile'], height_m=row['target_leg_m'],
                zero_success=z['success'], pulse_success=row['success'],
                yaw_RMS_change_deg=row['rms_deg'][2]-z['rms_deg'][2],
                yaw_peak_change_deg=row['peak_deg'][2]-z['peak_deg'][2],
                velocity_rmse_change=row['velocity_rmse']-z['velocity_rmse'],
                stop_distance_change_m=row['stop_distance_m']-z['stop_distance_m']))
        paired_zero_pulse[label] = changes
    contrast_summary = {label: dict(
        zero_mean_yaw_score_deg=groups['zero']['summary']['mean_yaw_score_deg'],
        pulse_mean_yaw_score_deg=groups['pulse']['summary']['mean_yaw_score_deg'],
        observed_mean_yaw_RMS_change_deg=float(np.mean([r['yaw_RMS_change_deg'] for r in paired_zero_pulse[label]])),
        observed_mean_yaw_peak_change_deg=float(np.mean([r['yaw_peak_change_deg'] for r in paired_zero_pulse[label]])))
        for label, groups in split.items()}
    reference_J = split['B1-route']['pulse']['summary']['mean_yaw_score_deg']
    model_J = [per_model[label]['mean_yaw_score_deg'] for label in models]
    score = yaw_gate(model_J, reference_J)
    gates = dict(source_and_model_RMS_valid=True, delivery_complete=all(r['complete'] for r in delivery),
        identical_submitted_waveforms=same_exposure, all_zero_contexts_pass=all(v['passed'] for v in context.values()),
        all_candidate_physical_design_pass=all(r['physical_safety_passed'] and r['design_joint_passed'] for label in models for r in data[label]),
        all_models_preserve_classical_union_and_flags=all(v['classical_union_success_and_components_preserved'] for v in per_model.values()),
        all_models_direction_and_15percent_0p05deg_gate=score)
    qualified = all(gates.values())
    result = dict(verified=True, round=198, gates=gates, diagnostic_recovery_gate_passed=qualified,
        contexts=context, pairs=pairs, per_model=per_model,
        score=dict(reference='B1-route, fixed before outcomes', reference_mean_yaw_score_deg=reference_J,
            candidate_mean_yaw_scores_deg=model_J, three_model_mean_deg=float(np.mean(model_J)),
            required_max_mean_deg=min(.85*reference_J, reference_J-.05), passed=score),
        observed_zero_vs_pulse_changes=paired_zero_pulse,
        observed_contrast_summary=contrast_summary,
        exposure=dict(profiles=8, operational_height_profile_cases=40, pulse_records=200,
            controllers=5, zero_records=25, all_expected200_delivered=True if all(r['complete'] for r in delivery) else False,
            submitted_float32_waveforms_identical=same_exposure, per_profile=exposure, rows=delivery),
        source_sha256=sha(__file__), input_sha256=inputs, completion_sha256=sha(OUT/'completion.json'),
        data_review_sha256=sha(OUT/'round197_data_review.json'), new_evaluations=0, training_updates=0,
        decision=('Fixed-transfer diagnostic supports a recovery-capability witness, not matched learning or novelty.' if qualified
                  else 'Close this frozen-transfer recovery superiority claim at unchanged gates; retain any positive case-wise witnesses and all regressions.'),
        limits='Same applied world torque is not the same realized motion/contact response. Reference-derived pulse includes traction, '
            'not terrain equivalence. Three old policies were not trained on phase/pulse; zero controls do not remove all context shift. '
            'Five heights/eight profiles and old trained seeds are repeated operational conditions, not fresh independent generalization. '
            'Zero-pulse changes are nonlinear observed contrasts, not additive natural-contact causal effects.',
        next='199 select/reject concrete remaining mechanism and matched-learning contrast without profile/gain/budget rescue;'
            '200 directionreview/cleanup. Full contribution/new3→formal5seeds/independenttests/statistics/PPO timing/manuscript remain open.')
    write(OUT/'round198_pair_review.json', result)
    print('GATES', gates, 'OVERALL', qualified, flush=True)
    print('SCORE', result['score'], flush=True)
    for k, v in pairs.items():
        print(k, 'lost', v['lost'], 'gained', v['gained'], 'newflags', v['new_axis_or_task_failures'],
              'Jreduction', v['yaw_score_reduction_deg'], flush=True)


if __name__ == '__main__':
    run()
