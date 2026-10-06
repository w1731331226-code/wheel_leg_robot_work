"""Frozen paired gates and observed switch transients; no rollout or training."""
import json
import numpy as np
import run_phase_support_qualification as runner
from review_floor_broad_qualification import ordered, wrap
from review_nom_yaw_filter import compare, unit
from review_reference_role import candidate_names
from score_reference_learning import load_rows

OUT, ROOT, sha, write = runner.OUT, runner.ROOT, runner.sha, runner.write


def switch_indices(phase):
    return np.flatnonzero(np.diff(phase[:, 3]) != 0)+1


def run():
    unit()
    fixture = np.zeros((4, 6)); fixture[:, 3] = [1, 0, 0, 1]
    np.testing.assert_array_equal(switch_indices(fixture), [1, 3])
    assert not len(switch_indices(fixture[:1]))
    p = runner.verify(); completion = json.loads((OUT/'completion.json').read_text())
    review = json.loads((OUT/'round192_data_review.json').read_text())
    assert completion['verified'] and review['verified'] and completion['completed_new_evaluations'] == 328
    assert review['completion_sha256'] == sha(OUT/'completion.json')
    inputs = {}; locations = {}; data = {}; pairs = {}; gates = {}; calibration = {}; retention = {}
    def read(ref, arm, law, panel):
        file = ROOT/ref['path']; assert sha(file) == ref['sha256']
        inputs[ref['path']] = ref['sha256']
        rows = json.loads(file.read_text())['runs']
        wanted = {c['seed']: c for c in p['panels'][panel]}
        load_rows(file, [wanted[r['seed']] for r in rows])
        for row in rows:
            locations[(arm, law, panel, row['seed'])] = file.parent
        return rows
    for law in p['classical']:
        for panel, cases in p['panels'].items():
            refs = ([p['original_development_references'][law+'/'+panel]] if panel != 'legacy'
                    else p['currentGPU_original_legacy_references'][law])
            wanted = {c['seed']: i for i, c in enumerate(cases)}
            arms = {}
            arms['original'] = [row for ref in refs for row in read(ref, 'original', law, panel)]
            refs = [r for r in p['fixedfloor_references'][law] if
                    ('controlled' if r['panel'].startswith('controlled') else r['panel']) == panel]
            arms['floor_only'] = [row for ref in refs for row in read(ref, 'floor_only', law, panel)]
            refs = [dict(path=str((OUT/j['path']).relative_to(ROOT)), sha256=j['sha256'])
                    for j in completion['records'] if j['law'] == law and j['panel'] == panel]
            arms['phase_support'] = [row for ref in refs for row in read(ref, 'phase_support', law, panel)]
            for arm, rows in arms.items():
                arms[arm] = wrap(ordered(rows, [wanted[r['seed']] for r in rows], len(cases)))
                assert [r['scenario'] for r in arms[arm]['runs']] == [c['scenario'] for c in cases]
                data[(law, panel, arm)] = arms[arm]
            for against in ('original', 'floor_only'):
                pairs[panel+'/'+law+'/vs_'+against] = candidate_names(compare(arms[against], arms['phase_support']))
            original_pair = pairs[panel+'/'+law+'/vs_original']
            if panel != 'legacy':
                gates[panel+'/'+law] = original_pair['fixed_variant_qualification']
            else:
                gates[panel+'/'+law] = (not original_pair['lost'] and not original_pair['new_axis_or_task_failures']
                    and original_pair['physical_design_all_pass'] and arms['phase_support']['summary']['success_count'] == 28)
        for panel, ids in [('regular', [6300092]), ('controlled', [6301009, 6301011, 6301013, 6301015])]:
            rows = {r['seed']: r for r in data[(law, panel, 'phase_support')]['runs']}
            retention[panel+'/'+law] = dict(required_case_ids=ids, passed=all(rows[i]['success'] for i in ids))
        control_pair = pairs['controlled/'+law+'/vs_floor_only']
        retention['controlled_no_loss/'+law] = dict(passed=not control_pair['lost'] and
            not control_pair['new_axis_or_task_failures'] and data[(law, 'controlled', 'phase_support')]['summary']['success_count'] >= 32)
        row85 = next(r for r in data[(law, 'regular', 'phase_support')]['runs'] if r['seed'] == 6300085)
        retention['repair85/'+law] = dict(passed=row85['success'] and row85['design_joint_passed'],
                                         min_design_margin_rad=row85['min_active_design_margin_rad'])
        cpu_ref = p['cpu_legacy_reference']; cpu_file = ROOT/cpu_ref['path']
        assert sha(cpu_file) == cpu_ref['sha256']; inputs[cpu_ref['path']] = cpu_ref['sha256']
        cpu = json.loads(cpu_file.read_text())['runs']
        for arm in ('original', 'phase_support'):
            failures = []
            for row, old in zip(data[(law, 'legacy', arm)]['runs'], cpu):
                assert row['seed'] == old['name']; flags = []
                if not row['success']: flags.append('full_task')
                if row['velocity_rmse'] > old['velocity_rmse']*1.05+.005+1e-12: flags.append('velocity')
                for i, axis in enumerate(('roll', 'pitch')):
                    if row['peak_deg'][i] > old['peak_deg'][i]+.1+1e-12: flags.append(axis)
                if flags:
                    failures.append(dict(case=row['seed'], flags=flags, CPU_velocity=old['velocity_rmse'],
                        candidate_velocity=row['velocity_rmse'], CPU_peak=old['peak_deg'], candidate_peak=row['peak_deg']))
            calibration[arm+'/'+law] = dict(passed=not failures, failures=failures)
    events = []; episode_stats = []
    for (law, panel, arm), result in data.items():
        if arm != 'phase_support': continue
        for row in result['runs']:
            directory = locations[(arm, law, panel, row['seed'])]
            values = {}
            for field in ('phase_trace', 'role_trace', 'complete_trace'):
                file = directory/row[field]['path']; assert sha(file) == row[field]['sha256']
                inputs[str(file.relative_to(ROOT))] = row[field]['sha256']
                with np.load(file, allow_pickle=False) as z:
                    values[field] = z['trace']
                    if field == 'complete_trace':
                        pre, post = z['pre'], z['post']; assert z['state_sizes'].tolist() == [17, 16, 6]
            phase, role, trace = [values[k] for k in ('phase_trace', 'role_trace', 'complete_trace')]
            runner.probe.check_log(phase, arm)
            np.testing.assert_array_equal(phase[:, :2], trace[:, :2])
            np.testing.assert_array_equal(phase[:, 2], trace[:, 43])
            np.testing.assert_array_equal(phase[:, 5], role[:, 5])
            indices = switch_indices(phase)
            for i in indices:
                events.append(dict(law=law, panel=panel, case=row['seed'], step=int(i+1),
                    pre_s=float(trace[i, 2]), post_s=float(trace[i, 3]),
                    direction='to_zero' if phase[i, 3] else 'to_nonzero', arrival_s=row['arrival_s'],
                    command_before=float(phase[i-1, 2]), command_after=float(phase[i, 2]),
                    anchor_before_m=float(phase[i-1, 5]), anchor_after_m=float(phase[i, 5]),
                    anchor_changed=bool(phase[i-1, 5] != phase[i, 5]),
                    requested_guard_jump_N=float(role[i, 9]-role[i-1, 9]),
                    applied_guard_jump_N=float(role[i, 10]-role[i-1, 10]),
                    max_abs_command_jump_Nm=float(abs(pre[i, 33:39]-pre[i-1, 33:39]).max()),
                    max_abs_actual_torque_jump_Nm=float(abs(post[i, 33:39]-post[i-1, 33:39]).max()),
                    max_abs_hip_command_Nm=float(abs(pre[i, 33:37]).max()),
                    max_abs_wheel_command_Nm=float(abs(pre[i, 37:39]).max()),
                    post_design_margin_rad=float(trace[i, 24]), post_body_vx_m_s=float(trace[i, 44])))
            episode_stats.append(dict(law=law, panel=panel, case=row['seed'], switches=len(indices),
                effective_anchor_changes=int(np.count_nonzero(np.diff(phase[:, 5]))),
                max_requested_guard_N=float(role[:, 9].max()), max_applied_guard_N=float(role[:, 10].max()),
                max_actual_torque_excess_Nm=row['max_actual_torque_excess_Nm'],
                max_command_torque_excess_Nm=row['max_command_torque_excess_Nm'],
                min_design_margin_rad=row['min_active_design_margin_rad']))
        print('TRANSIENTS CHECKED', panel, law, len(result['runs']), flush=True)
    physical_design = all(v['physical_design_all_pass'] for k, v in pairs.items() if k.endswith('vs_original'))
    qualified = physical_design and all(gates.values()) and all(v['passed'] for v in retention.values()) and all(v['passed'] for v in calibration.values())
    transient_summary = dict(events=len(events), effective_anchor_events=sum(e['anchor_changed'] for e in events),
        max_abs_applied_guard_jump_N=max(abs(e['applied_guard_jump_N']) for e in events),
        max_abs_command_jump_Nm=max(e['max_abs_command_jump_Nm'] for e in events),
        max_abs_actual_torque_jump_Nm=max(e['max_abs_actual_torque_jump_Nm'] for e in events),
        all_episode_torque_bounds_pass=all(e['max_actual_torque_excess_Nm'] <= 1e-6 and e['max_command_torque_excess_Nm'] <= 1e-6 for e in episode_stats),
        interpretation='Observed one-substep differences mix switch, changed speed command, feedback and contact dynamics. '
            'Virtual guard-force jump is not a contact-force jump. No new after-the-fact jump threshold, '
            'comfort claim, unique causal attribution or global dynamic safety certificate.')
    write(OUT/'round193_pair_review.json', dict(verified=True, round=193, pairs=pairs, gates=gates,
        benefit_retention=retention, CPU_legacy_calibration=calibration, all_candidate_physical_design_pass=physical_design,
        overall_engineering_qualification=qualified, switch_summary=transient_summary,
        source_sha256=sha(__file__), input_sha256=inputs, completion_sha256=sha(OUT/'completion.json'),
        data_review_sha256=sha(OUT/'round192_data_review.json'), new_evaluations=0, training_updates=0,
        decision=('Retain phase_support as a stronger conventional development reference only.' if qualified
                  else 'Close phase_support candidate at original gates; no threshold/floor/gain/smoothing rescue.'),
        limits='984 rows on164 already used development/regression cases; not independent generalization or trained seeds. '
            'No default replacement or new PPO admission; engineering phase switch is not an established new method.',
        next='194 define a defensible remaining method and learning-necessity contrast from this closure;195 deepreview/cleanup. '
            'Full contribution/formal5seeds/independent tests/statistics/PPO timing/manuscript remain open.'))
    write(OUT/'round193_switch_transients.json', dict(verified=True, round=193, summary=transient_summary,
        events=events, episodes=episode_stats, paired_review_sha256=sha(OUT/'round193_pair_review.json'),
        source_sha256=sha(__file__), new_evaluations=0, training_updates=0))
    for k, v in pairs.items():
        print(k, 'success', v['original_success'], v['candidate_success'], 'lost', v['lost'],
              'new_flags', v['new_axis_or_task_failures'], 'J_reduction', v['yaw_score_reduction_deg'],
              'peak_reduction', v['yaw_peak_reduction_deg'], 'gate', v['fixed_variant_qualification'], flush=True)
    print('CPU', {k: v['passed'] for k, v in calibration.items()}, 'RETENTION', retention, flush=True)
    print('TRANSIENTS', transient_summary, 'OVERALL', qualified, flush=True)


if __name__ == '__main__':
    run()
