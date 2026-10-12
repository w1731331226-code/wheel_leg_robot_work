"""Offline gain selection and development retention; never admits a learner.

The independently selected Cart gains are strong benchmarks, not a matched
geometry experiment. Original B1 parking is part of this new baseline contract;
the round377 light-withdrawal PPO receipt does not qualify it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_classical_candidate_v1'
OLD = OUT.parent / 'fixed_force_prequalification_v1'
SELECTION = ROOT / 'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/classical_selection/B1_3.json'
CHARTS = ('CartLive3', 'CartReset3')
PANELS = ('regular', 'controlled', 'legacy')
GAMMAS = (0, .5, 1, 2)
RAW_RECEIVER_SOURCE = 'wheelleg_warp/review_cartesian_classical_raw.py'
CPU_REFERENCE = ROOT / 'wheelleg_ppo/tools/results/fixes_2026-09-17/final/baseline.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def cases_for(proposal, panel):
    jobs = proposal['selection_jobs'] if panel == 'selection' else proposal['eval_jobs']
    return [case for job in jobs if job['panel'] == panel for case in job['cases']]


def validated_result(result, cases):
    """Recompute original success, physical/design predicates and original J."""
    from train_height_comparison import summary
    from review_yaw_sector import success
    rows = result['runs']
    assert len(rows) == len(cases) and rows
    for row, case in zip(rows, cases):
        assert row['seed'] == case['seed'] and row['scenario'] == case['scenario']
        for key in ('success', 'physical_safety_passed', 'design_joint_passed'):
            assert type(row[key]) is bool
        for key in ('min_actual_A_leg_m', 'min_actual_B_leg_m', 'geometric_limit_m',
                    'min_eight_joint_margin_rad', 'max_actual_torque_excess_Nm',
                    'max_command_torque_excess_Nm', 'min_active_design_margin_rad',
                    'duration_s', 'task_goal_progress_m'):
            assert finite(row[key]), key
        assert type(row['physical_steps']) is int and row['physical_steps'] > 0
        assert row['physical_evidence_steps'] == row['physical_steps']
        assert row['duration_s'] > 0 and math.isclose(row['duration_s'], row['physical_steps'] * .0005, rel_tol=0, abs_tol=1e-8)
        physical = (min(row['min_actual_A_leg_m'], row['min_actual_B_leg_m']) >= row['geometric_limit_m']
                    and row['min_eight_joint_margin_rad'] >= 0
                    and max(row['max_actual_torque_excess_Nm'], row['max_command_torque_excess_Nm']) <= 1e-6)
        assert row['physical_safety_passed'] == physical
        assert row['design_joint_passed'] == (row['min_active_design_margin_rad'] >= 0)
        assert row['success'] == success(row)
    computed = summary(rows)
    assert result['summary'] == computed, 'Stored summary differs from original summary(rows)'
    assert result['physical'] == sum(row['physical_safety_passed'] for row in rows)
    assert result['design'] == sum(row['design_joint_passed'] for row in rows)
    if computed['complete']:
        assert finite(computed['mean_yaw_score_deg']) and computed['mean_yaw_score_deg'] >= 0
    return dict(runs=rows, summary=computed, physical=result['physical'], design=result['design'])


def selection_decision(proposal, results):
    """All eight selection results must pass before either gain is selected."""
    assert proposal['parameters']['gamma_candidates'] == list(GAMMAS)
    conditions = proposal['conditions']
    assert [(c['chart'], c['gamma']) for c in conditions] == [(chart, gain) for chart in CHARTS for gain in GAMMAS]
    assert set(results) == {c['label'] for c in conditions}
    cases = cases_for(proposal, 'selection')
    assert len(cases) == 32
    candidates = []
    for condition in conditions:
        result = validated_result(results[condition['label']], cases)
        assert result['summary']['complete'] and result['physical'] == result['design'] == 32, 'Any selection hard/source/evidence failure stops the queue'
        candidates.append(dict(**condition, gamma_index=GAMMAS.index(condition['gamma']), summary=result['summary']))
    selected = [min((c for c in candidates if c['chart'] == chart), key=lambda c: (-c['summary']['success_count'], c['summary']['mean_yaw_score_deg'], c['gamma_index'])) for chart in CHARTS]
    return dict(verified=True, episodes=256, selected=selected, candidates=candidates,
                selection_metric_received=True, baseline_qualified=False, new_learning_admitted=False,
                formal5_admitted=False, limits='Development selection32 only. Independently tuned gains are benchmarks; their chart difference does not isolate geometry. Freeze before any new full164 outcome; no fallback or retry.')


def cpu_reference_rows(cpu_pair, b0_rows):
    """Read the SHA-bound public historical CPU28, not a scientific final set.

    The bridge's B0 peak minus delta would be a floating-point reconstruction,
    not an exact read of the CPU peaks. Read the named legacy source directly
    and check the bridge, so the original .1 degree boundary needs no new error
    allowance or outcome-dependent tolerance.
    """
    assert cpu_pair['passed'] and cpu_pair['verified_pairs'] and len(cpu_pair['records']) == len(b0_rows) == 28
    assert sha(CPU_REFERENCE) == cpu_pair['CPU_baseline_sha256']
    original = read(CPU_REFERENCE)['runs']
    assert len(original) == 28
    rows = []
    for pair, b0, cpu in zip(cpu_pair['records'], b0_rows, original):
        assert pair['case'] == b0['seed'] == cpu['name'] and all(pair['checks'].values())
        assert cpu['success'] is True and all(b0['scenario'][key] == value for key, value in cpu['scenario'].items())
        assert pair['new_velocity'] == b0['velocity_rmse'] and pair['new_arrival'] == b0['arrival_s']
        for key in ('old_velocity', 'old_arrival', 'roll_delta', 'pitch_delta'):
            assert finite(pair[key])
        assert pair['old_velocity'] == cpu['velocity_rmse'] and pair['old_arrival'] == cpu['arrival_s']
        assert pair['roll_delta'] == b0['peak_deg'][0] - cpu['peak_deg'][0]
        assert pair['pitch_delta'] == b0['peak_deg'][1] - cpu['peak_deg'][1]
        assert all(finite(value) for value in cpu['peak_deg'])
        rows.append({**cpu, 'seed': cpu['name']})
    return rows


def baseline_decision(proposal, selection, results, references, cpu_rows):
    """Numerical retention/headroom only; raw qualification is a separate gate."""
    from review_fixed_force_evaluation import retention
    selected = selection['selected']
    assert selection['verified'] and [c['chart'] for c in selected] == list(CHARTS)
    assert len({c['label'] for c in selected}) == 2
    for condition in selected:
        assert {k: condition[k] for k in ('label', 'chart', 'gamma')} in proposal['conditions']
    assert set(results) == {(c['label'], panel) for c in selected for panel in PANELS}
    old_labels = ('guard_B0', 'guard_B1_route', 'guard_B1_withdraw')
    assert set(references) == {(label, panel) for label in old_labels for panel in PANELS}
    for (label, panel), record in references.items():
        checked = validated_result(record, cases_for(proposal, panel))
        assert checked['summary']['complete'] and checked['physical'] == checked['design'] == len(checked['runs'])
    assert len(cpu_rows) == 28 and [row['seed'] for row in cpu_rows] == [case['seed'] for case in cases_for(proposal, 'legacy')]
    bounds = proposal['capability_and_learning_gate']['nondegradation']
    panels, eligibility = {}, {}
    for condition in selected:
        label = condition['label']; passed = True
        for panel in PANELS:
            cases = cases_for(proposal, panel)
            result = validated_result(results[label, panel], cases)
            comparisons = {old: retention(result['runs'], references[old, panel]['runs'], bounds, panel == 'legacy') for old in old_labels}
            if panel == 'legacy':
                comparisons['original_CPU'] = retention(result['runs'], cpu_rows, bounds, True)
            gates = dict(complete=result['summary']['complete'], all_physical_design=result['physical'] == result['design'] == len(cases),
                         preserve_every_old_classical_success=all(c['passed'] for c in comparisons.values()))
            if panel == 'legacy':
                gates['all28_success'] = result['summary']['success_count'] == 28
            panel_passed = all(gates.values()); passed = passed and panel_passed
            panels[label + '/' + panel] = dict(summary=result['summary'], physical=result['physical'], design=result['design'], episodes=len(cases),
                                               comparisons=comparisons, gates=gates, eligible=panel_passed)
        eligibility[label] = passed
    strong = [(old, references[old, 'controlled']['summary']) for old in old_labels]
    strong += [(c['label'], panels[c['label'] + '/controlled']['summary']) for c in selected if eligibility[c['label']]]
    assert all(s['complete'] and finite(s['mean_yaw_score_deg']) for _, s in strong)
    best = min(s['mean_yaw_score_deg'] for _, s in strong)
    count = max(34, max(s['success_count'] for _, s in strong))
    attainable = best > .05 and all(eligibility.values())
    return dict(verified=True, evaluation_episodes=328, unique_development_cases=164, selected=selected,
                panels=panels, selected_chart_eligibility=eligibility, metric_retention_passed=all(eligibility.values()),
                strongest_controlled_references=[dict(label=label, summary=s) for label, s in strong],
                B_controlled_J_deg=best, learned_controlled_success_min=count, learned_controlled_J_upper_deg=min(.85 * best, best - .05),
                original_metric_gate_numerically_attainable=attainable, independent_raw_received=False, baseline_qualified=False,
                stop_before_learning=True, new_learning_admitted=False, formal5_admitted=False,
                required_matched_learner_parking='original B1 parking', required_matched_learner_initialization='fresh matched actor/critic/optimizer/RMS/RNG and information/authority/budget; no old weights',
                remaining_original_gates=proposal['capability_and_learning_gate']['remaining_original_gates'],
                limits='Metrics and retention on reused development164 only. New independent raw receipt is required. Round377 withdrawal-parking PPO receipt is not this original-parking qualification; no final ID/OOD, learned benefit or formal5 admission.')


def verify_inputs(out=OUT):
    proposal = read(out / 'proposal.json')
    for mapping in (proposal['source_sha256'], proposal['input_sha256']):
        for name, value in mapping.items():
            assert sha(ROOT / name) == value, 'Frozen input/source changed: ' + name
    old_proposal = read(OLD / 'proposal.json'); source = read(OLD / 'evaluation_source.json')
    gate = read(OLD / 'baseline_gate.json'); raw = read(OLD / 'baseline_raw_review.json'); cpu = read(OLD / 'CPU_legacy_pair.json')
    old_hash = sha(OLD / 'proposal.json')
    assert source['verified'] and gate['passed'] and raw['verified'] and cpu['passed'] and cpu['verified_pairs']
    assert source['proposal_sha256'] == gate['proposal_sha256'] == cpu['proposal_sha256'] == old_hash
    assert raw['baseline_gate_sha256'] == sha(OLD / 'baseline_gate.json')
    assert raw['completion_sha256'] == gate['completion_sha256'] == sha(OLD / 'baseline/completion.json')
    assert gate['CPU_legacy_pair_sha256'] == sha(OLD / 'CPU_legacy_pair.json')
    for mapping in (source['source_sha256'], gate['input_sha256'], cpu['input_sha256']):
        for name, value in mapping.items():
            assert sha(ROOT / name) == value, 'Old qualified source/metric input changed: ' + name
    assert proposal['capability_and_learning_gate']['nondegradation'] == old_proposal['nondegradation']
    for panel, count in zip(PANELS, (96, 40, 28)):
        assert cases_for(proposal, panel) == cases_for(old_proposal, panel) and len(cases_for(proposal, panel)) == count
    selection_cases = cases_for(proposal, 'selection')
    assert selection_cases == [{k: row[k] for k in ('seed', 'scenario')} for row in read(SELECTION)['runs']]
    assert len(selection_cases) == 32 and not ({c['seed'] for c in selection_cases} & {c['seed'] for panel in PANELS for c in cases_for(proposal, panel)})
    references = {}
    for condition in old_proposal['classical_conditions']:
        for panel in PANELS:
            path = OLD / 'baseline' / (condition['label'] + '_' + panel + '_aggregate.json')
            references[condition['label'], panel] = validated_result(read(path), cases_for(proposal, panel))
            assert {k: v for k, v in references[condition['label'], panel].items() if k != 'runs'} == {k: v for k, v in gate['panels'][condition['label'] + '/' + panel].items() if k != 'episodes'}
    return proposal, references, cpu_reference_rows(cpu, references['guard_B0', 'legacy']['runs'])


def load_phase(out, proposal, completion, phase, selected=None, runtime=None):
    runtime = (runtime or out).resolve()
    conditions = proposal['conditions'] if phase == 'selection' else selected
    jobs = proposal['selection_jobs'] if phase == 'selection' else proposal['eval_jobs']
    records = [r for r in completion['records'] if r.get('phase', 'selection' if r['panel'] == 'selection' else 'evaluation') == phase]
    expected = [(c['label'], j['panel'], j['batch']) for c in conditions for j in jobs]
    assert [(r['condition'], r['panel'], r['batch']) for r in records] == expected
    results = {}; files = {}
    for condition in conditions:
        for panel in (('selection',) if phase == 'selection' else PANELS):
            cases = cases_for(proposal, panel); ordered = [None] * len(cases)
            for record in records:
                if record['condition'] != condition['label'] or record['panel'] != panel:
                    continue
                job = next(j for j in jobs if j['panel'] == panel and j['batch'] == record['batch'])
                assert record['indices'] == job['indices'] and record['cases'] == job['cases']
                assert record['chart'] == condition['chart'] and record['gamma'] == condition['gamma']
                path = (runtime / record.get('result_path', record.get('path'))).resolve()
                assert path.is_relative_to(runtime), 'Result outside admitted runtime directory'
                value = record.get('result_sha256', record.get('sha256')); assert sha(path) == value
                result = validated_result(read(path), job['cases']); files[str(path)] = value
                for index, row in zip(job['indices'], result['runs']):
                    assert ordered[index] is None; ordered[index] = row
            assert all(row is not None for row in ordered)
            from train_height_comparison import summary
            aggregate = dict(runs=ordered, summary=summary(ordered), physical=sum(r['physical_safety_passed'] for r in ordered), design=sum(r['design_joint_passed'] for r in ordered))
            results[condition['label'] if phase == 'selection' else (condition['label'], panel)] = aggregate
    return results, files


def verify_execution_budget(out, proposal, completion, complete):
    budget = read(ROOT / proposal['prospective_budget']['job_budget_review_path'])
    assert budget['totals']['graph_worldstep_cap'] == proposal['prospective_budget']['task_graph_world_step_cap']
    assert budget['totals']['constructors'] == 68 and budget['totals']['episodes'] == 584
    assert len(completion['records']) == (68 if complete else 32)
    total_graph, total_fd, episodes = 0, 0, 0
    ordered_fd = []
    for index, record in enumerate(completion['records']):
        job = next(j for j in budget['jobs'] if (j['phase'], j['panel'], j['batch']) == (record['phase'], record['panel'], record['batch']))
        assert record['indices'] == job['indices'] and len(record['cases']) == job['num_envs'] <= 10
        assert type(record['graph_calls']) is int and 0 < record['graph_calls'] <= job['deadline_packets']
        assert record['actual_graph_world_steps'] == 40 * job['num_envs'] * record['graph_calls'] <= job['graph_worldstep_cap_per_constructor']
        assert len(record['FD_calls']) <= (10 if index == 0 else 5)
        total_graph += record['actual_graph_world_steps']; total_fd += len(record['FD_calls']); episodes += job['num_envs']
        ordered_fd.extend(record['FD_calls'])
    assert completion['episodes'] == episodes == (584 if complete else 256)
    assert total_graph == completion['actual_graph_world_steps'] <= budget['totals']['graph_worldstep_cap']
    assert total_fd == len(completion['FD_calls']) <= proposal['prospective_budget']['constructor_FD_plan']['total_max_calls']
    assert ordered_fd == completion['FD_calls'], 'Per-constructor FD calls differ from the complete ordered ledger'
    for key in ('new_model_forward_rows', 'new_optimizer_updates', 'new_learning_samples'):
        assert completion[key] == 0


def independent_raw_receipt(path, out, completion_path, selection_path, runtime, admitted):
    receipt = read(path)
    assert receipt['verified'] and receipt['independent_raw_received'] and receipt['episodes'] == 584
    assert receipt['proposal_sha256'] == sha(out / 'proposal.json') and receipt['completion_sha256'] == sha(completion_path)
    assert receipt['selection_review_sha256'] == sha(selection_path)
    assert receipt['receiver_type'] == 'independent_cartesian_classical_raw_v1'
    required = ('original_physical_design', 'original_contact_task', 'actual_cart_command_history_reset', 'original_B1_parking')
    assert all(receipt['scope_checks'][key] is True for key in required)
    assert receipt['receiver_source'] == RAW_RECEIVER_SOURCE, 'A different checker cannot qualify this raw contract'
    assert RAW_RECEIVER_SOURCE in admitted['source_sha256'], 'The actual raw checker is not admitted in this source manifest'
    source = ROOT / RAW_RECEIVER_SOURCE
    assert source.is_file(), 'The actual raw checker has not been implemented'
    assert sha(source) == admitted['source_sha256'][RAW_RECEIVER_SOURCE] == receipt['receiver_sha256']
    completion = read(completion_path)
    assert len(receipt['records']) == len(completion['records']) == 68
    for checked, record in zip(receipt['records'], completion['records']):
        assert (checked['condition'], checked['panel'], checked['batch']) == (record['condition'], record['panel'], record['batch'])
        assert checked['independent_raw_received'] and checked['raw_artifacts'] == record['raw_artifacts'] and checked['raw_artifacts']
        for name, value in checked['raw_artifacts'].items():
            raw_path = (runtime / name).resolve(); assert raw_path.is_relative_to(runtime.resolve()) and sha(raw_path) == value
    return True


def run(out, completion_path, phase, selection_path=None, raw_review=None, runtime=None):
    proposal, references, cpu = verify_inputs(out)
    completion = read(completion_path)
    assert completion['verified'] and completion['proposal_sha256'] == sha(out / 'proposal.json')
    runtime = (runtime or out).resolve()
    assert Path(completion['runtime_path']).resolve() == runtime
    admission = Path(completion.get('source_admission_path', out / 'source_admission.json')).resolve()
    assert admission in ((out / 'source_admission.json').resolve(), runtime / 'source_admission.json')
    assert sha(admission) == completion['source_admission_sha256']
    admitted = read(admission)
    assert admitted['verified'] and admitted['proposal_sha256'] == sha(out / 'proposal.json')
    assert str(Path(__file__).resolve().relative_to(ROOT)) in admitted['source_sha256']
    for name, value in admitted['source_sha256'].items():
        assert sha(ROOT / name) == value, 'New admitted source changed: ' + name
    verify_execution_budget(out, proposal, completion, phase == 'baseline')
    if phase == 'selection':
        results, files = load_phase(out, proposal, completion, 'selection', runtime=runtime)
        review = selection_decision(proposal, results); destination = out / 'selection_metric_review.json'
    else:
        assert selection_path is not None
        selection = read(selection_path)
        assert selection['proposal_sha256'] == sha(out / 'proposal.json') and completion['selection_review_sha256'] == sha(selection_path)
        selection_results, _ = load_phase(out, proposal, completion, 'selection', runtime=runtime)
        rebuilt = selection_decision(proposal, selection_results)
        assert rebuilt['selected'] == selection['selected'] and rebuilt['candidates'] == selection['candidates']
        for record in completion['records']:
            if record['phase'] == 'evaluation':
                assert record['selection_review_sha256'] == sha(selection_path)
        results, files = load_phase(out, proposal, completion, 'evaluation', selection['selected'], runtime)
        review = baseline_decision(proposal, selection, results, references, cpu)
        assert completion['episodes'] == 584 and len(completion['records']) == 68
        assert completion['actual_graph_world_steps'] <= proposal['prospective_budget']['task_graph_world_step_cap']
        assert len(completion['FD_calls']) <= proposal['prospective_budget']['constructor_FD_plan']['total_max_calls']
        if raw_review is not None:
            review['independent_raw_received'] = independent_raw_receipt(raw_review, out, completion_path, selection_path, runtime, admitted)
            review['raw_review_sha256'] = sha(raw_review)
        review['baseline_qualified'] = review['metric_retention_passed'] and review['original_metric_gate_numerically_attainable'] and review['independent_raw_received']
        review['baseline_entry_blocked'] = not review['baseline_qualified']
        review['stop_before_learning'] = True  # A fresh scientific protocol remains separately necessary.
        destination = out / 'baseline_gate.json'
    review.update(proposal_sha256=sha(out / 'proposal.json'), completion_sha256=sha(completion_path), source_admission_sha256=sha(admission), runtime_path=str(runtime), input_sha256=files,
                  CPU28_reference_source=str(CPU_REFERENCE.relative_to(ROOT)), CPU28_reference_sha256=sha(CPU_REFERENCE), CPU28_reference_mode='direct frozen public legacy28 JSON values; no peak-delta reconstruction',
                  reviewer_sha256=sha(__file__), new_physics_steps=0, new_FD_calls=0, new_model_forward_rows=0, new_optimizer_updates=0)
    assert not destination.exists(), 'Never overwrite an old receipt'
    destination.write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return review


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--check-inputs', action='store_true')
    parser.add_argument('--phase', choices=('selection', 'baseline'))
    parser.add_argument('--completion', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--raw-review', type=Path)
    parser.add_argument('--runtime', type=Path)
    args = parser.parse_args()
    if args.check_inputs:
        verify_inputs(args.output); print('PASS current proposal and source-bound historical development references; no new baseline qualification')
    else:
        assert args.phase and args.completion
        review = run(args.output, args.completion, args.phase, args.selection, args.raw_review, args.runtime)
        print(json.dumps({k: review[k] for k in ('verified', 'baseline_qualified', 'new_learning_admitted', 'formal5_admitted')}, ensure_ascii=False))
