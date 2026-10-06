"""Update current task opportunity and qualify historical learning witnesses."""
import json
from collections import Counter
import run_phase_support_qualification as runner
from score_reference_learning import load_rows
from review_nom_yaw_filter import full_flags
from dashboard.live_env import atomic_json

ROOT, sha = runner.ROOT, runner.sha
BASE = ROOT/'wheelleg_warp/results/paper_recovery_20261004'
OUT = runner.OUT


def category(classical, learned, other):
    if classical: return 'current_classical_success'
    if learned: return 'historical_main_learning_witness'
    if other: return 'closed_context_witness_only'
    return 'no_success_in_inspected_conditions'


def run():
    assert category(True, True, True) == 'current_classical_success'
    assert category(False, True, True) == 'historical_main_learning_witness'
    assert category(False, False, True) == 'closed_context_witness_only'
    assert category(False, False, False) == 'no_success_in_inspected_conditions'
    p = runner.verify(); closure = json.loads((OUT/'round193_pair_review.json').read_text())
    assert closure['verified'] and closure['overall_engineering_qualification']
    main = BASE/'reference_budget_learning_v1'; mp = json.loads((main/'proposal.json').read_text())
    progress = json.loads((main/'main_progress.json').read_text())
    assert progress['status'] == 'complete' and progress['trained_policy_steps'] == 2400000
    cases = p['panels']['regular']+p['panels']['controlled']; assert len(cases) == 136
    assert mp['regular']+mp['controlled'] == cases
    inputs = {}; ledger = {c['seed']: dict(case=c['seed'], scenario=c['scenario'],
        current_classical_successes=[], historical_main_successes=[], closed_context_successes=[],
        current_classical_failures={}, historical_main_failures={}) for c in cases}
    def read(file, expected, expected_sha=None):
        digest = sha(file)
        if expected_sha is not None: assert digest == expected_sha
        inputs[str(file.relative_to(ROOT))] = digest
        return load_rows(file, expected)['runs']
    completion = json.loads((OUT/'completion.json').read_text())
    totals = {}
    for job in completion['records']:
        if job['panel'] == 'legacy': continue
        expected = [p['panels'][job['panel']][i] for i in job['indices']]
        rows = read(OUT/job['path'], expected, job['sha256'])
        for row in rows:
            entry = ledger[row['seed']]
            if row['success']: entry['current_classical_successes'].append(job['law'])
            else: entry['current_classical_failures'][job['law']] = [k for k, v in full_flags(row).items() if v]
        totals[job['law']] = totals.get(job['law'], 0)+sum(r['success'] for r in rows)
    main_rows = 0
    for seed in mp['seeds']:
        for arm in mp['arms']:
            for panel in ('regular', 'controlled'):
                rows = read(main/'runs'/arm/str(seed)/(panel+'_final.json'), mp[panel]); main_rows += len(rows)
                label = arm+'/'+str(seed)
                for row in rows:
                    entry = ledger[row['seed']]
                    if row['success']: entry['historical_main_successes'].append(label)
                    else: entry['historical_main_failures'][label] = [k for k, v in full_flags(row).items() if v]
    role = BASE/'reference_role_probe_v1'; rp = json.loads((role/'proposal.json').read_text())
    rc = json.loads((role/'completion.json').read_text()); role_rows = 0
    for job in rc['records']:
        if job['arm'] != 'consistent_pair': continue
        rows = read(role/job['path'], rp['cases'], job['sha256']); role_rows += len(rows)
        for row in rows:
            if row['success']:
                ledger[row['seed']]['closed_context_successes'].append('consistent_pair/'+job['noise']+'/'+job['classical'])
    witness = BASE/'task_mode_witness_v1'
    mode = json.loads((witness/'physical_mode_analysis.json').read_text())
    application = json.loads((witness/'passage_evidence_application.json').read_text())
    assert mode['verified'] and application['verified'] and mode['evaluations'] == application['records'] == 205
    assert sha(ROOT/'wheelleg_warp/analyze_task_mode_witness.py') == mode['source_sha256']
    assert sha(ROOT/'wheelleg_warp/apply_passage_evidence.py') == application['source_sha256']
    assert sha(witness/'parallel_passage_evidence_contract.json') == application['contract_sha256']
    path_rows = {(r['controller'], r['case']): r for r in application['rows']}
    geometry_rows = []
    for row in mode['low_success_records_detail']:
        if row['case'] not in (6301001, 6301003, 6301005, 6301007): continue
        qualification = path_rows[(row['controller'], row['case'])]
        assert qualification['original_success'] and qualification['evidence']['prescribed_path'] is False
        for relative in (f'runs/{row["controller"]}/result.json', f'runs/{row["controller"]}/geometry.json',
                         f'runs/{row["controller"]}/case_{row["case"]}.npz'):
            file = witness/relative
            assert sha(file) == mode['input_sha256'][relative]
            inputs[str(file.relative_to(ROOT))] = mode['input_sha256'][relative]
        geometry_rows.append(dict(controller=row['controller'], case=row['case'],
            original_success=True, centre_descriptor=row['centre_lane']['descriptor'],
            centre_inside_fraction=row['centre_lane']['centre_inside_fraction'],
            secondary_ground_status=qualification['ground_rollover']['status'],
            secondary_air_status=qualification['airborne_passage']['status']))
    assert len(geometry_rows) == 7
    for file in (OUT/'round193_pair_review.json', OUT/'completion.json', main/'proposal.json',
                 main/'main_progress.json', role/'proposal.json', role/'completion.json',
                 role/'round183_pair_review.json', witness/'physical_mode_analysis.json',
                 witness/'passage_evidence_application.json', witness/'parallel_passage_evidence_contract.json'):
        inputs[str(file.relative_to(ROOT))] = sha(file)
    for entry in ledger.values():
        entry['category'] = category(bool(entry['current_classical_successes']),
            bool(entry['historical_main_successes']), bool(entry['closed_context_successes']))
    unresolved = sorted(i for i, e in ledger.items() if not e['current_classical_successes'])
    counts = dict(Counter(e['category'] for e in ledger.values()))
    assert totals == {'B0': 128, 'B1-route': 127} and main_rows == 1632 and role_rows == 80
    assert unresolved == [6301001, 6301003, 6301005, 6301007, 6301033, 6301035, 6301037, 6301039]
    assert counts == dict(current_classical_success=128, historical_main_learning_witness=4,
                          closed_context_witness_only=1, no_success_in_inspected_conditions=3)
    recommendation = dict(status='Mechanism-study recommendation for195 review only; no new protocol/source/runner admitted',
        question='Do existing policy corrections improve disturbance recovery when applied exposure cannot be reduced by route change?',
        contrast='A controlled body-wrench pulse with a frozen amplitude/time profile identical across controllers and independent '
                 'of robot lateral route; compare phase B0/B1-route and all3 frozen V6 seeds. Existing obstacle task remains primary.',
        prospective_limit='At most200 no-learning diagnostic episodes:8 declared reference profiles×5 registered heights×5 controllers. '
                          'Not a registered budget:195 must decide scope, profile construction, correct torque frame/point, '
                          'fair actor/controller information, lifecycle and gates before any source or execution.',
        profile_rule='If admitted, derive one fixed excitation scale/profile set from reference classical contact traces before '
                     'viewing diagnostic outcomes; never rescale/tune signs/timing to rescue a model. Replayed body wrench is '
                     'an operational stimulus, not dynamically equivalent to terrain contact or pure exogenous natural force.',
        information='No contact truth, body wrench label, terrain geometry, future pulse or hidden parameter in Actor. Keep same '
                    '39-packet and nominal/control limits for all methods; frozen-model context transfer is diagnostic only.',
        method_hypothesis='A causal proprioceptive disturbance estimate coordinating virtual-wrench corrections could improve '
                          'yaw recovery while preserving roll, speed, height and design. No observer, memory architecture or '
                          'new algorithm is selected/admitted by these witnesses.',
        future_learning_evidence='If the contrast warrants a concrete method, derive its information/execution/constraint contract '
                                'and distinction from existing work, then test same-budget3 trained seeds against strong classical '
                                'and generic6D residual PPO plus a contribution-specific matched ablation.5 formal seeds and '
                                'fresh independent generalization remain required.',
        stopping='No automatic PPO, observer/history scaffolding or more anchor heuristics. Negative transfer closes this diagnostic '
                 'claim, not all learning; positive transfer still does not establish learned superiority or a new method.')
    atomic_json(OUT/'round194_learning_need.json', dict(verified=True, round=194,
        current_classical_development_totals=totals, development_cases=136, category_counts=counts,
        current_classical_unresolved_case_ids=unresolved, cases=sorted(ledger.values(), key=lambda e:e['case']),
        historical_primary_learning_rows=main_rows, inspected_closed_context_rows=role_rows,
        historical_low_witness_path_qualification=geometry_rows, recommendation=recommendation,
        source_sha256=sha(__file__), input_sha256=inputs, new_evaluations=0, training_updates=0,
        decision='Current classical union covers128/136. Regular6300019 is not shared infeasibility. '
                 'Four low-height historical main wins do not establish equal-exposure recovery/centred passage; '
                 'upper6301033 has one noisy closed classical-context witness, so no global impossibility claim. '
                 'No longPPO or new-method novelty admitted.',
        limits='Original task labels retained. Secondary path contract does not replace primary or prove continuous/loaded tyre '
               'passage.Historical learned policies were not trained on phase Nom; no matched learning contrast. '
               'Counts are repeated conditions on the same development IDs, not independent tests/training seeds.',
        next='195 direction review and verified redundancy cleanup; decide the bounded equal-exposure mechanism study. '
             'Complete paper contribution/trained comparisons/formal5seeds/independenttests/statistics/PPO timing/manuscript remain open.'))
    print('PASS current328 qualification reused;272 development rows/1632 historical main/80 closed-context rows;', counts, flush=True)
    print('CURRENT unresolved8', unresolved, 'LOW path witnesses', len(geometry_rows), 'NO new evaluation/training', flush=True)


if __name__ == '__main__':
    run()
