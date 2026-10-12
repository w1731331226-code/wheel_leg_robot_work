"""CPU fixtures exercise selection leakage, retention and receipt boundaries."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from review_cartesian_classical_baseline import (OUT, OLD, SELECTION, PANELS, RAW_RECEIVER_SOURCE, CPU_REFERENCE, read, sha,
    cases_for, validated_result, selection_decision, baseline_decision, cpu_reference_rows,
    load_phase, verify_execution_budget, independent_raw_receipt)
from train_height_comparison import summary


def result(rows):
    return dict(runs=rows, summary=summary(rows), physical=sum(r['physical_safety_passed'] for r in rows), design=sum(r['design_joint_passed'] for r in rows))


class BaselineReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal = read(OUT / 'proposal.json')
        cls.selection_rows = read(SELECTION)['runs']
        cls.refs = {(label, panel): read(OLD / 'baseline' / (label + '_' + panel + '_aggregate.json'))
                    for label in ('guard_B0', 'guard_B1_route', 'guard_B1_withdraw') for panel in PANELS}
        cls.cpu = cpu_reference_rows(read(OLD / 'CPU_legacy_pair.json'), cls.refs['guard_B0', 'legacy']['runs'])

    def selection(self):
        records = {}
        factors = {'CartLive3': (1, .8, .7, .7), 'CartReset3': (1, .7, .8, .6)}
        for condition in self.proposal['conditions']:
            rows = copy.deepcopy(self.selection_rows)
            factor = factors[condition['chart']][self.proposal['parameters']['gamma_candidates'].index(condition['gamma'])]
            for row in rows:
                row['rms_deg'][2] *= factor
            records[condition['label']] = result(rows)
        return records

    def full(self, selected):
        records = {}
        for condition in selected['selected']:
            for panel in PANELS:
                rows = copy.deepcopy(self.refs['guard_B0', panel]['runs'])
                for row in rows:
                    row.update(success=True, reason='completed', velocity_rmse=0., arrival_s=.1, stop_distance_m=0., tail_speed_m_s=0., height_rmse_m=0.,
                        final_mean_fk_leg_m=row['target_leg_m'], peak_deg=[0., 0., 1.], relative_peak_deg=[0., 0., 1.], rms_deg=[0., 0., .3],
                        touched_contact_mask=row['required_contact_mask'], terrain_passed=True, terrain_exit_passed=True)
                records[condition['label'], panel] = result(rows)
        return records

    def test_original_metric_and_identity_are_recomputed(self):
        record = result(copy.deepcopy(self.selection_rows))
        validated_result(record, cases_for(self.proposal, 'selection'))
        record['summary']['mean_yaw_score_deg'] += .001
        with self.assertRaises(AssertionError):
            validated_result(record, cases_for(self.proposal, 'selection'))
        record = result(copy.deepcopy(self.selection_rows)); record['runs'][0]['seed'] += 1
        with self.assertRaises(AssertionError):
            validated_result(record, cases_for(self.proposal, 'selection'))
        record = result(copy.deepcopy(self.selection_rows)); record['runs'][0]['duration_s'] = float('nan')
        with self.assertRaises(AssertionError):
            validated_result(record, cases_for(self.proposal, 'selection'))

    def test_independent_gain_and_fixed_tie_break(self):
        decision = selection_decision(self.proposal, self.selection())
        self.assertEqual([c['gamma'] for c in decision['selected']], [1, 2])
        self.assertFalse(decision['baseline_qualified']); self.assertFalse(decision['new_learning_admitted'])

    def test_cpu28_peaks_are_read_directly_and_bridge_corruption_is_rejected(self):
        original = read(CPU_REFERENCE)['runs']
        self.assertEqual([row['peak_deg'] for row in self.cpu], [row['peak_deg'] for row in original])
        bridge = read(OLD / 'CPU_legacy_pair.json'); bridge['records'][0]['roll_delta'] += .001
        with self.assertRaises(AssertionError):
            cpu_reference_rows(bridge, self.refs['guard_B0', 'legacy']['runs'])

    def test_any_selection_incomplete_or_physical_failure_stops(self):
        records = self.selection(); label = self.proposal['conditions'][0]['label']
        rows = records[label]['runs']; rows[0].update(reason='attitude_failure', success=False); records[label] = result(rows)
        with self.assertRaises(AssertionError):
            selection_decision(self.proposal, records)
        records = self.selection(); rows = records[label]['runs']
        rows[0].update(min_actual_A_leg_m=rows[0]['geometric_limit_m'] - .001, physical_safety_passed=False, success=False); records[label] = result(rows)
        with self.assertRaises(AssertionError):
            selection_decision(self.proposal, records)

    def test_complete_retention_is_still_not_raw_or_learning_qualification(self):
        chosen = selection_decision(self.proposal, self.selection())
        decision = baseline_decision(self.proposal, chosen, self.full(chosen), self.refs, self.cpu)
        self.assertTrue(decision['metric_retention_passed']); self.assertTrue(decision['original_metric_gate_numerically_attainable'])
        self.assertLess(decision['B_controlled_J_deg'], .8248525201597937)
        self.assertEqual(decision['learned_controlled_success_min'], 40)
        self.assertAlmostEqual(decision['learned_controlled_J_upper_deg'], min(.85 * decision['B_controlled_J_deg'], decision['B_controlled_J_deg'] - .05))
        self.assertFalse(decision['baseline_qualified']); self.assertFalse(decision['independent_raw_received'])
        self.assertTrue(decision['stop_before_learning']); self.assertFalse(decision['formal5_admitted'])

    def test_lost_old_success_and_cpu_bound_make_selected_chart_ineligible(self):
        chosen = selection_decision(self.proposal, self.selection()); records = self.full(chosen)
        label = chosen['selected'][0]['label']; rows = records[label, 'regular']['runs']
        rows[0].update(success=False, stop_distance_m=.7); records[label, 'regular'] = result(rows)
        decision = baseline_decision(self.proposal, chosen, records, self.refs, self.cpu)
        self.assertFalse(decision['selected_chart_eligibility'][label]); self.assertFalse(decision['original_metric_gate_numerically_attainable'])
        self.assertNotIn(label, [r['label'] for r in decision['strongest_controlled_references']])
        records = self.full(chosen); rows = records[label, 'legacy']['runs']
        rows[0]['velocity_rmse'] = self.cpu[0]['velocity_rmse'] * 1.05 + .005001
        records[label, 'legacy'] = result(rows)
        decision = baseline_decision(self.proposal, chosen, records, self.refs, self.cpu)
        self.assertFalse(decision['panels'][label + '/legacy']['comparisons']['original_CPU']['passed'])
        self.assertFalse(decision['selected_chart_eligibility'][label])

    def test_absolute_headroom_closed_and_no_fallback_gain(self):
        chosen = selection_decision(self.proposal, self.selection()); records = self.full(chosen)
        for condition in chosen['selected']:
            rows = records[condition['label'], 'controlled']['runs']
            for row in rows:
                row['rms_deg'][2] = 0.
            records[condition['label'], 'controlled'] = result(rows)
        decision = baseline_decision(self.proposal, chosen, records, self.refs, self.cpu)
        self.assertEqual(decision['B_controlled_J_deg'], 0.); self.assertFalse(decision['original_metric_gate_numerically_attainable'])
        self.assertEqual(decision['selected'], chosen['selected'])

    def test_job_identity_order_and_runtime_path_cannot_be_substituted(self):
        records = [dict(phase='selection', condition=c['label'], chart=c['chart'], gamma=c['gamma'], panel=j['panel'], batch=j['batch'],
                       indices=j['indices'], cases=j['cases'], result_path='../../outside.json', result_sha256='unused')
                   for c in self.proposal['conditions'] for j in self.proposal['selection_jobs']]
        with tempfile.TemporaryDirectory() as directory:
            swapped = copy.deepcopy(records); swapped[0], swapped[1] = swapped[1], swapped[0]
            with self.assertRaises(AssertionError):
                load_phase(OUT, self.proposal, {'records': swapped}, 'selection', runtime=Path(directory))
            with self.assertRaises(AssertionError):
                load_phase(OUT, self.proposal, {'records': records}, 'selection', runtime=Path(directory))

    def test_fixed_deadlines_extra_constructor_network_and_fd_substitution_are_rejected(self):
        budget = read(OUT / 'job_budget_review.json'); records = []
        for condition in self.proposal['conditions']:
            for job in self.proposal['selection_jobs']:
                frozen = next(j for j in budget['jobs'] if j['phase'] == 'selection' and j['batch'] == job['batch'])
                records.append(dict(phase='selection', panel='selection', batch=job['batch'], indices=job['indices'], cases=job['cases'],
                    graph_calls=frozen['deadline_packets'], actual_graph_world_steps=frozen['graph_worldstep_cap_per_constructor'], FD_calls=[dict(job=len(records), call=i) for i in range(10 if not records else 5)]))
        completion = dict(records=records, episodes=256, actual_graph_world_steps=sum(r['actual_graph_world_steps'] for r in records),
            FD_calls=[value for r in records for value in r['FD_calls']], new_model_forward_rows=0, new_optimizer_updates=0, new_learning_samples=0)
        verify_execution_budget(OUT, self.proposal, completion, False)
        extra = copy.deepcopy(completion); extra['records'].append(extra['records'][-1])
        over = copy.deepcopy(completion); over['records'][0]['graph_calls'] += 1
        network = copy.deepcopy(completion); network['new_model_forward_rows'] = 1
        fd = copy.deepcopy(completion); fd['FD_calls'][0], fd['FD_calls'][1] = fd['FD_calls'][1], fd['FD_calls'][0]
        for changed in (extra, over, network, fd):
            with self.assertRaises(AssertionError):
                verify_execution_budget(OUT, self.proposal, changed, False)

    def test_wrong_raw_checker_unfrozen_checker_and_unimplemented_checker_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory); completion = runtime / 'completion.json'; completion.write_text('{}')
            selection = runtime / 'selection_review.json'; selection.write_text('{}')
            receipt = dict(verified=True, independent_raw_received=True, episodes=584, proposal_sha256=sha(OUT / 'proposal.json'),
                completion_sha256=sha(completion), selection_review_sha256=sha(selection), receiver_type='independent_cartesian_classical_raw_v1',
                scope_checks={key: True for key in ('original_physical_design', 'original_contact_task', 'actual_cart_command_history_reset', 'original_B1_parking')},
                receiver_source='wheelleg_warp/review_fixed_force_evaluation.py', receiver_sha256='irrelevant')
            path = runtime / 'raw_review.json'; path.write_text(json.dumps(receipt))
            with self.assertRaises(AssertionError):
                independent_raw_receipt(path, OUT, completion, selection, runtime, {'source_sha256': {receipt['receiver_source']: 'irrelevant'}})
            receipt['receiver_source'] = RAW_RECEIVER_SOURCE; path.write_text(json.dumps(receipt))
            with self.assertRaises(AssertionError):
                independent_raw_receipt(path, OUT, completion, selection, runtime, {'source_sha256': {}})
            with self.assertRaises(AssertionError):
                independent_raw_receipt(path, OUT, completion, selection, runtime, {'source_sha256': {RAW_RECEIVER_SOURCE: 'irrelevant'}})


if __name__ == '__main__':
    unittest.main()
