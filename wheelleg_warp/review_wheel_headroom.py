"""Original task and registered same-state wheel-loss screening, no promotion."""
import json
import math

import numpy as np

from wheel_headroom_attribution import OUT, FIELDS, verify
from review_yaw_sector import ROOT, sha, success
from dashboard.live_env import atomic_json


def metrics(stats):
    s = np.asarray(stats, dtype=float)
    assert s.shape == (len(FIELDS),) and np.isfinite(s).all() and s[1] > 0
    return dict(valid_substeps=int(s[1]),recoverable_left_RMS_Nm=math.sqrt(s[6]/s[1]),
                hip_share_of_abs_rejected=s[9]/s[7] if s[7] > 0 else None,
                wheel_loss_RMS_Nm=math.sqrt(s[5]/s[1]),
                filtered_RMS_Nm=math.sqrt(s[3]/s[1]),accepted_RMS_Nm=math.sqrt(s[4]/s[1]))


def unit():
    s = np.zeros(len(FIELDS)); s[1] = 4; s[6] = .0004; s[7] = 2; s[9] = .4
    m = metrics(s); assert m['recoverable_left_RMS_Nm'] == .01 and m['hip_share_of_abs_rejected'] == .2
    s[7] = 0; assert metrics(s)['hip_share_of_abs_rejected'] is None


def run():
    unit(); p = verify(); parent = json.loads((ROOT/p['parent_proposal']).read_text())
    contract = json.loads((OUT/'source_contract.json').read_text())
    assert contract['unit_sha256'] == sha(OUT/'unit.json') and contract['owner_sha256'] == sha(OUT/'owner_check.json')
    assert p['original_review_sha256'] == sha((ROOT/p['parent_proposal']).parent/'review.json')
    c = json.loads((OUT/'completion.json').read_text()); ledger = json.loads((OUT/'completed_jobs.json').read_text())
    assert c['completed_episodes'] == ledger['completed_episodes'] == p['budget_gpu_episodes'] == 816
    assert c['records'] == ledger['records'] and c['training_updates'] == 0
    assert c['source_contract_sha256'] == sha(OUT/'source_contract.json')
    expected = {(m['seed'],condition,panel) for m in p['models'] for condition in p['conditions'] for panel in p['panels']}
    seen = set(); summaries = {}; case_metrics = {}; eligible = {panel: [] for panel in p['panels']}
    for job in c['records']:
        key = (job['seed'], job['condition'], job['panel']); assert key in expected and key not in seen; seen.add(key)
        path = OUT/job['path']; assert sha(path) == job['sha256']; d = json.loads(path.read_text())
        cases = parent['cases'][job['panel']]; assert len(d['runs']) == len(cases) == job['episodes']
        values = []; scores = []
        for r, case in zip(d['runs'], cases):
            assert r['seed'] == case['seed'] and r['scenario'] == case['scenario'] and r['success'] == success(r)
            physical = min(r['min_actual_A_leg_m'],r['min_actual_B_leg_m']) >= r['geometric_limit_m'] and r['min_eight_joint_margin_rad'] >= 0 and max(r['max_actual_torque_excess_Nm'],r['max_command_torque_excess_Nm']) <= 1e-6
            assert r['physical_safety_passed'] == physical and r['design_joint_passed'] == (r['min_active_design_margin_rad'] >= 0)
            assert r['physical_evidence_steps'] == r['physical_steps'] > 0
            s = np.asarray(r['headroom_stats']); a = np.asarray(r['leg_room_stats']); w = np.asarray(r['steering_stats'])
            assert s.shape == (len(FIELDS),) and a.shape == (18,) and w.shape == (8,) and np.isfinite(s).all()
            assert s[0] == s[1] == a[0] == a[13] == w[0] == r['physical_steps'] and s[2] == 0
            assert np.max(s[11:13]) <= p['validation']['accepted_identity_tolerance_Nm']
            assert s[13] <= p['validation']['lambda_order_tolerance'] and s[14] <= 1e-9 and s[18] <= 1e-9
            assert np.isclose(s[7],s[8]+s[9],rtol=1e-12,atol=1e-9)
            np.testing.assert_allclose(s[[3,4,16]],w[[2,3,6]],rtol=1e-12,atol=1e-9)
            assert a[6] == a[8] == a[9] == 0 and w[7] <= 1e-9
            assert abs(a[15]/a[0]-r['mean_residual_lambda']) < 1e-12
            if job['condition'] == 'zero_leg_fixed_assist':
                assert s[15] <= p['validation']['zero_leg_extra_coupling_tolerance_Nm'] and a[3] == a[4] == a[5] == 0
            label = '/'.join(map(str,key)); case_metrics[label+'/'+str(r['seed'])] = metrics(s); values.append(s)
            scores.append(r['rms_deg'][2]*math.sqrt(r['duration_s']/(3.5+1.5*r['task_goal_progress_m']/abs(r['scenario']['speed']))) if r['reason'] == 'completed' else None)
        assert d['summary']['success_count'] == sum(r['success'] for r in d['runs'])
        assert d['physical'] == sum(r['physical_safety_passed'] for r in d['runs']) and d['design'] == sum(r['design_joint_passed'] for r in d['runs'])
        assert d['summary']['complete'] == all(s is not None for s in scores)
        if d['summary']['complete']: assert math.isclose(d['summary']['mean_yaw_score_deg'],sum(scores)/len(scores),abs_tol=1e-12)
        m = metrics(np.sum(values, axis=0)); gate = p['decision_gate']
        qualified = d['summary']['complete'] and d['physical'] == d['design'] == len(cases)
        authority = qualified and m['recoverable_left_RMS_Nm'] >= gate['recoverable_left_steering_RMS_Nm_min'] and m['hip_share_of_abs_rejected'] is not None and m['hip_share_of_abs_rejected'] >= gate['leg_coupling_share_of_absolute_rejected_steering_min']
        summaries[label] = dict(**m,success=d['summary']['success_count'],physical=d['physical'],design=d['design'],Jpsi=d['summary']['mean_yaw_score_deg'],authority_screen_passed=authority)
        if job['condition'] == 'learned_room_leg_fixed_assist' and authority: eligible[job['panel']].append(job['seed'])
    assert seen == expected
    passed = all(len(v) >= p['decision_gate']['required_seeds_per_panel'] for v in eligible.values())
    atomic_json(OUT/'review.json',dict(verified=True,completed_gpu_episodes=816,unique_development_cases=136,training_updates=0,
        summary=summaries,per_case=case_metrics,eligible_seeds_by_panel=eligible,authority_screen_passed=passed,
        proposal_sha256=sha(OUT/'proposal.json'),completion_sha256=sha(OUT/'completion.json'),source_contract_sha256=sha(OUT/'source_contract.json'),reviewer_sha256=sha(__file__),
        limits='Read-only diagnostic torque counterfactual, not a controller intervention, independent generalization, full raw physics reconstruction or learning benefit. Pooled valid-substep RMS and additive absolute losses; RMS/energies are not additive. Actual float32 increments remain separately in original steering_stats.',
        decision=p['decision_gate']['if_pass'] if passed else p['decision_gate']['otherwise']))
    print('PASS816 original task/physical/design/recording review; authority screen',passed,eligible,flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--self-check',action='store_true'); args = parser.parse_args()
    unit() if args.self_check else run()
