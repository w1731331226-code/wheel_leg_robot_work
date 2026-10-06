"""Registered development control screen; formal learning gates remain separate."""
import json
import math

import numpy as np

from gpu_wheel_component import OUT, FIELDS, verify
from score_reference_learning import load_rows
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json


def nondegradation(candidate, reference, nd):
    failures = []
    for a,b in zip(candidate['runs'], reference['runs']):
        assert a['seed'] == b['seed']
        if not b['success']: continue
        flags = []
        if not a['success']: flags.append('lost_success')
        if a['velocity_rmse'] is None or a['velocity_rmse'] > nd['velocity_multiplier']*b['velocity_rmse']+nd['velocity_add_m_s']: flags.append('velocity')
        if a['arrival_s'] is None or a['arrival_s'] > nd['arrival_multiplier']*b['arrival_s']+nd['arrival_add_s']: flags.append('arrival')
        if a['scenario']['terrain'] == 'legacy' and any(a['peak_deg'][j] > b['peak_deg'][j]+nd['legacy_roll_pitch_add_deg'] for j in (0,1)): flags.append('legacy_attitude')
        if flags: failures.append(dict(case=a['seed'],flags=flags))
    return failures


def paired(a,b):
    complete = a['summary']['complete'] and b['summary']['complete']
    return dict(success_gain=a['summary']['success_count']-b['summary']['success_count'],
                Jpsi_reduction=b['summary']['mean_yaw_score_deg']-a['summary']['mean_yaw_score_deg'] if complete else None,
                lost_success_cases=[x['seed'] for x,y in zip(a['runs'],b['runs']) if y['success'] and not x['success']],
                gained_success_cases=[x['seed'] for x,y in zip(a['runs'],b['runs']) if not y['success'] and x['success']])


def unit():
    nd = dict(velocity_multiplier=1.05,velocity_add_m_s=.005,arrival_multiplier=1.05,arrival_add_s=.05,legacy_roll_pitch_add_deg=.1)
    row = dict(seed=1,success=True,velocity_rmse=1.,arrival_s=2.,scenario=dict(terrain='legacy'),peak_deg=[0.,0.,0.])
    original = dict(runs=[row]); assert not nondegradation(original,original,nd)
    worse = dict(runs=[{**row,'success':False,'velocity_rmse':2.,'arrival_s':3.,'peak_deg':[1.,0.,0.]}])
    assert nondegradation(worse,original,nd)[0]['flags'] == ['lost_success','velocity','arrival','legacy_attitude']


def run():
    unit(); p = verify(); old = json.loads((ROOT/p['parent_proposal']).read_text()); gate = p['primary_gate']
    assert sha(ROOT/gate['nondegradation_source']) == gate['nondegradation_source_sha256']
    nd = json.loads((ROOT/gate['nondegradation_source']).read_text())['nondegradation']; assert nd == gate['nondegradation']
    c = json.loads((OUT/'completion.json').read_text()); ledger = json.loads((OUT/'completed_jobs.json').read_text())
    assert c['completed_evaluations'] == ledger['completed_evaluations'] == p['budget_first_episode_evaluations'] == 1632
    assert c['records'] == ledger['records'] and c['training_updates'] == 0 and c['source_contract_sha256'] == sha(OUT/'source_contract.json')
    expected = {(m['seed'],condition,allocator,panel) for m in p['models'] for condition in p['conditions'] for allocator in p['allocators'] for panel in p['panels']}
    results = {}; summaries = {}; recording = {}; allcase_stats = {}; first_steps = 0
    for job in c['records']:
        key = (job['seed'],job['condition'],job['allocator'],job['panel']); assert key in expected and key not in results
        path = OUT/job['path']; assert sha(path) == job['sha256']; d = load_rows(path,old['cases'][job['panel']]); assert len(d['runs']) == job['evaluations']
        values = []; qualified = True
        for r in d['runs']:
            s = np.asarray(r['wheel_projection_stats']); a = np.asarray(r['leg_room_stats']); w = np.asarray(r['steering_stats'])
            assert s.shape == (len(FIELDS),) and np.isfinite(s).all() and np.all(s >= -1e-12)
            assert s[0] == s[1]+s[2] and s[0] == a[0] == w[0]
            # Invalid attempts remain in the data and reject qualification.
            if s[2]: qualified = False
            else:
                assert s[0] == r['physical_steps'] == r['physical_evidence_steps'] > 0
                assert np.max(s[15:16]) <= 1e-9 and np.max(s[16:18]) <= 1e-6 and np.max(s[18:20]) <= 1e-12 and s[21] == 0
                assert np.isclose((s[6]+s[7])/2,s[8]+s[9],rtol=1e-12,atol=1e-9)
                np.testing.assert_allclose(s[[3,6,12,13,14]],w[[2,3,4,5,6]],rtol=1e-12,atol=1e-9)
                np.testing.assert_allclose(s[[4,5]],s[10],rtol=1e-12,atol=1e-9)
                assert abs(s[14]/s[1]-r['mean_residual_lambda']) < 1e-12
                assert a[6] == a[8] == a[9] == 0
                if job['allocator'] == 'uniform_original':
                    assert s[8] == 0 and s[11] == 0 and w[7] <= 1e-9
                    np.testing.assert_array_equal(s[4:6],s[6:8])
                if job['condition'] == 'zero_leg_fixed_assist': assert a[3] == a[4] == a[5] == 0
            values.append(s); first_steps += r['physical_steps']
            allcase_stats['/'.join(map(str,key))+'/'+str(r['seed'])] = dict(valid_substeps=int(s[1]),selected_common_RMS_Nm=math.sqrt(s[8]/s[1]) if s[1] else None,selected_differential_RMS_Nm=math.sqrt(s[9]/s[1]) if s[1] else None,invalid_attempts=int(s[2]))
        total = np.sum(values,axis=0); name = '/'.join(map(str,key)); results[key] = d; recording[key] = qualified
        summaries[name] = dict(success=d['summary']['success_count'],physical=d['physical'],design=d['design'],complete=d['summary']['complete'],Jpsi=d['summary']['mean_yaw_score_deg'],recording_qualified=qualified,
            selected_common_RMS_Nm=math.sqrt(total[8]/total[1]) if total[1] else None,selected_differential_RMS_Nm=math.sqrt(total[9]/total[1]) if total[1] else None,
            original_differential_RMS_Nm=math.sqrt(total[10]/total[1]) if total[1] else None,mean_signed_differential_gain_Nm=total[11]/total[1] if total[1] else None)
    assert set(results) == expected
    references = {panel:[] for panel in p['panels']}
    for filename,value in gate['classical_reference_files'].items():
        path = ROOT/filename; assert sha(path) == value; panel = path.stem.rsplit('_',1)[1]
        references[panel].append((path.stem,load_rows(path,old['cases'][panel])))
    panels = {}; leg = {}
    for panel,n in p['panels'].items():
        pairs = []; increments = []
        for model in p['models']:
            seed = model['seed']; original = results[(seed,'zero_leg_fixed_assist','uniform_original',panel)]; component = results[(seed,'zero_leg_fixed_assist','component_wheels',panel)]
            comp = paired(component,original); failures = {}
            for label,ref in [('concurrent_uniform',original),*references[panel]]:
                failures[label] = nondegradation(component,ref,nd)
            valid = component['summary']['complete'] and component['physical'] == component['design'] == n and recording[(seed,'zero_leg_fixed_assist','component_wheels',panel)]
            pairs.append(dict(seed=seed,**comp,nondegradation_failures=failures,qualified=valid and not any(failures.values())))
            learned = results[(seed,'learned_room_leg_fixed_assist','component_wheels',panel)]
            inc = paired(learned,component); valid_leg = learned['summary']['complete'] and learned['physical'] == learned['design'] == n and recording[(seed,'learned_room_leg_fixed_assist','component_wheels',panel)]
            increments.append(dict(seed=seed,**inc,qualified=valid_leg,nondegradation_failures=nondegradation(learned,component,nd)))
        yaw = all(x['Jpsi_reduction'] is not None for x in pairs) and sum(x['Jpsi_reduction'] > 0 for x in pairs) >= 2 and sum(x['Jpsi_reduction'] for x in pairs) > 0
        panels[panel] = dict(pairs=pairs,passed=all(x['qualified'] for x in pairs) and yaw)
        leg[panel] = dict(pairs=increments,consistent_increment=all(x['qualified'] and not x['nondegradation_failures'] and x['Jpsi_reduction'] is not None and x['Jpsi_reduction'] > 0 for x in increments))
    passed = all(x['passed'] for x in panels.values())
    atomic_json(OUT/'review.json',dict(verified=True,evaluations=1632,unique_development_cases=136,training_updates=0,recorded_first_episode_physical_substeps=first_steps,
        summary=summaries,per_case=allcase_stats,allocation_reference_gate=panels,frozen_leg_increment=leg,allocation_reference_qualified=passed,
        proposal_sha256=sha(OUT/'proposal.json'),completion_sha256=sha(OUT/'completion.json'),source_contract_sha256=sha(OUT/'source_contract.json'),reviewer_sha256=sha(__file__),
        reviewer_dependency_sha256={'wheelleg_warp/score_reference_learning.py':sha(ROOT/'wheelleg_warp/score_reference_learning.py')},
        unevaluated_original_requirements=gate['unevaluated_original_requirements'],
        limits='Development fixed-policy context intervention, not independent generalization, matched new learning, full raw physics reconstruction or formal superiority. Zero-leg model labels are numerical repeats. First-terminal counts are not total simulator work. Original lambda is not new per-wheel acceptance; parent scalar identity deviation is expected in component mode.',
        decision=gate['positive'] if passed else gate['negative']))
    print('PASS1632 original-rule and accepted-command review; reference gate',passed,{k:v['passed'] for k,v in panels.items()},flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--self-check',action='store_true'); args = parser.parse_args()
    unit() if args.self_check else run()
