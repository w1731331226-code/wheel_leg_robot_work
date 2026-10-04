"""Offline verification of the complete finite yaw-sector development probe."""
from pathlib import Path
import argparse, hashlib, json, math

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def success(r):
    relative = r['scenario']['relative_attitude']
    attitude = r['relative_peak_deg'] if relative else r['peak_deg']
    return (r['reason'] == 'completed' and r['physical_safety_passed'] and r['design_joint_passed']
        and (r['touched_contact_mask'] & r['required_contact_mask']) == r['required_contact_mask']
        and r['terrain_passed'] and r['terrain_exit_passed']
        and max(attitude) <= 5 and r['peak_deg'][2] <= 5
        and (not relative or max(r['peak_deg'][:2]) <= 10)
        and r['velocity_rmse'] <= .2 * abs(r['scenario']['speed'])
        and r['stop_distance_m'] <= .6 and r['tail_speed_m_s'] <= .03
        and r['height_rmse_m'] <= .02 and abs(r['final_mean_fk_leg_m'] - r['target_leg_m']) <= .02)


def run(out):
    reg = json.loads((out/'registration.json').read_text())
    result = json.loads((out/'result.json').read_text())
    ledger = json.loads((out/'completed_jobs.json').read_text())
    assert reg['budget_episodes'] == result['completed_episodes'] == ledger['completed_episodes'] == 320
    assert reg['policy_training_steps'] == result['training_steps'] == 0
    assert result['registration_sha256'] == sha(out/'registration.json')
    assert not reg['independent_test'] and not reg['old_gate_or_final_used']
    assert all(sha(ROOT/name) == value for name,value in reg['source_sha256'].items())
    for model in reg['models']:
        prefix=Path(model['prefix']); checkpoint=model['checkpoint']
        assert sha(prefix.with_suffix('.zip')) == checkpoint['checkpoint_sha256']
        assert sha(prefix.with_suffix('.pkl')) == checkpoint['normalization_sha256']
    labels=['B1']+[f'{m["seed"]}_{c}' for m in reg['models'] for c in reg['conditions']]
    assert [v['path'] for v in ledger['records']] == [label+'.json' for label in labels]
    assert len(set(c['seed'] for c in reg['cases'])) == 32
    stats={}; totals={}; patterns={}
    for label,record in zip(labels,ledger['records']):
        path=out/record['path']; assert sha(path)==record['sha256']
        d=json.loads(path.read_text());rows=d['runs']
        assert len(rows)==record['episodes']==32
        for r,case in zip(rows,reg['cases']):
            assert r['seed']==case['seed'] and r['scenario']==case['scenario']
            assert type(r['success']) is bool and r['success']==success(r)
            assert r['physical_evidence_steps']==r['physical_steps'] and r['physical_steps']>0
            physical=(min(r['min_actual_A_leg_m'],r['min_actual_B_leg_m'])>=r['geometric_limit_m']
                and r['min_eight_joint_margin_rad']>=0
                and max(r['max_actual_torque_excess_Nm'],r['max_command_torque_excess_Nm'])<=1e-6)
            assert r['physical_safety_passed']==physical
            assert r['design_joint_passed']==(r['min_active_design_margin_rad']>=0)
            assert math.isfinite(r['duration_s']) and r['duration_s']>0
            assert r['terrain_evidence_passed']==bool(r['terrain_passed'] and r['terrain_exit_passed'])
            if label!='B1':
                n,opposed,active=r['request_sample_counts']
                assert n>0 and 0<=opposed<=n and 0<=active<=n and 0<=r['actor_packet_clipped_samples']<=n
        assert d['summary']['success_count']==sum(r['success'] for r in rows)
        scores=[]
        for r in rows:
            if r['reason']!='completed':
                scores.append(None); continue
            duration=r['physical_steps']*.0005
            assert math.isclose(duration,r['duration_s'],abs_tol=1e-8)
            assert r['arrival_s'] is not None and duration>=r['arrival_s']+2-1e-8
            ref=3.5+1.5*r['task_goal_progress_m']/abs(r['scenario']['speed'])
            assert math.isfinite(r['rms_deg'][2]) and r['rms_deg'][2]>=0
            scores.append(r['rms_deg'][2]*math.sqrt(duration/ref))
        complete=all(v is not None for v in scores)
        assert d['summary']['total']==32 and d['summary']['complete']==complete
        if complete: assert math.isclose(d['summary']['mean_yaw_score_deg'],sum(scores)/32,abs_tol=1e-12)
        else: assert d['summary']['mean_yaw_score_deg'] is None
        assert d['physical']==sum(r['physical_safety_passed'] for r in rows)
        assert d['design']==sum(r['design_joint_passed'] for r in rows)
        assert result['results'][label]=={k:v for k,v in d.items() if k!='runs'}
        stats[label]=d
        if label!='B1':
            totals[label]=[sum(r['request_sample_counts'][i] for r in rows) for i in range(3)]
        patterns[label]=dict(terrain_missing=sum(not r['terrain_passed'] for r in rows),
            yaw_failed=sum(r['peak_deg'][2]>5 for r in rows),noncompleted=sum(r['reason']!='completed' for r in rows))
    paired=[]
    for m in reg['models']:
        a,s,l=[stats[f'{m["seed"]}_{c}'] for c in reg['conditions']]
        paired.append(dict(seed=m['seed'],sector_minus_all=s['summary']['success_count']-a['summary']['success_count'],
            sector_minus_legs=s['summary']['success_count']-l['summary']['success_count'],
            nondegradation=s['physical']>=a['physical'] and s['design']>=a['design']))
    gate=all(p['sector_minus_all']>0 and p['nondegradation'] for p in paired) and sum(p['sector_minus_legs']>0 for p in paired)>=2
    assert paired==result['paired'] and gate==result['filter_training_candidate_gate']
    review=dict(verified=True,episodes=320,jobs=10,unique_development_cases=32,paired=paired,
        filter_training_candidate_gate=gate,request_sample_counts=totals,failure_patterns=patterns,
        result_sha256=sha(out/'result.json'),registration_sha256=sha(out/'registration.json'),
        scope='Source/model/case/ledger, physical counts, exact success and paired gate verified offline. Finite development probe, not a novel-method or general-safety proof.',
        reviewer_sha256=sha(__file__))
    (out/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in review.items() if k not in ('request_sample_counts','failure_patterns')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
