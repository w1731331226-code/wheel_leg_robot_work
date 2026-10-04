"""Read-only confirmation audit, including natural early-termination failures."""
from pathlib import Path
import json,collections,hashlib
from train_height_comparison import summary
from training_contract import digest
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'wheelleg_warp/results/paper_recovery_20261004/channel_validation_v1'

def original_success(r):
    if r['reason']!='completed':return False
    relative=r['scenario']['relative_attitude'];att=r['relative_peak_deg'] if relative else r['peak_deg']
    return bool(r['physical_safety_passed'] and r['design_joint_passed'] and r['terrain_passed'] and r['terrain_exit_passed']
        and (r['touched_contact_mask']&r['required_contact_mask'])==r['required_contact_mask']
        and max(att)<=5 and r['peak_deg'][2]<=5 and (not relative or max(r['peak_deg'][:2])<=10)
        and r['velocity_rmse'] is not None and r['velocity_rmse']<=.2*abs(r['scenario']['speed'])
        and r['stop_distance_m']<=.6 and r['tail_speed_m_s']<=.03 and r['height_rmse_m'] is not None
        and r['height_rmse_m']<=.02 and abs(r['final_mean_fk_leg_m']-r['target_leg_m'])<=.02)

def run():
    spec=json.loads((S/'protocol.json').read_text());contract=json.loads((S/'source_contract.json').read_text())
    assert contract['protocol_sha256']==digest(S/'protocol.json')
    assert all(digest(ROOT/n)==value for n,value in contract['source_sha256'].items())
    ledger_text=(S/'completed_jobs.json').read_text();jobs=json.loads(ledger_text)
    assert jobs['completed_episodes']==sum(r['episodes'] for r in jobs['records'])
    assert jobs['completed_episodes']<=spec['evaluation_episodes_budget']
    sets={'regular':spec['regular'],'pressure':[r for group in spec['pressure'].values() for r in group],'legacy_regression':spec['legacy_regression']}
    decoded={};early=[];identities=set();primary=[]
    for job in jobs['records']:
        assert digest(S/job['path'])==job['sha256']
        r=json.loads((S/job['path']).read_text());label=r['controller']['label'];category=r['category'];cases=sets[category]
        assert r['protocol_sha256']==contract['protocol_sha256'] and r['source_contract_sha256']==digest(S/'source_contract.json')
        assert (label,category) not in identities;identities.add((label,category))
        rows=r['runs'];assert len(rows)==len(cases)==job['episodes']
        assert [v['seed'] for v in rows]==[c['seed'] for c in cases]
        assert all(v['scenario']==c['scenario'] for v,c in zip(rows,cases))
        assert all(v['physical_evidence_steps']==v['physical_steps']>0 for v in rows)
        assert all(v['success']==original_success(v) for v in rows)
        assert summary(rows)==r['summary']
        assert sum(v['physical_safety_passed'] for v in rows)==r['physical']
        assert sum(v['design_joint_passed'] for v in rows)==r['design']
        for v in rows:
            if v['reason']!='completed':early.append(dict(controller=label,category=category,case=v['seed'],reason=v['reason'],
                duration_s=v['duration_s'],physical_envelope_passed=v['physical_safety_passed'],task_success=v['success']))
        decoded[f'{label}/{category}']=dict(summary=r['summary'],physical=r['physical'],design=r['design'],
            reasons=dict(collections.Counter(v['reason'] for v in rows)),factors=r['factors'])
    for model in spec['models']:
        prefix=f"{model['arm']}_{model['seed']}";a=decoded.get(f'{prefix}_all/regular');b=decoded.get(f'{prefix}_legs_only/regular')
        if a and b:primary.append(dict(model=prefix,success_difference=b['summary']['success_count']-a['summary']['success_count'],
            fraction_difference=(b['summary']['success_count']-a['summary']['success_count'])/96,
            design_failure_difference=a['design']-b['design'],physical_failure_difference=a['physical']-b['physical']))
    complete=jobs['completed_episodes']==4472 and len(jobs['records'])==78
    if complete:
        assert len(primary)==6
        final=json.loads((S/'completion.json').read_text());assert final['completed_episodes']==4472 and final['training_steps']==0
    result=dict(verified_completed_jobs=True,complete=complete,verified_jobs=len(jobs['records']),verified_case_records=jobs['completed_episodes'],
        decoded=decoded,early_terminated_cases=early,available_primary_model_effects=primary,
        complete_primary_estimand=(sum(p['fraction_difference'] for p in primary)/6 if complete else None),
        statistical_scope='Six fixed models from3seed pairs; same96 cases repeated, no independent training-run significance. Pressure/old28 separate, old28 not independent.',
        interpretation='Completed evaluation job means all cases have returned terminal results; summary.complete concerns full arrival/holding trajectory for yaw metric. Timeouts remain task failures even with physical-envelope pass, no zero-imputed yaw score.',
        ledger_sha256=hashlib.sha256(ledger_text.encode()).hexdigest(),source_sha256=digest(Path(__file__)))
    (S/'reviewed_jobs_snapshot.json').write_text(ledger_text)
    atomic_json(S/'independent_review.json',result)
    print('AUDITED',result['verified_jobs'],'jobs',result['verified_case_records'],'case records; complete',complete,'early exits',len(early),flush=True)
    for p in primary:print('PRIMARY AVAILABLE',p,flush=True)

if __name__=='__main__':run()
