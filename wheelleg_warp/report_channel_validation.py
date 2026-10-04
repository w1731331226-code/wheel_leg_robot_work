"""Export verified finite-panel tables; no metric selection or significance test."""
from pathlib import Path
import csv,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'wheelleg_warp/results/paper_recovery_20261004/channel_validation_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()

def run():
    review=json.loads((S/'independent_review.json').read_text())
    assert sha(S/'reviewed_jobs_snapshot.json')==review['ledger_sha256']
    snapshot=json.loads((S/'reviewed_jobs_snapshot.json').read_text())
    assert snapshot['completed_episodes']==review['verified_case_records']
    records={v['path']:v for v in snapshot['records']}
    for path,rec in records.items():assert sha(S/path)==rec['sha256']
    rows=[]
    for key,r in review['decoded'].items():
        label,category=key.split('/');stats=r['summary']
        rows.append(dict(controller=label,category=category,cases=stats['total'],success=stats['success_count'],
            physical=r['physical'],design=r['design'],full_trajectory_yaw_metric=stats['complete'],
            yaw_score_deg=stats['mean_yaw_score_deg'],early_terminated=stats['total']-r['reasons'].get('completed',0)))
        if category=='pressure':
            for factor,p in r['factors'].items():
                rows.append(dict(controller=label,category='pressure_'+factor,cases=p['summary']['total'],success=p['summary']['success_count'],
                    physical=p['physical'],design=p['design'],full_trajectory_yaw_metric=p['summary']['complete'],
                    yaw_score_deg=p['summary']['mean_yaw_score_deg'],early_terminated=None))
    fields=['controller','category','cases','success','physical','design','full_trajectory_yaw_metric','yaw_score_deg','early_terminated']
    with (S/'verified_tables.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    effects=review['available_primary_model_effects'];overall=None;byseed=[]
    if len(effects)==6:
        overall=dict(mean_success_difference_pp=100*float(np.mean([v['fraction_difference'] for v in effects])),
            model_effects=effects,mean_design_failure_difference=float(np.mean([v['design_failure_difference'] for v in effects])),
            mean_physical_failure_difference=float(np.mean([v['physical_failure_difference'] for v in effects])))
        for seed in [1609,1610,1611]:
            selected=[v for v in effects if v['model'].endswith(str(seed))];assert len(selected)==2
            byseed.append(dict(seed=seed,paired_reward_model_mean_difference_pp=100*float(np.mean([v['fraction_difference'] for v in selected]))))
    out=dict(verified_snapshot=True,complete=review['complete'],case_records=review['verified_case_records'],jobs=review['verified_jobs'],
        regular_primary_complete=len(effects)==6,regular_primary=overall,seed_clusters=byseed,table_rows=len(rows),
        scope='Fixed96-case finite panel and six fixed controllers,3seed clusters. No training-seed significance/p-values or aggregate pressure continuous-domain claim; blank metrics remain blank.',
        table_counting='Pressure aggregate and factor rows describe same48 cases and must not be summed. Table rows are not evaluation jobs or extra episodes.',
        all_categories_completed_required_for_final_conclusion=True,snapshot_sha256=review['ledger_sha256'],
        review_sha256=sha(S/'independent_review.json'),table_sha256=sha(S/'verified_tables.csv'),source_sha256=sha(Path(__file__)))
    (S/'table_report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('TABLES',len(rows),'rows',review['verified_case_records'],'verified case records; full',review['complete'],'primary6ready',len(effects)==6)
    if overall:print('PRIMARY FINITE PANEL',overall,'seed clusters',byseed)

if __name__=='__main__':run()
