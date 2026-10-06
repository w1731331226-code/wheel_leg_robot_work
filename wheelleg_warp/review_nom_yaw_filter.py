"""Paired full-task/axis-gate review of the fixed alpha probe, no new evaluation."""
import json
import numpy as np

from nom_yaw_filter_probe import OUT, rec
from score_reference_learning import load_rows
from analyze_reward_failures import flags


def full_flags(row):
    f=flags(row)
    attitude=row['relative_peak_deg'] if row['scenario']['relative_attitude'] else row['peak_deg']
    # Explicit axes preserve the original5deg envelope even if yaw already failed.
    f['roll_axis']=attitude[0]>5
    f['pitch_axis']=attitude[1]>5
    return f


def added_failures(before,after):
    assert before.keys()==after.keys()
    return [k for k in before if after[k] and not before[k]]


def unit():
    assert added_failures(dict(yaw=True,roll=False),dict(yaw=True,roll=True))==['roll']
    assert not added_failures(dict(yaw=True,roll=True),dict(yaw=False,roll=True))
    assert added_failures(dict(success_gate=False),dict(success_gate=True))==['success_gate']


def compare(original,latest):
    rows=[]
    for a,b in zip(original['runs'],latest['runs']):
        assert a['seed']==b['seed'] and a['scenario']==b['scenario']
        fa,fb=full_flags(a),full_flags(b)
        assert a['success']==(not any(fa.values())) and b['success']==(not any(fb.values()))
        rows.append(dict(case=a['seed'],height_m=a['target_leg_m'],original_success=a['success'],latest_success=b['success'],
            original_flags=fa,latest_flags=fb,new_axis_or_task_failures=added_failures(fa,fb),
            original_peak_deg=a['peak_deg'],latest_peak_deg=b['peak_deg'],
            velocity_rmse_change=b['velocity_rmse']-a['velocity_rmse'],arrival_s_change=b['arrival_s']-a['arrival_s'],
            stop_distance_change=b['stop_distance_m']-a['stop_distance_m'],tail_speed_change=b['tail_speed_m_s']-a['tail_speed_m_s'],
            height_rms_change=b['height_rmse_m']-a['height_rmse_m']))
    lost=[r['case'] for r in rows if r['original_success'] and not r['latest_success']]
    gained=[r['case'] for r in rows if not r['original_success'] and r['latest_success']]
    regressions=[dict(case=r['case'],flags=r['new_axis_or_task_failures']) for r in rows if r['new_axis_or_task_failures']]
    jp=original['summary']['mean_yaw_score_deg'];jq=latest['summary']['mean_yaw_score_deg']
    original_peak=float(np.mean([r['original_peak_deg'][2] for r in rows]));latest_peak=float(np.mean([r['latest_peak_deg'][2] for r in rows]))
    clean=all(r['physical_safety_passed'] and r['design_joint_passed'] for r in latest['runs'])
    return dict(original_success=original['summary']['success_count'],latest_success=latest['summary']['success_count'],
        lost=lost,gained=gained,new_axis_or_task_failures=regressions,physical_design_all_pass=clean,
        original_mean_yaw_score_deg=jp,latest_mean_yaw_score_deg=jq,yaw_score_reduction_deg=jp-jq,yaw_score_relative_reduction=(jp-jq)/jp,
        original_mean_yaw_peak_deg=original_peak,latest_mean_yaw_peak_deg=latest_peak,yaw_peak_reduction_deg=original_peak-latest_peak,
        original_yaw_threshold_failures=sum(r['original_flags']['yaw'] for r in rows),latest_yaw_threshold_failures=sum(r['latest_flags']['yaw'] for r in rows),
        original_roll_threshold_failures=sum(r['original_flags']['roll_axis'] for r in rows),latest_roll_threshold_failures=sum(r['latest_flags']['roll_axis'] for r in rows),
        fixed_variant_qualification=clean and not lost and not regressions and latest_peak<original_peak,
        rows=rows)


def run():
    unit();p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'completion.json').read_text())
    previous=json.loads((OUT/'round177_data_review.json').read_text());assert previous['verified'] and c['verified'] and c['comparison_records']==160
    contract=json.loads((OUT/'source_contract.json').read_text());assert all(rec.sha(rec.ROOT/n)==v for n,v in contract['source_sha256'].items())
    jobs={(j['condition'],j['classical']):j for j in c['records']};pairs={};inputs={}
    for noise in ('clean','noisy'):
        for label in ('B0','B1-route'):
            if noise=='clean':
                ref=p['reuse_references'][label];left=rec.ROOT/ref['path'];assert rec.sha(left)==ref['sha256']
            else:
                j=jobs[('original_noisy',label)];left=OUT/j['path'];assert rec.sha(left)==j['sha256']
            j=jobs[('latest_'+noise,label)];right=OUT/j['path'];assert rec.sha(right)==j['sha256']
            for file in (left,right):inputs[str(file.relative_to(rec.ROOT))]=rec.sha(file)
            a,b=load_rows(left,p['cases']),load_rows(right,p['cases'])
            pairs[noise+'/'+label]=compare(a,b)
    qualification=all(v['fixed_variant_qualification'] for v in pairs.values())
    regressions=sum(len(v['new_axis_or_task_failures']) for v in pairs.values())
    assert not qualification and regressions==12
    assert all(v['physical_design_all_pass'] and not v['lost'] for v in pairs.values())
    assert all(v['yaw_peak_reduction_deg']>0 and v['yaw_score_reduction_deg']>0 for v in pairs.values())
    result=dict(verified=True,round=178,pairs=pairs,unique_development_cases=20,comparison_rows=160,paired_comparison_records=80,
        added_axis_gate_comparison_records=regressions,unique_added_roll_failure_cases=sorted({r['case'] for v in pairs.values() for r in v['new_axis_or_task_failures']}),
        fixed_variant_qualification=qualification,source_sha256=rec.sha(__file__),input_sha256=inputs,
        completion_sha256=rec.sha(OUT/'completion.json'),data_review_sha256=rec.sha(OUT/'round177_data_review.json'),new_evaluations=0,training_updates=0,
        decision='Reject alpha1 as fixed full-task improvement/baseline replacement: yaw improves but original5deg roll envelope gains violations in all4 comparisons. Keep code/data as diagnostic; no alpha/gain sweep or new PPO justified.',
        interpretation='Removing lowpass trades yaw and roll in this model: not a universal no-benefit result, not pure delay attribution. Original cases already failing yaw remain failures; explicit axes expose hidden component deterioration. One fixed synthetic-noise realization/case, no population or training-seed inference.',
        next='179 use coupled attitude/available reference evidence to choose one supported next mechanism, not rescue fixedalpha1.180 direction review/cleanup. Full method/5seed/fresh generalization/manuscript remains incomplete.')
    rec.atomic_json(OUT/'round178_pair_review.json',result)
    for k,v in pairs.items():print(k,'success',v['original_success'],v['latest_success'],'yawpeak reduction',v['yaw_peak_reduction_deg'],
        'J reduction',v['yaw_score_reduction_deg'],'new flags',v['new_axis_or_task_failures'],'qualified',v['fixed_variant_qualification'],flush=True)


if __name__=='__main__':run()
