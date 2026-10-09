"""Receive all492 current baselines before admitting any fresh scientific learner."""
import json
from train_fixed_force_study import OUT
from train_height_comparison import summary
from score_reference_learning import load_rows
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run():
    assert not (OUT/'baseline_gate.json').exists()
    p=json.loads((OUT/'proposal.json').read_text());a=json.loads((OUT/'evaluation_source.json').read_text());c=json.loads((OUT/'baseline/completion.json').read_text())
    assert c['verified'] and c['episodes']==492 and len(c['records'])==30 and a['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    panels={};inputs={}
    for condition in p['classical_conditions']:
        for panel in ('regular','controlled','legacy'):
            cases=[case for job in p['eval_jobs'] if job['panel']==panel for case in job['cases']]
            rows=[None]*len(cases)
            for entry in c['records']:
                if entry['condition']!=condition['label'] or entry['panel']!=panel:continue
                f=OUT/'baseline'/entry['path'];assert sha(f)==entry['sha256'];job=next(j for j in p['eval_jobs'] if j['panel']==panel and j['batch']==entry['batch'])
                result=load_rows(f,job['cases']);inputs[str(f.relative_to(ROOT))]=sha(f)
                for index,row in zip(entry['indices'],result['runs']):assert rows[index] is None;rows[index]=row
            assert all(r is not None for r in rows) and [r['seed'] for r in rows]==[r['seed'] for r in cases]
            s=summary(rows);panels[condition['label'],panel]=dict(summary=s,physical=sum(r['physical_safety_passed'] for r in rows),design=sum(r['design_joint_passed'] for r in rows),episodes=len(rows))
            atomic_json(OUT/'baseline'/f'{condition["label"]}_{panel}_aggregate.json',dict(runs=rows,**panels[condition['label'],panel]))
    controlled=[panels[condition['label'],'controlled'] for condition in p['classical_conditions']]
    complete=all(r['summary']['complete'] for r in controlled);j=min(r['summary']['mean_yaw_score_deg'] for r in controlled) if complete else None
    physical=all(r['physical']==r['design']==r['episodes'] for r in panels.values())
    legacy=panels['guard_B0','legacy']['summary']['success_count']==28
    attainable=complete and j is not None and j>=.05
    passed=physical and legacy and attainable
    atomic_json(OUT/'baseline_gate.json',dict(round=335,passed=passed,proposal_sha256=sha(OUT/'proposal.json'),completion_sha256=sha(OUT/'baseline/completion.json'),
        all_current_physical_design=physical,guard_B0_legacy28_retained=legacy,primary_absolute005_attainable=attainable,best_current_controlled_J_deg=j,
        controlled_success_target=max(34,max(r['summary']['success_count'] for r in controlled)),panels={f'{label}/{panel}':r for (label,panel),r in panels.items()},input_sha256=inputs,
        new_learning_admitted=False,formal5_admitted=False,limits='source/results/task flags received;raw dense independentreview/main source lock stillneeded. CPU28retention reference check notfullnumeric trajectory pair.'))
    print('BASELINE335 received492;gate',passed,'controlledJ',j,'controlledcounts',[r['summary']['success_count'] for r in controlled])


if __name__=='__main__':run()
