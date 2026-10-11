"""All currentcontrolled gains/losses and saved contact-mode witnesses; no new rollouts."""
import json
import numpy as np
from train_fixed_force_study import OUT
from review_yaw_sector import ROOT,sha
from score_reference_learning import load_rows
from analyze_reward_failures import flags
from analyze_task_mode_witness import centre_metrics,unit,COL
from path_qualification_draft import conditions
from dashboard.live_env import atomic_json


def run():
    unit();assert not (OUT/'task_opportunity_audit.json').exists()
    p=json.loads((OUT/'proposal.json').read_text());review=json.loads((OUT/'evaluation_review.json').read_text());assert review['verified'] and review['candidate_benefit_branch_closed']
    assert review['completion_sha256']==sha(OUT/'models/completion.json')
    baseline=json.loads((OUT/'baseline_raw_review.json').read_text());assert baseline['completion_sha256']==sha(OUT/'baseline/completion.json')
    data={};locations={};geometry={};hashes={}
    for kind in ('baseline','models'):
        completion=json.loads((OUT/kind/'completion.json').read_text())
        for entry in completion['records']:
            if entry['panel']!='controlled':continue
            job=next(j for j in p['eval_jobs'] if j['panel']=='controlled' and j['batch']==entry['batch'])
            f=OUT/kind/entry['path'];assert sha(f)==entry['sha256'];rows=load_rows(f,job['cases'])['runs'];hashes[str(f.relative_to(ROOT))]=sha(f)
            gf=f.parent/'geometry.json';assert sha(gf)==entry['checked']['geometry_sha256'];geometry[str(f.parent)]=json.loads(gf.read_text());hashes[str(gf.relative_to(ROOT))]=sha(gf)
            for w,row in enumerate(rows):
                assert row['success']==(not any(flags(row).values()))
                key=(entry['condition'],row['seed']);assert key not in data;data[key]=row;locations[key]=(f.parent,w)
    assert len(data)==360
    classical=[c['label'] for c in p['classical_conditions']];cases=[c['seed'] for j in p['eval_jobs'] if j['panel']=='controlled' for c in j['cases']]
    union={case for case in cases if any(data[label,case]['success'] for label in classical)}
    pairs={};gains=[]
    for seed in p['seeds']:
        for arm in p['arms']:
            label=f'{arm}_{seed}';good={case for case in cases if data[label,case]['success']}
            gained=sorted(good-union);lost=sorted(union-good)
            assert len(good)-len(union)==len(gained)-len(lost)
            pairs[label]=dict(success=len(good),gained_over_classical_union=gained,lost_classical_union=lost)
            gains.extend((label,case) for case in gained)
    gained_cases=sorted({case for _,case in gains});selected=gains+[(label,case) for case in gained_cases for label in classical]
    witnesses=[];steps=0
    for label,case in selected:
        row=data[label,case];directory,w=locations[label,case];geo=geometry[str(directory)];boxes=geo['boxes'][w]
        assert row['scenario']['terrain']=='legacy' and len(boxes)==1
        f=directory/row['complete_trace']['path'];assert sha(f)==row['complete_trace']['sha256'];hashes[str(f.relative_to(ROOT))]=sha(f)
        with np.load(f,allow_pickle=False) as z:
            assert z['columns'].tolist()==COL;t=z['trace'];assert len(t)==row['physical_steps']
        steps+=len(t);side='left' if row['scenario']['height_l'] else 'right';box=boxes[0]
        witnesses.append(dict(condition=label,case=case,success=row['success'],flags=[k for k,v in flags(row).items() if v],
            target_height_m=row['target_leg_m'],obstacle_height_m=max(row['scenario']['height_l'],row['scenario']['height_r']),speed_m_s=row['scenario']['speed'],
            peak_deg=row['peak_deg'],mean_body_height_tracking_error_m=row['height_rmse_m'],minimum_design_margin_rad=row['min_active_design_margin_rad'],
            centre_lane=centre_metrics(t,box,side),sampled_geometry=conditions(t,box,side,1 if row['scenario']['speed']>0 else -1)))
    learned=[w for w in witnesses if w['condition'] not in classical]
    atomic_json(OUT/'task_opportunity_audit.json',dict(round=349,verified=True,controlled_evaluation_rows=360,classical_success_union=sorted(union),classical_union_failures=sorted(set(cases)-union),
        model_pairs=pairs,learned_success_witnesses=len(gains),learned_unique_gain_cases=gained_cases,unresolved_cases=sorted(set(cases)-union-set(gained_cases)),
        detailed_trajectory_count=len(witnesses),saved_physical_steps_read=steps,witnesses=witnesses,
        gained_witness_centre_descriptors={name:sum(w['centre_lane']['descriptor']==name for w in learned) for name in ('centre_inside_entire_sampled_span','partial_centre_lane','centre_lateral_for_entire_span')},
        gained_witness_sampled_geometry_prerequisite_passed=sum(w['sampled_geometry']['geometry_prerequisite'] for w in learned),input_sha256=hashes,
        main_review_sha256=sha(OUT/'evaluation_review.json'),auditor_sha256=sha(__file__),centre_helper_sha256=sha(ROOT/'wheelleg_warp/analyze_task_mode_witness.py'),geometry_helper_sha256=sha(ROOT/'wheelleg_warp/path_qualification_draft.py'),
        new_physics_steps=0,new_learning_samples=0,formal5_admitted=False,old_candidate_benefit_branch_closed=True,
        interpretation='All40cases/all6models retained. Gains areexistence witnesses underthe ORIGINALcontact/taskpredicate,notnewheldout evidence,deployable modeloracle,algorithm advantage orcentredloadedtraversal. Centreoutside alone doesnotprove tyre bypass/cheating. No success here doesnotprove unreachable.',
        decision='Close furtherflat-only wheeloff/half/reflection tuning:knownB0 solvesflat andthere isno learningadvantage.350review originalasymmetric-contact capability/retention tradeoff andthephysicalmeaning ofgainwitnesses before anynewmethod. Do notweakenbaseline,retroactivelyreplaceoldgates orchoosewinningseed.'))
    print('PASS349 full360ledger',len(gains),'learnedwins/',len(gained_cases),'cases;',len(witnesses),'savedtraces;0physics',flush=True)


if __name__=='__main__':run()
