"""Independent full200 evidence and frozen utility gates; never simulates."""
import json
from pathlib import Path
import numpy as np
import joint_reference_adapter as adapter
from joint_reference_classical import dispatch
from train_height_comparison import summary
from score_reference_learning import load_rows
from complete_contact_witness import check_files
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/nonzero_reference_mechanism_v1'


def run():
    assert not (OUT/'review.json').exists()
    p=json.loads((OUT/'proposal.json').read_text());a=json.loads((OUT/'runner_admission.json').read_text());c=json.loads((OUT/'completion.json').read_text())
    assert c['verified'] and c['evaluations']==p['evaluations']==200
    assert c['admission_sha256']==sha(OUT/'runner_admission.json') and a['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    data={}; streams=bytes_total=steps_total=0; request_checks=0
    for record in c['records']:
        job=next(j for j in p['jobs'] if j['batch']==record['batch']);assert record['indices']==job['indices']
        file=OUT/record['path'];assert sha(file)==record['sha256']
        rows=load_rows(file,job['cases'])['runs']
        checked=check_files(file.parent,job['cases'],{r['seed']:r for r in rows})
        assert checked['physics_steps']==record['first_episode_world_steps']
        steps_total+=checked['physics_steps']
        for row in rows:
            fields=['complete_trace','reference_actor_trace']
            if record['arm']=='old_B0':fields=['complete_trace','zero_actor_trace']
            if record['arm'].startswith('joint_'):fields.append('joint_reference_trace')
            else:fields.extend(['gyro_trace','role_trace','phase_trace','parking_trace'])
            for field in fields:
                f=file.parent/row[field]['path'];assert sha(f)==row[field]['sha256'];streams+=1;bytes_total+=f.stat().st_size
            actor_key='zero_actor_trace' if record['arm']=='old_B0' else 'reference_actor_trace'
            with np.load(file.parent/row[actor_key]['path'],allow_pickle=False) as z:
                actor=z['trace'].copy()
            dim=row[actor_key]['dimension'];assert actor.shape==(int(np.ceil(row['physical_steps']/40)),481+dim)
            expected=dispatch(actor[:,:481],record['arm'],p['yaw_config'])
            np.testing.assert_array_equal(actor[:,481:],expected)
            request_checks+=len(actor)
            if record['arm'].startswith('joint_'):
                with np.load(file.parent/row['joint_reference_trace']['path'],allow_pickle=False) as z:
                    trace=z['trace'];adapter.check_log(trace,z['role'],z['phase'])
                    canonical,_=adapter.decode(expected,'M3',len(expected))
                    k=(trace[:,1].astype(int)-1)//40
                    np.testing.assert_array_equal(trace[:,10:16],canonical[k])
                with np.load(file.parent/row['complete_trace']['path'],allow_pickle=False) as z:
                    np.testing.assert_array_equal(trace[:,1],z['trace'][:,1])
                    np.testing.assert_allclose(trace[:,5:7],z['trace'][:,29:31],rtol=0,atol=1e-12)
        data.setdefault(record['arm'],[]).extend(rows)
        print('REVIEWED307',record['arm'],record['batch'],'streams',streams,flush=True)
    assert len(data)==5 and all(len(x)==40 for x in data.values())
    assert steps_total==c['first_episode_world_steps'] and sum(r['actual_graph_world_steps'] for r in c['records'])==c['actual_graph_world_steps']
    assert c['actual_graph_world_steps']<=p['physical_graph_world_step_budget'] and len(c['baseline_constructor_FD_calls'])<=p['constructor_FD_budget']
    order=[r['seed'] for r in data['old_B0']]
    assert all([r['seed'] for r in rows]==order for rows in data.values())
    scores={arm:summary(rows) for arm,rows in data.items()}
    baseline_names=('old_B0','old_B1_route','joint_yaw')
    union={r['seed'] for arm in baseline_names for r in data[arm] if r['success']}
    best=max(scores[arm]['success_count'] for arm in baseline_names)
    pairs={};gates={}
    for arm in ('joint_center','joint_lower'):
        successes={r['seed'] for r in data[arm] if r['success']}
        pairs[arm]={base:dict(lost=[r['seed'] for r in data[base] if r['success'] and r['seed'] not in successes],
            gained=[r['seed'] for r in data[base] if not r['success'] and r['seed'] in successes]) for base in baseline_names}
        gates[arm]=dict(success_best_plus2=len(successes)>=best+2,none_lost_from_baseline_union=union<=successes,
            all_physical_design=all(r['physical_safety_passed'] and r['design_joint_passed'] for r in data[arm]),
            meanJ_lower_than_joint_yaw=scores[arm]['mean_yaw_score_deg'] is not None and
                scores[arm]['mean_yaw_score_deg']<scores['joint_yaw']['mean_yaw_score_deg'])
    passed=any(all(g.values()) for g in gates.values())
    atomic_json(OUT/'review.json',dict(verified=True,round=307,scores=scores,pairs=pairs,gates=gates,
        fixed_reference_utility_gate_passed=passed,closed_fixed_variants=not passed,
        baseline_success_union=sorted(union),records=200,unique_development_cases=40,
        raw_streams=streams,raw_bytes=bytes_total,actor_request_rows_checked=request_checks,
        full_saved_actor_actions_equal_frozen_public_dispatch=True,reference_canonical_effective_filter_targets_checked=True,
        completion_sha256=sha(OUT/'completion.json'),reviewer_sha256=sha(__file__),
        actual_graph_world_steps=c['actual_graph_world_steps'],first_episode_world_steps=steps_total,
        baseline_constructor_FD_calls=len(c['baseline_constructor_FD_calls']),wall_seconds=c['wall_seconds'],
        new_simulation=0,new_training_samples=0,formal_PPO_admitted=False,
        limits='Five feedbackarms on40development cases,not200independent trials/trainingseeds orfinalOOD. Same inputinterface/requestalgorithm,not identicalphysicaltrajectories. Failureofthesefixedfeedback variants doesnotproveall learnedreference policiesimpossible.',
        next='308 diagnose only saved lostcases andcoupledroll/yaw/height/actuation witnesses;do notrefitclosedfixedfeedback orlaunchPPO withoutdistincttestablemechanism.310deepreview/cleanup.'))
    print('DONE307 full200 frozen gate',passed,'scores',scores,'pairs',pairs,flush=True)


if __name__=='__main__':
    run()
