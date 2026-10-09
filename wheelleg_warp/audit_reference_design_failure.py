"""Read the stopped attempt and kinematics only; no controller or physics query."""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import wheelleg_sim as sim
from joint_reference_mapping import select,LOW

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/reference_learning_prequalification_v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    result=OUT/'design_failure_audit.json';assert not result.exists()
    admission=json.loads((OUT/'main_source_admission.json').read_text())
    assert all(sha(ROOT/f)==h for f,h in admission['source_sha256'].items())
    old=json.loads((OUT/'training_failure_review.json').read_text());bad=old['violating_episode']
    protocol=json.loads((OUT/'proposal.json').read_text());case=protocol['training_banks']['32033']['3'][32]
    assert case['seed']==bad['seed'] and case['scenario']==bad['scenario']
    grouped={};matched=[];total=0;violations=[];inputs={}
    for directory in sorted((OUT/'training').iterdir()):
        if not directory.is_dir():continue
        file=directory/('interrupted_episodes.json' if (directory/'failure.json').exists() else 'episodes.json')
        data=json.loads(file.read_text());inputs[str(file.relative_to(ROOT))]=sha(file)
        rows=data['episodes'];total+=len(rows)
        for index,row in enumerate(rows):
            assert row['physical_evidence_steps']==row['physical_steps']>0
            assert row['design_joint_contract']=='active-1p4-v1'
            assert row['design_joint_passed']==(row['min_active_design_margin_rad']>=0)
            if not row['design_joint_passed']:violations.append((directory.name,index,row))
            if row['curriculum_stage']==3 and row['seed']==bad['seed'] and row['scenario']==bad['scenario']:
                matched.append(dict(run=directory.name,episode_index=index,episode=row))
        for label in ('low115','other_height'):
            panel=[r for r in rows if (r['scenario']['stand_height_m']==.115)==(label=='low115')]
            if panel:
                margin=np.array([r['min_active_design_margin_rad'] for r in panel])
                grouped[f'{directory.name}/{label}']=dict(episodes=len(panel),physical_pass=sum(r['physical_safety_passed'] for r in panel),
                    design_pass=sum(r['design_joint_passed'] for r in panel),min_design_margin_rad=float(margin.min()),
                    near_boundary_le001rad=int((margin<=.01).sum()),margin_quantiles_rad=np.quantile(margin,[0,.25,.5,.75,1]).tolist())
    assert total==2319 and len(violations)==1 and len(matched)==2
    assert violations[0][0]=='M_ref3_32033' and violations[0][2]==bad
    assert matched[0]['episode']['design_joint_passed'] and not matched[1]['episode']['design_joint_passed']
    assert all(r['episode']['physical_safety_passed'] and not r['episode']['success'] for r in matched)
    # This necessary height bound has no statement about actual leg orientation or dynamics.
    mean,difference,_=select(np.array([.115]),np.zeros(1),np.zeros((1,3)))
    np.testing.assert_array_equal(mean,[.115]);np.testing.assert_array_equal(difference,[0.])
    examples=[]
    for angle in (0.,.16):
        qa,qb=sim.ik(.115*math.cos(angle),.115*math.sin(angle))
        fk=sim.fk_joints(qa,qb)
        np.testing.assert_allclose(fk['leg_len'],.115,rtol=0,atol=2e-15)
        np.testing.assert_allclose(fk['phi5']+math.pi/2,angle,rtol=0,atol=2e-15)
        examples.append(dict(leg_length_m=fk['leg_len'],leg_angle_rad=angle,active_q_rad=[qa,qb],active_margin_rad=1.4-max(abs(qa),abs(qb))))
    assert examples[0]['active_margin_rad']>.35 and examples[1]['active_margin_rad']<0 and all(e['leg_length_m']>LOW for e in examples)
    assert bad['physical_safety_passed'] and bad['geometric_margin_passed']
    assert min(bad['min_actual_A_leg_m'],bad['min_actual_B_leg_m'])>LOW
    assert bad['radial_guard_limited_steps']==0 and bad['max_requested_radial_guard_N']==bad['max_applied_radial_guard_N']
    for name in ('wheelleg_warp/native/environment.py','wheelleg_warp/native/controller.py','wheelleg_warp/joint_reference_control.py',
                 'wheelleg_warp/reference_role_control.py','wheelleg_warp/joint_reference_mapping.py','wheelleg_warp/phase_support_probe.py',
                 'wheelleg_ppo/tools/wheelleg_sim.py','wheelleg_warp/audit_reference_design_failure.py'):
        inputs[name]=sha(ROOT/name)
    report=dict(round=324,verified=True,total_training_episodes_read=total,design_violation_count=1,matched_training_episodes=matched,
        descriptive_margin_groups=grouped,static_kinematic_examples=examples,reference_floor_m=LOW,
        failed_actual_length_clearance_m=min(bad['min_actual_A_leg_m'],bad['min_actual_B_leg_m'])-LOW,
        failed_joint_excess_rad=-bad['min_active_design_margin_rad'],joint_excess_float32_ulps=float(-bad['min_active_design_margin_rad']/np.spacing(np.float32(1.4))),
        source_chain=['reference mapping bounds tracking mean/difference and leg targets only',
            'project_leg_angle checks standing-branch reference pose against1.4rad',
            'radial guard acts on length/rate and islimited byoutward torque headroom',
            'project_bounds/final clamp enforce motor command capacity, notnext joint position',
            'collect_physical scores actual four activejoint positions afterevery0.5ms step;1.4rad gate differsfrom mechanical eightjoint limits'],
        findings=['failure geometry/torque/mechanical gates passed while active design gatefailed,so no contradiction between recorded flags',
            'height floor alone isinsufficient;CPU static counterexample isnotreconstructed failed pose orcontactcertificate',
            'nominal vertical115mm pose has.351rad active margin,so115mm target itself isnotan active1.4rad singular boundary',
            'samecase previous episode hadpositive designmargin butboth hadtaskfailure;trainedpolicy/RNG differences forbid fixedpolicy reproducibility orcausal claims',
            'failed radialguard had0clippedsteps;increasing itsforce cap isnotjustified bythisevidence',
            'base infeasibility234 andwholeepisode extrema cannotbe temporally aligned withjointpeak fromsaveddata'],
        limits=['no terminalstopped_q/peakjoint identity/peak-time control/reference/contact trace forfailed later trainingepisode',
            'no zero/classic baseline orsame-seedU6 episode onthisscenario;thirdU6 neverran',
            'training episodes usechangingpolicy/stages/exploration andarenotindependent scientifictest cases'],
        experiment_recommendation_for_round325='one bounded prospective failedcase zero/classic vsfrozenfailedpolicy deterministic+same frozenstochastic sequence diagnostic, withfirst-crossing state/pose/request/command/caps/contact capture;notrestart1.2M orselectfour-run winners',
        new_physics_graph_steps=0,new_controller_queries=0,new_training_samples=0,formal5_admitted=False,source_sha256=inputs)
    result.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('PASS324:2319 archivedepisodes,2samecase,1designfailure;geometry/activeposition separate;0physics/query/learning')


if __name__=='__main__':run()
