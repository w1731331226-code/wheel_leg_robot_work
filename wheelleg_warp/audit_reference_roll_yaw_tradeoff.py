"""Existing upperheight reference counterexamples; no new rollout or fitted gain."""
import json
import numpy as np
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

SOURCE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reference_role_probe_v1'
OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_task_pair_v1'


def witness(folder,row):
    path=folder/row['complete_trace']['path'];assert sha(path)==row['complete_trace']['sha256']
    role=folder/row['role_trace']['path'];assert sha(role)==row['role_trace']['sha256']
    with np.load(path,allow_pickle=False) as a,np.load(role,allow_pickle=False) as b:
        t=a['trace'];r=b['trace'];columns={str(n):i for i,n in enumerate(a['columns'])}
        assert len(t)==len(r)==row['physical_steps'] and np.isfinite(t).all() and np.isfinite(r).all()
        np.testing.assert_array_equal(t[:,1],r[:,1])
        mean_delta=r[:,3]-r[:,2]
        first=np.flatnonzero(mean_delta!=0.)
        yaw=np.flatnonzero(abs(t[:,columns['yaw']])>np.deg2rad(5.))
        result=dict(trace_sha256=sha(path),role_sha256=sha(role),mean_delta_min_m=float(mean_delta.min()),
            mean_delta_max_m=float(mean_delta.max()),first_mean_change_step=int(first[0]+1) if len(first) else None,
            first_yaw5_step=int(yaw[0]+1) if len(yaw) else None)
        if len(yaw):
            i=int(yaw[0]);cmd=t[i,[columns['command_left'],columns['command_right']]]
            bounds=t[i,[columns['pre_command_bound_left'],columns['pre_command_bound_right']]]
            result['yaw_witness']=dict(step=i+1,command=float(t[i,columns['speed_command']]),
                attitude_deg=np.rad2deg(t[i,[columns['roll'],columns['pitch'],columns['yaw']]]).tolist(),
                current_tracking_mean_m=float(r[i,3]),base_command_mean_m=float(r[i,2]),
                left_target_m=float(r[i,7]),right_target_m=float(r[i,8]),
                wheel_speed_rad_s=t[i,[columns['pre_wheel_speed_left'],columns['pre_wheel_speed_right']]].tolist(),
                normal_N=t[i,[columns['left_normal_N'],columns['right_normal_N']]].tolist(),
                command_Nm=cmd.tolist(),command_bound_Nm=bounds.tolist(),
                command_to_bound_ratio=(abs(cmd)/bounds).tolist())
        return result


def run():
    assert not (OUT/'round288_roll_yaw_tradeoff.json').exists()
    p=json.loads((SOURCE/'proposal.json').read_text())
    c=json.loads((SOURCE/'completion.json').read_text())
    prior=json.loads((SOURCE/'round183_pair_review.json').read_text())
    assert c['verified'] and prior['verified'] and not prior['consistent_pair_gate_passed']
    assert prior['completion_sha256']==sha(SOURCE/'completion.json')
    rows=[];new_yaw=0
    for noise in ('clean','noisy'):
        for law in ('B0','B1-route'):
            reference=p['reuse_references'][noise+'/'+law]
            oldfile=ROOT/reference['path'];assert sha(oldfile)==reference['sha256']
            old={r['seed']:r for r in json.loads(oldfile.read_text())['runs']}
            variants={}
            for arm in ('floor_only','consistent_pair'):
                entry=next(e for e in c['records'] if (e['arm'],e['noise'],e['classical'])==(arm,noise,law))
                path=SOURCE/entry['path'];assert sha(path)==entry['sha256']
                variants[arm]=(path.parent,{r['seed']:r for r in json.loads(path.read_text())['runs']})
            for seed in (6301033,6301035):
                base=old[seed];floor=variants['floor_only'][1][seed];pair=variants['consistent_pair'][1][seed]
                assert base['scenario']==floor['scenario']==pair['scenario'] and pair['target_leg_m']==.38
                assert base['physical_safety_passed'] and floor['physical_safety_passed'] and pair['physical_safety_passed']
                assert base['design_joint_passed'] and floor['design_joint_passed'] and pair['design_joint_passed']
                added=base['peak_deg'][2]<=5 and pair['peak_deg'][2]>5;new_yaw+=added
                detail=dict(noise=noise,classical=law,seed=seed,original_peak_deg=base['peak_deg'],
                    floor_peak_deg=floor['peak_deg'],pair_peak_deg=pair['peak_deg'],
                    new_yaw_violation=bool(added),roll_peak_reduced=pair['peak_deg'][0]<base['peak_deg'][0],
                    original_success=base['success'],pair_success=pair['success'],
                    floor_witness=witness(*variants['floor_only'][:1],floor),
                    pair_witness=witness(*variants['consistent_pair'][:1],pair))
                rows.append(detail)
    assert len(rows)==8 and new_yaw==7 and all(r['roll_peak_reduced'] for r in rows)
    atomic_json(OUT/'round288_roll_yaw_tradeoff.json',dict(
        verified=True,round=288,auditor_sha256=sha(__file__),prior_review_sha256=sha(SOURCE/'round183_pair_review.json'),
        comparison_records=8,unique_cases=2,added_yaw_records=7,records=rows,new_simulation=0,new_training_samples=0,
        diagnosis='Existingreferenceprojection successfully lowersupperrollpeak,butcreatesyawviolations in7/8conditions. Fixedsingle-axis heightallocation doesnotpreserve coupled heading task;closedbranch staysclosed.',
        limits='Same20developmentstudy,two cases/fixed artificial-noise realization,not8independentseeds. Reference variant andguard differ fromoriginal;floor_only comparator isolatesreferenceintent better. Mean/yaw chronology andcontact snapshot do notprovea unique dynamicalcause orfullfriction authority.',
        next='289 derivejoint feasiblemean/difference reference andspeed/load/heading constraints usingpublicsignals andaudit novelty;nocontrollerpatch/newgain/numericalthreshold/trajectory/PPO until290 deepreview accepts distincttestable mechanism.'))
    print('PASS2888priorpairedcounterexamples;rollloweredall8/newyaw7;0newruns',flush=True)


if __name__=='__main__':
    run()
