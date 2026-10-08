"""Terminal328 task/raw/gate audit; no simulation, fitting or changed thresholds."""
import json
import numpy as np
import execution_history_evaluation as evaluation
from score_reference_learning import load_rows
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_task_pair_v1'
KINDS=('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace','actor_trace')


def run():
    assert not (OUT/'task_review.json').exists()
    p,a,c=[json.loads((OUT/n).read_text()) for n in ('proposal.json','source_admission.json','completion.json')]
    assert c['verified'] and c['evaluations']==328 and len(c['records'])==20
    assert c['admission_sha256']==sha(OUT/'source_admission.json')
    assert a['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/n)==h for n,h in a['source_sha256'].items())
    data={};raw_files=raw_bytes=0;reproduction=[]
    for record in c['records']:
        job=next(j for j in p['eval_jobs'] if (j['panel'],j['batch'])==(record['panel'],record['batch']))
        assert record['indices']==job['indices']
        path=OUT/record['path'];assert sha(path)==record['sha256']
        rows=load_rows(path,job['cases'])['runs']
        checked=evaluation.recorder.prior.prior.check_files(path.parent,job['cases'],{r['seed']:r for r in rows})
        assert checked['physics_steps']==record['physics_steps']
        assert evaluation.recorder.logs(path.parent,job['cases'])==checked['physics_steps']
        key=record['arm']+'/'+record['panel']
        data.setdefault(key,[]).extend(rows)
        for row in rows:
            for kind in KINDS:
                file=path.parent/row[kind]['path'];assert sha(file)==row[kind]['sha256']
                raw_files+=1;raw_bytes+=file.stat().st_size
            with np.load(path.parent/row['actor_trace']['path'],allow_pickle=False) as z:
                actor=z['trace']
                assert actor.shape==(int(np.ceil(row['physical_steps']/40)),968)
                assert np.isfinite(actor).all()
                np.testing.assert_array_equal(actor[:,:481],actor[:,481:962])
                np.testing.assert_array_equal(actor[:,962:],0.)
            if record['arm']=='map_B0':
                file=path.parent/row['map_trace']['path'];assert sha(file)==row['map_trace']['sha256']
                with np.load(file,allow_pickle=False) as z:
                    trace=z['trace'];assert trace.shape==(row['physical_steps'],9)
                    assert np.isfinite(trace).all()
                    np.testing.assert_array_equal(trace[:,1],np.arange(1,len(trace)+1))
                    np.testing.assert_array_equal(trace[:,3],row['scenario']['stand_height_m'])
                    np.testing.assert_array_equal(trace[:,8],(trace[:,2]!=0.).astype(float))
                    np.testing.assert_array_equal(trace[trace[:,2]==0.,4:],0.)
                raw_files+=1;raw_bytes+=file.stat().st_size
        if record['arm']=='old_B0':
            archived=next(r for r in p['old_archived_baseline'] if (r['panel'],r['batch'])==(record['panel'],record['batch']))
            assert sha(ROOT/archived['path'])==archived['sha256']
            prior=json.loads((ROOT/archived['path']).read_text())['runs']
            for old,new in zip(prior,rows):
                assert old['seed']==new['seed'] and old['scenario']==new['scenario']
                flags={k:[old[k],new[k]] for k in ('success','reason','physical_safety_passed',
                    'design_joint_passed','terrain_passed','terrain_exit_passed') if old[k]!=new[k]}
                numeric={k:abs(old[k]-new[k]) for k in ('velocity_rmse','height_rmse_m','stop_distance_m','tail_speed_m_s')}
                peaks=float(np.max(abs(np.array(old['peak_deg'])-np.array(new['peak_deg']))))
                reproduction.append(dict(seed=new['seed'],panel=record['panel'],label_differences=flags,
                    metric_abs_delta=numeric,peak_abs_delta_deg=peaks,
                    old_physical_steps=old['physical_steps'],new_physical_steps=new['physical_steps']))
        print('RAW AUDIT',record['arm'],record['panel'],record['batch'],'files',raw_files,flush=True)
    assert sum(len(x) for x in data.values())==328 and len(reproduction)==164
    counts={k:dict(total=len(rows),success=sum(r['success'] for r in rows),
        physical=sum(r['physical_safety_passed'] for r in rows),
        design=sum(r['design_joint_passed'] for r in rows),
        mean_J=evaluation.summary(rows)['mean_yaw_score_deg']) for k,rows in data.items()}
    lost={};gained={}
    for panel in ('regular','controlled','legacy'):
        old=data['old_B0/'+panel];new=data['map_B0/'+panel]
        assert [r['seed'] for r in old]==[r['seed'] for r in new]
        lost[panel]=[a['seed'] for a,b in zip(old,new) if a['success'] and not b['success']]
        gained[panel]=[a['seed'] for a,b in zip(old,new) if not a['success'] and b['success']]
    numeric_exact=all(not any(r['metric_abs_delta'].values()) and r['peak_abs_delta_deg']==0. and
                      r['old_physical_steps']==r['new_physical_steps'] for r in reproduction)
    labels_exact=all(not r['label_differences'] for r in reproduction)
    gates=dict(old_label_reproduction=labels_exact,old_numeric_reproduction=numeric_exact,
        regular_preserved=not lost['regular'] and counts['map_B0/regular']['success']==96,
        legacy_preserved=not lost['legacy'] and counts['map_B0/legacy']['success']==28,
        all_map_physical_design=all(counts['map_B0/'+p][k]==counts['map_B0/'+p]['total']
                                  for p in ('regular','controlled','legacy') for k in ('physical','design')),
        no_lost_oldcontrolled=not lost['controlled'],
        controlled_atleast34=counts['map_B0/controlled']['success']>=34,
        controlled_atleast2new=len(gained['controlled'])>=2)
    passed=all(gates.values())
    atomic_json(OUT/'task_review.json',dict(
        verified=True,round=286,reviewer_sha256=sha(__file__),completion_sha256=sha(OUT/'completion.json'),
        counts=counts,lost=lost,gained=gained,gates=gates,engineering_continue_gate_passed=passed,
        reproduction=reproduction,raw_files=raw_files,raw_bytes=raw_bytes,
        baseline_constructor_transitionFD_calls=c['baseline_constructor_transitionFD_calls'],
        actual_firstepisode_physics_steps=sum(r['physical_steps'] for rows in data.values() for r in rows),
        queue_wall_seconds=c['queue_wall_seconds'],additional_simulation=0,new_training_samples=0,
        full_CPU_GPU_admitted=False,production_admitted=False,formal_PPO_admitted=False,
        decision='If engineeringgatefailed close map taskbenefit;retain model-consistency/initializer evidence andall counterexamples. No refit/retry/morebudget/thresholdchange. Source/backend old numericdifferences remainunknowncause,do notsilently accept.',
        limits='Reused established complete/gyro/role/phase/parking geometry checker plus independent source/SHA/Actor/maptrace/pair/gates. Singlefixeddevelopment taskpair,not independenttraining seeds/OOD/statistical newmethodproof;PPO goal remainsopen.'))
    print('DONE286 full328/rawaudit;engineeringgate',passed,'counts',counts,flush=True)


if __name__=='__main__':
    run()
