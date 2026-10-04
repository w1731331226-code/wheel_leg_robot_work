"""Offline full-sample and first-crossing audit of the read-only boundary trace."""
from pathlib import Path
import argparse,json,math,sys
import numpy as np
from review_yaw_sector import sha,success,ROOT
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
from state_estimation import leg_kinematics


def run(out):
    reg=json.loads((out/'registration.json').read_text());complete=json.loads((out/'completion.json').read_text());ledger=json.loads((out/'completed_jobs.json').read_text())
    assert reg['budget_episodes']==complete['completed_episodes']==ledger['completed_episodes']==135
    assert reg['training_updates']==complete['training_updates']==0 and not reg['old_gate_or_final_used']
    assert complete['registration_sha256']==sha(out/'registration.json') and complete['records']==ledger['records']
    assert all(sha(ROOT/n)==v for n,v in reg['source_sha256'].items())
    assert [r['label'] for r in ledger['records']]==['B0','B1','1609','1610','1611']
    for m in reg['models']:
        p=Path(m['prefix']);assert sha(p.with_suffix('.zip'))==m['checkpoint']['checkpoint_sha256'] and sha(p.with_suffix('.pkl'))==m['checkpoint']['normalization_sha256']
    details={};summary={};samples=0
    for record in ledger['records']:
        label=record['label'];rp=out/(label+'.json');tp=out/(label+'_trace.npz')
        assert sha(rp)==record['result_sha256'] and sha(tp)==record['trace_sha256']
        d=json.loads(rp.read_text());z=np.load(tp,allow_pickle=False);rows=d['runs'];assert len(rows)==len(reg['cases'])==27
        assert z['columns'].tolist()==reg['columns'] and len(z['offsets'])==28 and z['offsets'][0]==0 and z['offsets'][-1]==len(z['trace'])
        assert np.all(np.diff(z['offsets'])>0) and np.isfinite(z['trace']).all();entries=[];scores=[]
        for i,(r,c) in enumerate(zip(rows,reg['cases'])):
            assert r['seed']==c['seed'] and r['scenario']==c['scenario'] and r['success']==success(r)
            t=z['trace'][z['offsets'][i]:z['offsets'][i+1]]
            assert len(t)==r['physical_steps']==r['physical_evidence_steps'] and np.all(t[:,0]==1)
            np.testing.assert_allclose(t[:,1],np.arange(1,len(t)+1)*.0005,rtol=0,atol=1e-9)
            margin=1.4-abs(t[:,10:14]).max(axis=1)
            np.testing.assert_allclose(margin,t[:,47],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.minimum.accumulate(margin),t[:,40],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(t[1:,2:6],t[:-1,10:14])
            np.testing.assert_array_equal(t[1:,6:10],t[:-1,14:18])
            np.testing.assert_allclose((t[:,18:24]+t[:,24:30]).astype(np.float32),t[:,30:36].astype(np.float32),rtol=0,atol=1e-6)
            assert np.all((t[:,36]>=0)&(t[:,36]<=1))
            assert np.isclose(t[:,40].min(),r['min_active_design_margin_rad'],rtol=0,atol=1e-12)
            assert r['design_joint_passed']==bool(margin.min()>=0)
            assert math.isclose(t[-1,1],r['duration_s'],abs_tol=1e-8)
            assert math.isclose(t[-1,38],min(r['min_actual_A_leg_m'],r['min_actual_B_leg_m']),abs_tol=1e-12)
            assert math.isclose(t[-1,39],r['min_eight_joint_margin_rad'],abs_tol=1e-12)
            if label not in ('B0','B1'):
                assert np.all(t[:,28:30]==0) and np.all(t[:,43]==0) and np.all(t[:,46]==0)
            if r['reason']=='completed':
                assert r['arrival_s'] is not None and r['duration_s']>=r['arrival_s']+2-1e-8
                scores.append(r['rms_deg'][2]*math.sqrt(r['duration_s']/(3.5+1.5*r['task_goal_progress_m']/abs(r['scenario']['speed']))))
            else:scores.append(None)
            crossing=np.flatnonzero(margin<0);entry=dict(case=r['seed'],height_m=r['scenario']['stand_height_m'],success=r['success'],physical=r['physical_safety_passed'],design=r['design_joint_passed'],minimum_margin_rad=float(margin.min()))
            if len(crossing):
                k=int(crossing[0]);joint=int(abs(t[k,10:14]).argmax());sign=float(np.sign(t[k,10+joint]))
                first_contact=np.flatnonzero(t[:,37].astype(int)>0)
                entry['first_crossing']=dict(time_s=float(t[k,1]),joint=joint,pre_q_rad=float(t[k,2+joint]),post_q_rad=float(t[k,10+joint]),
                    pre_velocity_rad_s=float(t[k,6+joint]),post_velocity_rad_s=float(t[k,14+joint]),
                    nominal_command_Nm=float(t[k,18+joint]),actor_command_Nm=float(t[k,24+joint]),total_command_Nm=float(t[k,30+joint]),
                    actual_torque_Nm=float(t[k,48+joint]),lambda_value=float(t[k,36]),contact_mask=int(t[k,37]),
                    filtered_request=t[k,41:44].tolist(),target_request=t[k,44:47].tolist(),
                    nominal_direction_relative_to_joint=sign*float(t[k,18+joint]),actor_direction_relative_to_joint=sign*float(t[k,24+joint]),
                    velocity_toward_boundary_before_step=sign*float(t[k,6+joint]),
                    first_recorded_target_contact_s=float(t[first_contact[0],1]) if len(first_contact) else None)
                mapped=[];target_mapped=[]
                for side in range(2):
                    jac=leg_kinematics(t[k,2+2*side:4+2*side],np.zeros(2))[3]
                    sign_side=1 if side==0 else -1
                    mapped.extend((jac@np.array([t[k,41]*3.4335,t[k,42]])*sign_side*t[k,36]).tolist())
                    target_mapped.extend((jac@np.array([t[k,44]*3.4335,t[k,45]])*sign_side).tolist())
                np.testing.assert_allclose(mapped,t[k,24:28],rtol=0,atol=1e-9)
                entry['first_crossing']['unsmoothed_target_joint_direction']=sign*float(target_mapped[joint])
                entry['first_crossing']['target_inward_but_executed_actor_outward']=bool(sign*target_mapped[joint]<0 and sign*t[k,24+joint]>0)
            entries.append(entry)
        assert d['summary']['success_count']==sum(r['success'] for r in rows) and d['summary']['total']==27
        all_complete=all(x is not None for x in scores);assert d['summary']['complete']==all_complete
        if all_complete:assert math.isclose(sum(scores)/27,d['summary']['mean_yaw_score_deg'],abs_tol=1e-12)
        else:assert d['summary']['mean_yaw_score_deg'] is None
        assert d['physical']==sum(r['physical_safety_passed'] for r in rows) and d['design']==sum(r['design_joint_passed'] for r in rows)
        assert record['stats']['physics_samples']==len(z['trace']);samples+=len(z['trace']);details[label]=entries
        crossings=[e['first_crossing'] for e in entries if 'first_crossing' in e]
        summary[label]=dict(success=d['summary']['success_count'],physical=d['physical'],design=d['design'],crossings=len(crossings),
            crossings_before_first_target_contact=sum(x['first_recorded_target_contact_s'] is None or x['time_s']<x['first_recorded_target_contact_s'] for x in crossings),
            earliest_crossing_s=min((x['time_s'] for x in crossings),default=None),
            actor_outward_sign_count=sum(x['actor_direction_relative_to_joint']>0 for x in crossings),
            nominal_inward_sign_count=sum(x['nominal_direction_relative_to_joint']<0 for x in crossings),
            inward_target_outward_execution_count=sum(x['target_inward_but_executed_actor_outward'] for x in crossings),
            already_moving_outward_count=sum(x['velocity_toward_boundary_before_step']>0 for x in crossings))
        print('VERIFIED',label,summary[label],flush=True)
    report=dict(verified=True,episodes=135,physics_samples=samples,summary=summary,first_crossing_cases=details,
        registration_sha256=sha(out/'registration.json'),completion_sha256=sha(out/'completion.json'),reviewer_sha256=sha(__file__),kinematic_helper_sha256=sha(ROOT/'wheelleg_ppo/tools/state_estimation.py'),training_updates=0,
        scope='Every recorded0.5ms sample/time/q-v continuity/control decomposition and design margin, original task rows and fixed model/source/hash verified. Contact masks track target bump/terrain, excluding ordinary floor support. Torque-direction counts are descriptive, not coupled-acceleration/contact causality; no safe-filter effectiveness claim.')
    (out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PASS135 episodes',samples,'physical samples, no learning')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
