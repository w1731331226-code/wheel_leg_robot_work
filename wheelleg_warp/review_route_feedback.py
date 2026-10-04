"""Independent command/trajectory/task reconstruction for the fixed route law."""
from pathlib import Path
import argparse,json,math
import numpy as np
from review_yaw_sector import sha,success,ROOT


def run(out):
    reg=json.loads((out/'registration.json').read_text());result=json.loads((out/'result.json').read_text());ledger=json.loads((out/'completed_jobs.json').read_text())
    assert reg['budget_episodes']==result['completed_episodes']==ledger['completed_episodes']==216
    assert reg['training_steps']==result['training_steps']==0 and reg['development_only'] and not reg['old_gate_or_final_used']
    assert result['registration_sha256']==sha(out/'registration.json') and all(sha(ROOT/n)==v for n,v in reg['source_sha256'].items())
    candidate=reg['candidate'];assert candidate==dict(name='B1_3',kp=.4,kd=.3,roll_gain=0)
    tau=(2+candidate['kd'])/(.4+candidate['kp']);assert tau==reg['tau_s']
    jobs=[panel+'_'+label for panel in reg['sets'] for label in reg['arms']]
    assert [r['job'] for r in ledger['records']]==jobs
    decoded={};stats={};lost={};height_groups={}
    for record in ledger['records']:
        job=record['job'];panel,label=job.split('_');arm=reg['arms'][label]
        rp=out/(job+'.json');tp=out/(job+'_trace.npz')
        assert sha(rp)==record['result_sha256'] and sha(tp)==record['trace_sha256']
        d=json.loads(rp.read_text());rows=d['runs'];cases=reg['sets'][panel]
        assert len(rows)==record['episodes']==len(cases)
        z=np.load(tp,allow_pickle=False);assert z['columns'].tolist()==reg['columns'] and len(z['offsets'])==len(rows)+1
        assert z['offsets'][0]==0 and z['offsets'][-1]==len(z['trace']) and np.all(np.diff(z['offsets'])>0) and np.isfinite(z['trace']).all()
        maxima=[];scores=[]
        for i,(r,c) in enumerate(zip(rows,cases)):
            assert r['seed']==c['seed'] and r['scenario']==c['scenario'] and r['success']==success(r)
            assert r['physical_evidence_steps']==r['physical_steps']>0
            physical=min(r['min_actual_A_leg_m'],r['min_actual_B_leg_m'])>=r['geometric_limit_m'] and r['min_eight_joint_margin_rad']>=0 and max(r['max_actual_torque_excess_Nm'],r['max_command_torque_excess_Nm'])<=1e-6
            assert r['physical_safety_passed']==physical and r['design_joint_passed']==(r['min_active_design_margin_rad']>=0)
            t=z['trace'][z['offsets'][i]:z['offsets'][i+1]]
            assert np.all(t[:-1,8]==0) and t[-1,8]==1 and np.all(np.diff(t[:,0])>0)
            assert math.isclose(t[-1,0],r['duration_s'],abs_tol=1e-10)
            ref=np.zeros(len(t));moving=abs(t[:,3])>.05
            ref[moving]=-arm*np.sign(t[moving,3])*np.arctan2(t[moving,4],abs(t[moving,3])*tau)
            ref=np.clip(ref,-math.radians(3),math.radians(3))
            np.testing.assert_allclose(ref,t[:,5],rtol=0,atol=1e-12)
            wheel=np.clip((-(candidate['kp']*t[:,1]+candidate['kd']*t[:,2])+(.4+candidate['kp'])*ref)/.3,-1,1).astype(np.float32)
            np.testing.assert_array_equal(wheel,t[:,6].astype(np.float32));assert np.max(abs(t[:,6]))<=1
            maxima.append(float(abs(t[:,7]).max()))
            if r['reason']=='completed':
                duration=r['physical_steps']*.0005;assert math.isclose(duration,r['duration_s'],abs_tol=1e-8) and duration>=r['arrival_s']+2-1e-8
                scores.append(r['rms_deg'][2]*math.sqrt(duration/(3.5+1.5*r['task_goal_progress_m']/abs(r['scenario']['speed']))))
            else:scores.append(None)
        assert d['summary']['total']==len(rows) and d['summary']['success_count']==sum(r['success'] for r in rows)
        complete=all(s is not None for s in scores);assert d['summary']['complete']==complete
        if complete:assert math.isclose(sum(scores)/len(scores),d['summary']['mean_yaw_score_deg'],abs_tol=1e-12)
        else:assert d['summary']['mean_yaw_score_deg'] is None
        assert d['physical']==sum(r['physical_safety_passed'] for r in rows) and d['design']==sum(r['design_joint_passed'] for r in rows)
        assert math.isclose(float(np.mean(maxima)),d['mean_case_max_abs_y_m'],abs_tol=1e-12)
        assert result['results'][job]=={k:v for k,v in d.items() if k!='runs'}
        decoded[job]=rows;stats[job]={k:v for k,v in d.items() if k!='runs'}
        if panel=='controlled':
            height_groups[label]={str(h):dict(success=sum(r['success'] for r in rows if r['scenario']['stand_height_m']==h),total=sum(r['scenario']['stand_height_m']==h for r in rows)) for h in [.115,.16,.24,.3,.38]}
    preservation={}
    for panel in reg['sets']:
        b,c=[decoded[panel+'_'+name] for name in ['B1','correct']]
        preservation[panel]=all((not rb['success'] or rc['success']) and (rb['reason']!='completed' or rc['reason']=='completed') and (not rb['physical_safety_passed'] or rc['physical_safety_passed']) and (not rb['design_joint_passed'] or rc['design_joint_passed']) for rb,rc in zip(b,c))
        lost[panel]=[dict(seed=rc['seed'],base_peak_deg=rb['peak_deg'],correct_peak_deg=rc['peak_deg'],correct_reason=rc['reason']) for rb,rc in zip(b,c) if rb['success'] and not rc['success']]
    b,c,r=[stats['controlled_'+label] for label in ['B1','correct','reverse']]
    cb,cc=[stats['regular_'+label]['summary'] for label in ['B1','correct']]
    gate=all(preservation.values()) and c['summary']['success_count']>b['summary']['success_count'] and c['mean_case_max_abs_y_m']<min(b['mean_case_max_abs_y_m'],r['mean_case_max_abs_y_m']) and cc['complete'] and cb['complete'] and cc['mean_yaw_score_deg']<=cb['mean_yaw_score_deg']+.05
    assert preservation==result['preservation'] and gate==result['fixed_law_continue_gate']
    review=dict(verified=True,episodes=216,jobs=6,unique_development_cases=72,fixed_law_continue_gate=gate,preservation=preservation,lost_B1_successes=lost,
        controlled_height_groups=height_groups,results=stats,result_sha256=sha(out/'result.json'),registration_sha256=sha(out/'registration.json'),reviewer_sha256=sha(__file__),
        scope='Full command equation replay, finite trace/y metrics, case identity, actual physical/design metrics, original success/yaw summary and preregistered gate; not a safety proof or learning advantage.')
    (out/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in review.items() if k!='results'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
