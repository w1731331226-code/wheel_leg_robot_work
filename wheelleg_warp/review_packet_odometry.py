"""Independent offline reconstruction of packet-only integration and task rows."""
from pathlib import Path
import argparse, json, math
import numpy as np
from review_yaw_sector import sha, success, ROOT


def run(out):
    reg=json.loads((out/'registration.json').read_text());result=json.loads((out/'result.json').read_text())
    ledger=json.loads((out/'completed_jobs.json').read_text())
    assert reg['budget_episodes']==result['completed_episodes']==ledger['completed_episodes']==128
    assert reg['training_steps']==result['training_steps']==0 and not reg['old_gate_or_final_used']
    assert result['registration_sha256']==sha(out/'registration.json')
    assert all(sha(ROOT/n)==v for n,v in reg['source_sha256'].items())
    assert [r['label'] for r in ledger['records']]==['B1','1609','1610','1611']
    for model in reg['models']:
        p=Path(model['prefix']);r=model['checkpoint']
        assert sha(p.with_suffix('.zip'))==r['checkpoint_sha256'] and sha(p.with_suffix('.pkl'))==r['normalization_sha256']
    metrics={}
    for record in ledger['records']:
        label=record['label'];rp=out/(label+'.json');tp=out/(label+'_trace.npz')
        assert sha(rp)==record['result_sha256'] and sha(tp)==record['trace_sha256']
        data=json.loads(rp.read_text());z=np.load(tp,allow_pickle=False)
        assert z['columns'].tolist()==reg['columns'] and len(z['offsets'])==33
        assert z['offsets'][0]==0 and z['offsets'][-1]==len(z['trace']) and np.all(np.diff(z['offsets'])>1)
        assert len(data['runs'])==32 and np.isfinite(z['trace']).all()
        errors=[];scores=[]
        for i,(row,case) in enumerate(zip(data['runs'],reg['cases'])):
            assert row['seed']==case['seed'] and row['scenario']==case['scenario']
            assert row['success']==success(row) and row['physical_evidence_steps']==row['physical_steps']>0
            t=z['trace'][z['offsets'][i]:z['offsets'][i+1]]
            assert t[0,0]==0 and t[0,1]==t[0,2]==0
            assert np.all(t[:-1,6]==0) and t[-1,6]==1 and np.all(np.diff(t[:,0])>0)
            assert math.isclose(t[-1,0],row['duration_s'],abs_tol=1e-10)
            np.testing.assert_allclose(np.diff(t[:-1,0]),.02,rtol=0,atol=1e-10)
            # Native B1 packets are float32; VecNormalize inversion is float64.
            # The trace container is float64, so reconstruct input arithmetic
            # instead of silently replacing its sin/add/multiply precision.
            packet=t[:,3:6].astype(np.float32 if label=='B1' else np.float64)
            velocity=np.sin(packet[:,0])*packet[:,1]+np.cos(packet[:,0])*packet[:,2]
            increment=.5*(velocity[1:]+velocity[:-1])*np.diff(t[:,0])
            estimate=np.r_[0,np.cumsum(increment)]
            np.testing.assert_allclose(estimate,t[:,1],rtol=0,atol=1e-12)
            errors.append(float(abs(t[:-1,1]-t[:-1,2]).max()))
            if row['reason']=='completed':
                duration=row['physical_steps']*.0005
                assert math.isclose(duration,row['duration_s'],abs_tol=1e-8) and duration>=row['arrival_s']+2-1e-8
                scores.append(row['rms_deg'][2]*math.sqrt(duration/(3.5+1.5*row['task_goal_progress_m']/abs(row['scenario']['speed']))))
            else:scores.append(None)
        np.testing.assert_allclose(errors,data['per_case_max_error_m'],rtol=0,atol=1e-12)
        assert math.isclose(max(errors),data['max_nonterminal_y_error_m'],abs_tol=1e-12)
        assert data['summary']['total']==32 and data['summary']['success_count']==sum(r['success'] for r in data['runs'])
        complete=all(s is not None for s in scores);assert data['summary']['complete']==complete
        if complete:assert math.isclose(sum(scores)/32,data['summary']['mean_yaw_score_deg'],abs_tol=1e-12)
        else:assert data['summary']['mean_yaw_score_deg'] is None
        assert result['results'][label]=={k:v for k,v in data.items() if k!='runs'}
        metrics[label]=dict(max_nonterminal_error_m=max(errors),mean_case_max_error_m=float(np.mean(errors)),
            samples=len(z['trace']),success=data['summary']['success_count'])
    passed=all(m['max_nonterminal_error_m']<=reg['accuracy_budget_m'] for m in metrics.values())
    assert passed==result['accuracy_budget_passed']
    review=dict(verified=True,episodes=128,unique_development_cases=32,metrics=metrics,accuracy_budget_passed=passed,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),'wheelleg_warp/review_yaw_sector.py':sha(ROOT/'wheelleg_warp/review_yaw_sector.py')},
        result_sha256=sha(out/'result.json'),registration_sha256=sha(out/'registration.json'),
        scope='Reconstructed packet frame rotation/integration, timestamps, known origin, source/models/hash/identity, original task success and yaw score. No control intervention, new learning or sim-to-real assertion.')
    (out/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
    print(json.dumps(review,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
