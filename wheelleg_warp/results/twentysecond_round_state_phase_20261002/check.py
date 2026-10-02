"""Replay each causal phase decision and original request-domain check offline."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    templates=np.load(OUT/'template/trajectory.npz',allow_pickle=False)
    refs=[templates[f'features{d}'] for d in range(2)]
    plans=[np.load(OUT.parent/f'eighteenth_round_robust_reference_20261002/solve_world{d}/best.npz',allow_pickle=False)['schedule'] for d in range(2)]
    results={};passed={}
    for mode in ('template','state_phase','state_phase_full','clock_full'):
        folder=OUT/mode;r=json.loads((folder/'verification.json').read_text());registration=json.loads((folder/'registered_cases.json').read_text())
        z=np.load(folder/'trajectory.npz',allow_pickle=False);records=z['records'];requests=z['requests'];n=len(r['episodes'])
        assert sha(folder/'trajectory.npz')==r['trajectory_sha256']
        assert np.isfinite(records).all() and np.isfinite(requests).all() and abs(requests).max()<=1
        assert abs(np.diff(np.concatenate([np.zeros_like(requests[:1]),requests]),axis=0)).sum(axis=2).max()<=.1+1e-12
        for e in r['episodes']:
            assert e['physical_steps']==e['physical_evidence_steps']
            assert e['design_joint_passed']==(e['min_active_design_margin_rad']>=0)
        if mode!='template':
            m=np.load(folder/'metric.npz',allow_pickle=False)
            np.testing.assert_allclose(m['metric'],m['lift'].T@m['P']@m['lift'],rtol=0,atol=1e-10)
            metric=m['metric'];assert np.linalg.eigvalsh(metric).min()>0
        current=np.zeros((n,6));previous_cursor=np.full(n,-1);phase=np.zeros(n,int);finished=np.zeros(n,bool)
        changed=0;matched=0
        for frame,u in zip(records,requests):
            target=current.copy()
            for w in np.flatnonzero(frame[:,2]>previous_cursor):
                row=frame[w];scene=registration['scenarios'][w];direction=int(scene['speed']<0);cursor=int(row[2]);x=row[6:13]
                if cursor==0:assert row[5]==x[3]
                if mode=='template':
                    target[w]=plans[direction][min(cursor,399)]
                    np.testing.assert_array_equal(x,refs[w][cursor])
                else:
                    finished[w] |= x[6]>0
                    assert bool(row[4])==bool(finished[w])
                    weight=1. if mode in ('state_phase_full','clock_full') else float(np.clip((abs(scene['speed'])-.75)/.25,0,1)*np.clip((.12-scene['stand_height_m'])/(.12-.115),0,1))
                    if finished[w] or weight==0:target[w]=0
                    else:
                        if mode=='clock_full':phase[w]=min(cursor,399)
                        else:
                            reference=refs[direction];end=np.flatnonzero(reference[:,6]>0);last=int(end[0])-1 if len(end) else len(reference)-1
                            choices=np.arange(phase[w],min(phase[w]+2,last)+1)
                            if not len(choices):choices=np.array([phase[w]])
                            query=x[:6].copy();query[3]*=reference[0,3]/row[5]
                            error=reference[choices,:6]-query;cost=np.einsum('ni,ij,nj->n',error,metric,error)
                            phase[w]=int(choices[np.argmin(cost)])
                        target[w]=weight*plans[direction][phase[w]]
                        matched+=1;changed+=phase[w]!=min(cursor,399)
                    assert phase[w]==int(row[3])
            delta=target-current;current=np.clip(current+delta*np.minimum(1.,.1/np.maximum(abs(delta).sum(axis=1),1e-30))[:,None],-1,1)
            np.testing.assert_allclose(current,u,rtol=0,atol=1e-13)
            previous_cursor=frame[:,2].copy()
        for name,h in r['source_sha256'].items():
            paths=[ROOT/name]
            if Path(name).name=='experiment.py':paths += [OUT/'experiment_at_weighted_phase.py',OUT/'experiment_at_full_phase.py']
            assert any(p.is_file() and sha(p)==h for p in paths),name
        passed[mode]={i for i,e in enumerate(r['episodes']) if e['success']}
        results[mode]=dict(total=r['total'],physical=r['physical'],design=r['design'],success=r['success'],
            phase_decisions_checked=int(matched),phase_differs_from_clock=int(changed),
            physical_evidence_steps=sum(e['physical_steps'] for e in r['episodes']))
    gain=passed['state_phase_full']-passed['clock_full'];loss=passed['clock_full']-passed['state_phase_full']
    assert len(gain)==19 and not loss
    result=dict(results=results,full_state_vs_full_clock=dict(gained_cases=sorted(gain),lost_cases=sorted(loss)),
        full_amplitude_vs_scaled=dict(gained_cases=sorted(passed['state_phase_full']-passed['state_phase']),lost_cases=sorted(passed['state_phase']-passed['state_phase_full'])),
        all_original_request_domains_checked=True,all_causal_decisions_reconstructed=True,
        production_promoted=False,learning=False,full_admission=False,
        conclusion='State phase matters, but neither phase matching nor removing amplitude interpolation meets full design gates; investigate geometry-consistent virtual effort transfer, not more scalar weight/phase tuning.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target=OUT/'verification.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS all causal choices/domain/evidence. Scaled state86/100, full state89/100, full clock70/100; no admission.')


if __name__=='__main__':run()
