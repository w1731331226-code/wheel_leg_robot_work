"""Reconstruct geometry transfer and all phase/request decisions from known states."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    saved=np.load(OUT/'template/trajectory.npz',allow_pickle=False)
    refs=[saved[f'features{d}'] for d in range(2)]
    plans=[np.load(OUT.parent/f'eighteenth_round_robust_reference_20261002/solve_world{d}/best.npz',allow_pickle=False)['schedule'] for d in range(2)]
    roundtrip=0.;symmetry=0.
    for d,reference in enumerate(refs):
        J=reference[:,7:15].reshape(-1,2,2,2);u=plans[d][:len(reference),:4].reshape(-1,2,2,1)
        force=np.linalg.solve(J,u)
        roundtrip=max(roundtrip,float(abs(J@force-u).max()))
        symmetry=max(symmetry,float(abs(J@force.mean(axis=1)[:,None]-u).max()))
    assert roundtrip<1e-12 and symmetry<1e-5
    results={};sets={}
    for mode in ('motor','virtual_current','virtual_2khz','zero'):
        folder=OUT/mode;r=json.loads((folder/'verification.json').read_text());reg=json.loads((folder/'registered_cases.json').read_text())
        data=np.load(folder/'trajectory.npz',allow_pickle=False);records=data['records'];u=data['requests'];m=np.load(folder/'metric.npz');metric=m['metric']
        physical_requests=data['physical_requests'] if mode=='virtual_2khz' else None
        assert sha(folder/'trajectory.npz')==r['trajectory_sha256']
        np.testing.assert_allclose(metric,m['lift'].T@m['P']@m['lift'],rtol=0,atol=1e-10)
        assert np.linalg.eigvalsh(metric).min()>0
        n=len(r['episodes']);previous=np.zeros((n,6));cursor=np.full(n,-1);phase=np.zeros(n,int);finished=np.zeros(n,bool)
        max_tv=0.;mapping_error=0.
        for i,(frame,request) in enumerate(zip(records,u)):
            target=previous.copy()
            for w in np.flatnonzero(frame[:,2]>cursor):
                x=frame[w,6:];d=int(reg['scenarios'][w]['speed']<0);finished[w] |= x[6]>0
                assert finished[w]==bool(frame[w,4])
                if mode=='zero' or finished[w]:target[w]=0
                else:
                    ref=refs[d];park=np.flatnonzero(ref[:,6]>0);last=int(park[0])-1 if len(park) else len(ref)-1
                    candidates=np.arange(phase[w],min(phase[w]+2,last)+1)
                    if not len(candidates):candidates=np.array([phase[w]])
                    query=x[:6].copy();query[3]*=ref[0,3]/frame[w,5]
                    error=ref[candidates,:6]-query;cost=np.einsum('ni,ij,nj->n',error,metric,error)
                    phase[w]=int(candidates[np.argmin(cost)])
                    target[w]=plans[d][phase[w]]
                    if mode!='motor':
                        reference_J=ref[phase[w],7:15].reshape(2,2,2)
                        common=np.linalg.solve(reference_J,target[w,:4].reshape(2,2,1))[:,:,0].mean(axis=0)
                        current_J=x[7:15].reshape(2,2,2)
                        target[w,:4]=(current_J@common).reshape(4);target[w,4:]=target[w,4:].mean()
                        back=np.linalg.solve(current_J,target[w,:4].reshape(2,2,1))[:,:,0]
                        mapping_error=max(mapping_error,float(abs(back-common).max()))
                        target[w]/=max(1.,float(abs(target[w]).max()))
                assert phase[w]==int(frame[w,3])
            if mode=='virtual_2khz':
                physical=physical_requests[10*i:10*i+10]
                change=np.diff(np.concatenate([previous[None],physical]),axis=0)
                tv=abs(change).sum(axis=(0,2));max_tv=max(max_tv,float(tv.max()))
                assert abs(physical).max()<=1 and tv.max()<=.1+1e-12
                np.testing.assert_array_equal(request,physical[-1])
            else:
                delta=target-previous;expected=np.clip(previous+delta*np.minimum(1.,.1/np.maximum(abs(delta).sum(axis=1),1e-30))[:,None],-1,1)
                np.testing.assert_allclose(expected,request,rtol=0,atol=1e-13)
                max_tv=max(max_tv,float(abs(request-previous).sum(axis=1).max()))
                assert abs(request).max()<=1 and max_tv<=.1+1e-12
            previous=request;cursor=frame[:,2].copy()
        for e in r['episodes']:
            assert e['physical_steps']==e['physical_evidence_steps']
            assert e['design_joint_passed']==(e['min_active_design_margin_rad']>=0)
        for name,h in r['source_sha256'].items():
            choices=[ROOT/name]
            if Path(name).name=='experiment.py':choices += [OUT/'experiment_at_5ms.py',OUT/'experiment_at_2khz.py']
            assert any(p.is_file() and sha(p)==h for p in choices),name
        sets[mode]={w for w,e in enumerate(r['episodes']) if e['success']}
        results[mode]=dict(total=r['total'],physical=r['physical'],design=r['design'],success=r['success'],
            max_request_total_variation_per5ms=max_tv,current_J_virtual_target_error=mapping_error)
    assert len(sets['virtual_current']-sets['motor'])==8 and not sets['motor']-sets['virtual_current']
    assert sets['virtual_current']==sets['virtual_2khz']
    assert sets['zero']|sets['virtual_current']==set(range(100))
    scenes=json.loads((OUT/'zero/registered_cases.json').read_text())['scenarios']
    coverage=[]
    for i in range(50):
        assert scenes[i]['stand_height_m']==scenes[i+50]['stand_height_m'] and scenes[i]['speed']==scenes[i+50]['speed']
        zero=i in sets['zero'] and i+50 in sets['zero']
        virtual=i in sets['virtual_current'] and i+50 in sets['virtual_current']
        assert zero or virtual
        coverage.append(dict(height=scenes[i]['stand_height_m'],speed=scenes[i]['speed'],
            zero_passes_both_registered_parameters=zero,virtual_passes_both_registered_parameters=virtual))
    result=dict(results=results,individual_roundtrip_error_Nm=roundtrip,template_common_symmetrization_error_Nm=symmetry,
        recovered_by_geometry=sorted(sets['virtual_current']-sets['motor']),remaining_failures=sorted(set(range(100))-sets['virtual_current']),
        faster_remapping_additional_successes=0,production_promoted=False,learning=False,full_admission=False,
        public_operating_point_coverage=coverage,
        scope='Target virtual equality is checked before motor slew; finite-rate held requests are not claimed to preserve exact virtual equality at every physical state. Full task gates remain unchanged.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target=OUT/'verification.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS geometry/phase/request domains: motor89, virtual97, zero87/100. Public-point coverage overlaps; no selector or continuous-domain admission.')


if __name__=='__main__':run()
