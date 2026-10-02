"""Reconstruct public selection and every causal request; retain failed validation."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PRIOR=OUT.parent/'twentythird_round_virtual_reference_20261002'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    original=json.loads((OUT/'selector.json').read_text());assert not original['accepted']
    calibration=json.loads((OUT/'calibration_refined.json').read_text())
    pair=[json.loads((OUT/m/'verification.json').read_text()) for m in ('refined_zero','refined_virtual')]
    assert all(all(e['success'] for e in r['episodes']) for r in pair)
    selector=json.loads((OUT/'selector_refined.json').read_text());v2=json.loads((OUT/'selector_v2.json').read_text())
    assert selector['calibration_overlap']==[min(calibration['heights']),max(calibration['heights'])]
    assert selector['switch_height']==sum(selector['calibration_overlap'])/2==v2['switch_height']
    assert v2['reference_min_speed']==.75 and not v2['parameter_truth_used']
    templates=np.load(PRIOR/'template/trajectory.npz',allow_pickle=False);refs=[templates[f'features{d}'] for d in range(2)]
    plans=[np.load(OUT.parent/f'eighteenth_round_robust_reference_20261002/solve_world{d}/best.npz',allow_pickle=False)['schedule'] for d in range(2)]
    summaries={};inputs=[]
    modes=('overlap_zero','overlap_virtual','refined_zero','refined_virtual','boundary','broad','registered','speed_boundary_v2','broad_v2','registered_v2','random_v2','nearcut_v2')
    for mode in modes:
        folder=OUT/mode;r=json.loads((folder/'verification.json').read_text());reg=json.loads((folder/'registered_cases.json').read_text())
        data=np.load(folder/'trajectory.npz',allow_pickle=False);records=data['records'];requests=data['requests'];selected=data['use_reference']
        assert sha(folder/'trajectory.npz')==r['trajectory_sha256']
        n=len(r['episodes']);expected=[]
        for scene in reg['scenarios']:
            if reg['selector'] is None:expected.append(mode in ('overlap_virtual','refined_virtual'))
            else:
                rule=reg['selector'];assert rule==(v2 if mode.endswith('_v2') else selector)
                expected.append(scene['stand_height_m']<rule['switch_height'] and abs(scene['speed'])>rule.get('reference_min_speed',0.))
        np.testing.assert_array_equal(selected,expected);assert selected.tolist()==r['controller_selection']
        metric=np.load(folder/'metric.npz',allow_pickle=False)['metric'];current=np.zeros((n,6));phase=np.zeros(n,int);cursor=np.full(n,-1);finished=np.zeros(n,bool)
        assert abs(requests).max()<=1
        assert abs(np.diff(np.concatenate([np.zeros_like(requests[:1]),requests]),axis=0)).sum(axis=2).max()<=.1+1e-12
        for frame,u in zip(records,requests):
            target=current.copy()
            for w in np.flatnonzero(frame[:,2]>cursor):
                x=frame[w,6:];finished[w] |= x[6]>0;assert bool(frame[w,4])==finished[w]
                if not selected[w] or finished[w]:target[w]=0
                else:
                    d=int(reg['scenarios'][w]['speed']<0);reference=refs[d]
                    stop=np.flatnonzero(reference[:,6]>0);last=int(stop[0])-1 if len(stop) else len(reference)-1
                    choices=np.arange(phase[w],min(phase[w]+2,last)+1)
                    if not len(choices):choices=np.array([phase[w]])
                    query=x[:6].copy();query[3]*=reference[0,3]/frame[w,5]
                    error=reference[choices,:6]-query;cost=np.einsum('ni,ij,nj->n',error,metric,error)
                    phase[w]=int(choices[np.argmin(cost)])
                    refJ=reference[phase[w],7:15].reshape(2,2,2)
                    virtual=np.linalg.solve(refJ,plans[d][phase[w],:4].reshape(2,2,1))[:,:,0].mean(axis=0)
                    target[w,:4]=(x[7:15].reshape(2,2,2)@virtual).reshape(4);target[w,4:]=plans[d][phase[w],4:].mean()
                    target[w]/=max(1.,float(abs(target[w]).max()))
                assert phase[w]==int(frame[w,3])
            delta=target-current;current=np.clip(current+delta*np.minimum(1.,.1/np.maximum(abs(delta).sum(axis=1),1e-30))[:,None],-1,1)
            np.testing.assert_allclose(current,u,rtol=0,atol=1e-13);cursor=frame[:,2].copy()
        for e in r['episodes']:
            assert not e['success'] or e['physical_steps']==e['physical_evidence_steps']
            assert e['design_joint_passed']==(e['min_active_design_margin_rad']>=0 and e['physical_steps']==e['physical_evidence_steps'])
        for name,h in r['source_sha256'].items():
            paths=[ROOT/name]
            if name==str((OUT/'experiment.py').relative_to(ROOT)):
                paths += [OUT/'experiment_initial_calibration.py',OUT/'experiment_height_selector.py',OUT/'experiment_selector_v2_at_run.py']
            assert any(p.is_file() and sha(p)==h for p in paths),name
        summaries[mode]=dict(total=n,physical=r['physical'],design=r['design'],success=r['success'],reference_worlds=int(selected.sum()))
        inputs += [folder/'verification.json',folder/'registered_cases.json']
    assert summaries['speed_boundary_v2']['success']==144 and summaries['broad_v2']['success']==84
    assert summaries['registered_v2']['success']==15
    assert summaries['random_v2']['physical']==summaries['random_v2']['design']==64
    assert summaries['random_v2']['success']==62 and summaries['nearcut_v2']['success']==64
    result=dict(summaries=summaries,frozen_selector=v2,public_selection_and_all_requests_reconstructed=True,
        learning=False,production_promoted=False,full_admission=False,
        limitations='Finite public development coverage, zero Actor. Initial selector pressure regression remains retained. Two new terrain yaw failures remain failures; no continuous guarantee or learning advantage claim. Native implementation and real short PPO update/restore still pending.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target=OUT/'verification.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS frozen public selector and causal requests: boundary144/144, broad84/84, registered15/18, fresh126/128; no full admission')


if __name__=='__main__':run()
