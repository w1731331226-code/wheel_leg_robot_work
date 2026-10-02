"""Recalculate motion error timing, damping counterexample and full design gates."""
from pathlib import Path
import json
import re
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
from hashlib import sha256


def digest(p):return sha256(p.read_bytes()).hexdigest()


def run():
    design=ROOT/'wheelleg_warp/results/shared_height115_normal_20261002/design.npz'
    gains=np.load(design)['gains'];heights=np.array([.115,.16,.25,.30,.38])
    reports={};timing=[]
    for name in ('instrumented','candidate','production_panel'):
        folder=OUT/name;r=json.loads((folder/'verification.json').read_text());z=np.load(folder/'trace.npz');trace=z['trace'];requests=z['extra']
        valid=trace[:,:,4]>0;counts=valid.sum(axis=0)
        assert counts.tolist()==[e['physical_steps'] for e in r['episodes']]
        assert all(e['physical_steps']==e['physical_evidence_steps'] for e in r['episodes'])
        assert not trace[:,:,5][valid].any()
        assert abs(requests).max()<=1 and abs(np.diff(np.concatenate([np.zeros_like(requests[:1]),requests]),axis=0)).sum(axis=2).max()<=.1+1e-12
        margins=[float(np.min(1.4-abs(trace[valid[:,w],w,:4]))) for w in range(18)]
        np.testing.assert_array_equal(margins,r['active_design_margin_rad'])
        gates=[bool(e['success'] and e['physical_safety_passed'] and m>=0) for e,m in zip(r['episodes'],margins)]
        assert gates==r['original_full_gates']
        if name!='candidate':assert np.max(trace[:,:,7][valid])<=1e-6
        for file,h in r['source_sha256'].items():
            path=ROOT/file
            if digest(path)!=h:
                path=OUT/{'controller.py':'controller_before.py','environment.py':'environment_before.py'}[path.name]
            assert digest(path)==h,file
        reports[name]=dict(groups=r['groups'],full_pass_count=sum(gates),startup_case_design_margin_rad=margins[10],
            step_velocity_rmse=r['episodes'][7]['velocity_rmse'],physical_steps=int(counts.sum()),
            max_same_state_shadow_delta_Nm=float(trace[:,:,7][valid].max()))
        motion=np.load(folder/'motion.npz')['trace']
        for world in (1,4,7,10):
            t=motion[:,world];moving=t[:,5]>0;projected=t[:,6]>0
            error=t[:,3]-t[:,2];total=float(t[:,4].sum());duration=float(t[:,5].sum())
            np.testing.assert_allclose(t[moving,4],error[moving]**2*.0005,rtol=0,atol=1e-16)
            assert abs(np.sqrt(total/duration)-r['episodes'][world]['velocity_rmse'])<1e-12
            index=np.clip(np.searchsorted(heights,t[:,13],side='left')-1,0,3)
            ratio=np.clip((t[:,13]-heights[index])/(heights[index+1]-heights[index]),0,1)
            k=(1-ratio[:,None,None])*gains[index]+ratio[:,None,None]*gains[index+1]
            shift=k[:,1,0]*(t[:,8]-t[:,7])/k[:,1,3]
            np.testing.assert_allclose(shift[moving],t[moving,9],rtol=0,atol=1e-12)
            np.testing.assert_allclose((t[:,11]-t[:,10])[moving],(k[:,0,3]*shift)[moving],rtol=0,atol=1e-12)
            required=int(t[0,18]);hit=np.flatnonzero(((t[:,17].astype(int)&required)!=0)&(t[:,0]>0))
            contact=None if not len(hit) else float(t[hit[0],1])
            masks=[('ramp',moving&(t[:,1]<=2)),('cruise',moving&(t[:,1]>2))]
            if contact is not None:
                masks=[('ramp',moving&(t[:,1]<=2)),('precontact',moving&(t[:,1]>2)&(t[:,1]<contact)),('contact_onward',moving&(t[:,1]>=contact))]
            phases=[]
            for label,mask in masks:
                if not mask.any():continue
                phases.append(dict(phase=label,duration_s=float(t[mask,5].sum()),squared_error_fraction=float(t[mask,4].sum()/total),
                    local_rmse=float(np.sqrt(t[mask,4].sum()/t[mask,5].sum())),projected_fraction=float(projected[mask].mean()),
                    minimum_equivalent_reference_shift_m_s=float(t[mask,9].min())))
            timing.append(dict(run=name,world=world,rmse=float(np.sqrt(total/duration)),first_contact_s=contact,phases=phases))
    assert reports['instrumented']['full_pass_count']==14
    assert reports['candidate']['full_pass_count']==reports['production_panel']['full_pass_count']==15
    assert reports['instrumented']['startup_case_design_margin_rad']<0<reports['production_panel']['startup_case_design_margin_rad']
    fixture=json.loads((OUT/'fixture.json').read_text())
    for amplitude in (1.,-1.):
        a=next(r for r in fixture['rows'] if not r['projected'] and r['amplitude']==amplitude)
        b=next(r for r in fixture['rows'] if r['projected'] and r['amplitude']==amplitude)
        assert abs(b['old_differential_H'])<1e-5 and abs(b['new_differential_H']-a['old_differential_H'])<1e-5
    assert digest(OUT/'controller_candidate.py')==digest(ROOT/'wheelleg_warp/native/controller.py')
    peak=float(re.findall(r'^PASS ([0-9.e+-]+)$',(OUT/'legacy_cpu_gpu.log').read_text(),re.M)[-1]);assert peak<1e-5
    for folder in ('nominal_boundary','packet_check'):
        assert json.loads((OUT/folder/'verification.json').read_text())['passed']
    assert 'PASS both common-boundary modes' in (OUT/'production_damping_check.log').read_text()
    result=dict(reports=reports,motion_timing=timing,legacy_cpu_gpu_command_peak_Nm=peak,
        production_fix_promoted=True,braking_references_installed_in_default_factory=False,learning=False,full_admission=False,
        limitations='Restores existing differential motor-velocity damping only. Timing correlations and equivalent references do not prove that deleting projection is safe. Same-state fixture is not a physical episode. Full18 remains15/18, with step-speed and pressure failures retained.',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/test_projection_damping.py')})
    target=OUT/'verification.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS timing, full-episode gates, original damping counterexample, production/source correspondence and regressions; full15/18, no PPO admission')


if __name__=='__main__':run()
