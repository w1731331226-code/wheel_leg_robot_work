"""Verify rejected interpolation evidence without re-enabling its implementation."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def sources(record):
    for name,digest in record['source_sha256'].items():
        path=ROOT/name
        choices=[path,OUT/'source_v2'/name]
        aliases={'environment.py':'environment_at_v1.py','braking_reference.py':'braking_reference_at_v1.py','panel.py':'panel_at_v1.py'}
        if path.name in aliases:choices.append(OUT/aliases[path.name])
        assert any(p.is_file() and sha(p)==digest for p in choices),name


def run():
    package=OUT/'source_v2/wheelleg_warp/native/braking_reference.npz'
    data=np.load(package,allow_pickle=False);plan=data['schedule']
    assert plan.shape==(400,2,6) and np.isfinite(plan).all() and abs(plan).max()<=1
    assert abs(np.diff(np.concatenate([np.zeros_like(plan[:1]),plan]),axis=0)).sum(axis=2).max()<=.1+1e-12
    for i,(path,digest) in enumerate(zip(data['source_paths'],data['source_sha256'])):
        source=ROOT/str(path);assert sha(source)==digest
        np.testing.assert_array_equal(plan[:,i],np.load(source,allow_pickle=False)['schedule'])
    panels={}
    for mode in ('registered','grid','registered_v2','grid_v2','cell_v2'):
        r=json.loads((OUT/mode/'verification.json').read_text());sources(r)
        assert r['maximum_native_request_error_Nm']<=1e-14 and not r['manual_injection'] and not r['learning']
        rows=r['episodes'];labels=r['registration']['labels'];groups={}
        for e in rows:
            assert e['physical_steps']==e['physical_evidence_steps'] and e['design_joint_contract']=='active-1p4-v1'
            assert e['design_joint_passed']==(e['min_active_design_margin_rad']>=0)
            assert not e['success'] or (e['physical_safety_passed'] and e['design_joint_passed'])
        for label in dict.fromkeys(labels):
            selected=[e for e,l in zip(rows,labels) if l==label]
            groups[label]=dict(total=len(selected),physical=sum(e['physical_safety_passed'] for e in selected),
                design=sum(e['design_joint_passed'] for e in selected),task=sum(e['success'] for e in selected))
        assert groups==r['groups'];panels[mode]=groups
    contrast=json.loads((OUT/'contrast.json').read_text());sources(contrast)
    assert all(e['success'] for e in contrast['results'][0]['episodes'])
    assert all(not e['success'] for e in contrast['results'][1]['episodes'])
    assert panels['grid_v2']['nominal']['task']==42 and panels['grid_v2']['training_corner']['task']==42
    assert panels['cell_v2']['training_corner']['task']==36
    for file in ('fixture.log','fixture_v2.log'):assert 'PASS native reference:' in (OUT/file).read_text()
    assert sha(ROOT/'wheelleg_warp/native/environment.py')==sha(OUT/'environment_before.py')
    assert not (ROOT/'wheelleg_warp/native/braking_reference.py').exists()
    assert not (ROOT/'wheelleg_warp/native/braking_reference.npz').exists()
    result=dict(panels=panels,paired_failures_pass_without_reference=True,
        scheduling_and_request_domain_verified=True,continuous_reference_rule_rejected=True,
        failed_active_branch_removed=True,production_remains_v5=True,learning=False,full_admission=False,
        conclusion='Feasible operating endpoints and a passing coarse grid do not certify interpolated motor sequences; fine-cell failures remain.',
        limitations='Public deterministic panels and descriptive paired reruns only. No universal infeasibility or robust-policy claim. Replay the archived source tree in an isolated checkout to reproduce experiments; active v5 intentionally has no failed automatic-reference option.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target=OUT/'verification.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS native scheduling; reject interpolation: coarse84/84 masks fine86/100. Production restored to v5; no admission.')


if __name__=='__main__':run()
