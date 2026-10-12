"""Saved-data numerical contract review; never constructs a robot or launches Warp."""
import json
from pathlib import Path
import numpy as np
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_training_interface_v1'


def ulps(a,b):
    a,b=map(np.float64,(a,b))
    if not np.isfinite([a,b]).all():raise ValueError('Finite double values required')
    if a==b:return 0
    if np.signbit(a)!=np.signbit(b):raise ValueError('This review compares same-sign deltas only')
    return abs(int(a.view(np.int64))-int(b.view(np.int64)))


def self_check():
    for value in (1.,-1.,1e-9,-1e-9):assert ulps(value,np.nextafter(value,np.inf))==1
    assert ulps(0.,-0.)==0
    for a,b in ((np.nan,0.),(-1.,1.)):
        try:ulps(a,b)
        except ValueError:pass
        else:raise AssertionError('Unsupported ULP comparison admitted')
    # Observation equations have a null direction even with unchanged endpoint rates.
    delta=np.zeros(40);delta[10]=1e-9
    gyro=(delta-.95*np.r_[0,delta[:-1]])/.05;roll=-.4*delta
    memory=0.;rates=[]
    for value in gyro:memory=.95*memory+.05*value;rates.append(memory)
    np.testing.assert_allclose(rates,delta,rtol=0,atol=1e-24)
    np.testing.assert_allclose(.3*roll+.12*np.array(rates),0,rtol=0,atol=1e-24)
    assert gyro[0]==gyro[-1]==roll[0]==roll[-1]==memory==0 and np.any(gyro!=0)
    print('PASS369 double ULP checks and nonunique hidden roll/gyro measurement histories;0physics',flush=True)


def run():
    self_check();file=OUT/'numerical_contract_review.json';assert not file.exists()
    audit=json.loads((OUT/'zero_failure_audit.json').read_text());admit=json.loads((OUT/'source_admission.json').read_text())
    assert all(sha(ROOT/name)==digest for name,digest in audit['input_sha256'].items())
    assert all(sha(ROOT/name)==digest for name,digest in admit['source_sha256'].items())
    assert audit['status']=='strict_zero_prefix_failed_no_interface_admission'
    assert not (OUT/'runtime/completion.json').exists()
    details={}
    with np.load(OUT/'runtime/B0/zero_prefix.npz') as a,np.load(OUT/'runtime/CartLive3/zero_prefix.npz') as b:
        x,y=a['controller_memory'][6,5],b['controller_memory'][6,5]
        details['memory5']=dict(world=6,absolute_delta=float(abs(x-y)),double_spacing=float(abs(np.spacing(x))),
            double_ULP_distance=ulps(x,y),relative_delta=float(abs(x-y)/abs(x)))
        for column in (6,7,8):
            x,y=a['phase_1'][:,:,column],b['phase_1'][:,:,column];mask=x!=y
            details[f'phase_column_{column}']=dict(different_values=int(mask.sum()),
                maximum_absolute_delta=float(abs(x-y).max()),maximum_double_ULP_distance=max(ulps(q,r) for q,r in zip(x[mask],y[mask])))
    assert details['memory5']['double_ULP_distance']==16066441
    atomic_json(file,dict(round=369,status='saved_data_identifiability_limit_and_contract_decision',
        input_audit_sha256=sha(OUT/'zero_failure_audit.json'),admission_sha256=sha(OUT/'source_admission.json'),
        reviewer_sha256=sha(__file__),verified_original_sources=len(admit['source_sha256']),numerical_details=details,
        source_chain=dict(filter='reference_role_control:memory5 += .05*(gyro-memory5)',
            offset='reference_role_probe:offset=clamp(.3*roll+.12*next_filtered_rate,+/-.035)',
            direct_cartesian_native_writes='Only16:22. clear_native and deliver do not directly write5.',
            other_writers='Native reset_rows zeros the whole memory before the packet. No done/reset occurred inside these40steps;shared-reference memory starts atoffset26.'),
        findings=[
            'Tiny absolute delta is not one/few double ULPs at the recorded near-zero gyro value;no tolerance can be inferred from absolute size alone.',
            '40 combined offsets and endpoint memory/sensor do not uniquely identify intermediate body roll,gyro or arithmetic. The null-direction selfcheck concerns measurement equations,not another physically valid trajectory.',
            'First phase offset mismatch atslot5 is first observed combined diagnostic mismatch,not proven first input/state divergence.',
            'Original B0 constructed onlystage1;Cart constructed three-stage caches. Model/meaninertia and per-substep full-body inputs were not archived by the original snapshot. Exact endpoint arrays do not remove this construction/information gap.',
            'A numerical/repeatability/capture effect remains a hypothesis. The failed records do not demonstrate a Cartesian formula/control-memory5 defect or harmless long-horizon behavior.'
        ],
        decision='Stop further inference from these same saved prefixes. Keep368 failure and all exact gates.370 must accept one bounded matched-construction input-chain diagnostic or stop this runtime admission route;no additional generic source-audit rounds.',
        proposed_diagnostic='Three fresh zero packets:matching three-stage B0 twice,matching CartLive3 once;1200graphworldsteps,pre-callFD80. Pending370 direction review and new source/runtime admission;not yet executed.',
        forbidden=['Delete differing fields or round logs to pass368','Choose a tolerance from the observed deltas','Resume368 failed directory or spend its remaining budget','Infer PPO benefit or admit learning'],
        new_physics_steps=0,new_FD=0,new_model_forward_rows=0,new_learning_samples=0,formal5_admitted=False,
        light_interface_admitted=False,goal_complete=False))
    print('REVIEW369 memory double ULPs',details['memory5']['double_ULP_distance'],'specific root cause unidentifiable;oldfailure unchanged',flush=True)


if __name__=='__main__':run()
