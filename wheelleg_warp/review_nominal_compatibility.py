"""Independent savedquery/source review;no controller construction or queries."""
import ast
import json
from pathlib import Path
import numpy as np
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_command_compatibility_v1'


def run():
    assert not (OUT/'review.json').exists()
    p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'completion.json').read_text())
    contract=json.loads((OUT/'source_contract.json').read_text())
    assert c['verified'] and c['individual_static_queries']==30 and not c['physics_clock_advanced']
    assert c['source_contract_sha256']==sha(OUT/'source_contract.json') and contract['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/n)==v for n,v in contract['source_sha256'].items())
    values=[]
    for j,integral in enumerate(p['speed_integral_states']):
        path=OUT/f'integral_{j}.npz';rows=[r for r in c['records'] if r['state11']==integral]
        assert len(rows)==10 and all(r['sha256']==sha(path) for r in rows)
        with np.load(path,allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
        assert float(x['integral'])==integral
        np.testing.assert_array_equal(x['memory_before'][:,11],integral)
        np.testing.assert_array_equal(x['memory_after'][:,11],integral)
        np.testing.assert_array_equal(x['q'],x['original_q'].astype(np.float32));np.testing.assert_array_equal(x['v'],x['original_v'].astype(np.float32))
        np.testing.assert_array_equal(x['ctrl_minus_required'],x['ctrl'].astype(float)-x['required_ctrl'])
        np.testing.assert_array_equal(x['diag'][:,14],0)
        for r in rows:
            i=r['world'];np.testing.assert_array_equal(x['ctrl'][i],r['ctrl'])
            assert float(abs(x['ctrl_minus_required'][i]).max())==r['maximum_command_gap_Nm']
        values.append(x)
    keep=np.delete(np.arange(values[0]['memory_before'].shape[1]),11)
    for j in (0,2):
        for name in ('q','v','sensor','command','environment_state','reference','required_ctrl','original_q','original_v'):
            np.testing.assert_array_equal(values[j][name],values[1][name])
        np.testing.assert_array_equal(values[j]['memory_before'][:,keep],values[1]['memory_before'][:,keep])
        np.testing.assert_array_equal(values[j]['ctrl'],values[1]['ctrl'])
        np.testing.assert_array_equal(values[j]['diag'][:,15:21],values[1]['diag'][:,15:21])
    tree=ast.parse((ROOT/'wheelleg_warp/reference_role_control.py').read_text())
    assignments={}
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            assignments.setdefault(n.targets[0].id,n.value)
    assert isinstance(assignments['x'],ast.Call)
    assert 'thcmd' not in {n.id for n in ast.walk(assignments['x']) if isinstance(n,ast.Name)}
    assert 'state[w, 11]' in ast.unparse(assignments['thcmd'])
    assert ast.unparse(assignments['average'])=='(vl[1] + vr[1]) / D(2)'
    assert ast.unparse(assignments['hl'])=='vl[1] + hub - average' and ast.unparse(assignments['hr'])=='vr[1] + hub - average'
    # Algebra:((vl+hub-(vl+vr)/2)+(vr+hub-(vl+vr)/2))/2=hub.
    sample=np.random.default_rng(267).normal(size=(256,3));left,right,hub=sample.T
    mean=.5*(left+hub-.5*(left+right)+right+hub-.5*(left+right))
    assert np.all(abs(mean-hub)<=32*np.finfo(float).eps*(1+abs(left)+abs(right)+abs(hub)))
    build_log=(OUT/'run.log').read_text()
    reductions=sum(line.startswith('reduction ') for line in build_log.splitlines())
    # Gain construction is itself offline design;it must not be described as zero FD work.
    constructor_source=dict(nominal_design_115='rm_controller.design_controller ->5ml.design/linearize calls',
        height115_factory='NativeEnv.height115_candidate ->current_vmc_table ->5ml.design/linearize calls',
        logged_reduction_reports=reductions,baseline_gain_construction_transitionFD_calls=10)
    assert reductions==10
    gaps=values[1]['ctrl_minus_required'];wheel_gap=np.max(abs(gaps[:,-2:]),axis=1)
    atomic_json(OUT/'review.json',dict(verified=True,completion_sha256=sha(OUT/'completion.json'),reviewer_sha256=sha(__file__),
        recorded_inputs_only_state11_differ=True,raw_and_final_command_endpoint_effect_exact0=True,
        projection_error_all0=True,mean_hub_cancels_old_hub_algebra=True,LQR_x_omits_thcmd=True,
        wheel_command_gap_Nm=wheel_gap.tolist(),maximum_fullcommand_gap_Nm=float(abs(gaps).max()),
        constructor_cost_correction=constructor_source,new_queries=0,new_integration_steps=0,new_transitionFD_calls=0,new_training_samples=0,
        controller_admitted=False,
        decision='Staticmotionfixturesconfirmstate11 legacycommonpath is canceled before LQR mean;plantreference is notexact currentNomclosedloop equilibrium. Stop30queryscreen;finite correction+pairedregression decision next.',
        limits='Algebra pre-angleprojection;measured30mirrorendpoints only,not entire asymmetry/braking/memorycontinuum. Notuniquecause oldRLfailures/stability ornewmethod contribution. Constructor10offlineFD were baselinegainbuilding,notadditionalexperimentalqueries;correct earlierzeroFD-total claim.'))
    print('PASS267 saved30query/source-algebra;NoNewQueries;baselineconstructorFD10 accounted',flush=True)


if __name__=='__main__':run()
