"""Independent saved50 review; no bank construction or controller/physics rerun."""
import ast
import json
import sys
from pathlib import Path
import numpy as np
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import wheelleg_sim as sim
import model_lqr as ml

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_map_v1'


def run():
    assert not (OUT/'query_review.json').exists()
    p,c,k,d=[json.loads((OUT/n).read_text()) for n in
             ('proposal.json','query_completion.json','query_contract.json','query_delivery.json')]
    assert c['verified'] and c['individual_static_queries']==50
    assert c['source_contract_sha256']==sha(OUT/'query_contract.json')
    assert d['query_completion_sha256']==sha(OUT/'query_completion.json')
    assert all(sha(ROOT/n)==h for n,h in k['source_sha256'].items())
    assert k['proposal_sha256']==sha(OUT/'proposal.json') and k['bank_exact_frozen']
    original=ast.parse((ROOT/'wheelleg_warp/reference_role_control.py').read_text())
    copy=ast.parse((ROOT/'wheelleg_warp/continuous_nominal_control.py').read_text())
    fn=next(n for n in copy.body if isinstance(n,ast.FunctionDef) and n.name=='control_step')
    extra=[n for n in fn.body if isinstance(n,ast.If) and 'reference.shape[1] == 21' in ast.unparse(n.test)]
    assert len(extra)==3
    fn.body=[n for n in fn.body if n not in extra]
    assert ast.dump(copy)==ast.dump(original)
    arrays={}
    for arm in ('old','map'):
        path=OUT/f'queries_{arm}.npz'
        records=[r for r in c['records'] if r['arm']==arm]
        assert len(records)==25 and all(r['query_sha256']==sha(path) for r in records)
        with np.load(path,allow_pickle=False) as z:arrays[arm]={n:z[n].copy() for n in z.files}
        x=arrays[arm]
        for n in ('ctrl','diag','memory_before','memory_after'):assert np.isfinite(x[n]).all()
        np.testing.assert_array_equal(x['q'],x['original_q'].astype(np.float32))
        np.testing.assert_array_equal(x['v'],x['original_v'].astype(np.float32))
        np.testing.assert_array_equal(x['ctrl_minus_required'],x['ctrl'].astype(float)-x['required_ctrl'])
        for n in ('targets','nominal_correction','clock','sensor'):np.testing.assert_array_equal(x[n],0.)
        np.testing.assert_array_equal(x['diag'][:,14],0.)
        for r in records:
            i=r['world'];ref=k['references'][i];case=k['constructor_cases'][i]['scenario']
            assert r['height']==ref['height']==x['base_reference'][i,2]
            assert r['speed']==ref['speed']==x['command'][i]
            assert .5<=abs(case['speed'])<=1. and case['mass']==7. and case['drive_difference']==0.
            if ref['speed']!=0:assert np.sign(case['speed'])==np.sign(ref['speed'])
            assert float(abs(x['ctrl_minus_required'][i]).max())==r['maximum_command_gap_Nm']
            assert float(abs(x['ctrl_minus_required'][i,-2:]).max())==r['wheel_gap_Nm']
            assert float(abs(x['diag'][i,15:21]-x['required_ctrl'][i]).max())==r['raw_nominal_gap_Nm']
    old,new=arrays['old'],arrays['map']
    for n in ('q','v','sensor','command','environment_state','clock','active','targets','nominal_correction',
              'heights','gains','feed','angles','memory_before','base_reference','required_ctrl','phase'):
        np.testing.assert_array_equal(old[n],new[n])
    np.testing.assert_array_equal(old['reference'],new['reference'][:,:16])
    for n in ('ctrl','diag','memory_after'):np.testing.assert_array_equal(old[n][20:],new[n][20:])
    for n in ('role','phase'):np.testing.assert_array_equal(old[n][:,20:],new[n][:,20:])
    np.testing.assert_array_equal(new['reference'][20:,16:],0.)
    with np.load(OUT/'delta_table.npz') as z:
        h,v,t=z['height_knots'],z['speed_knots'],z['delta']
        for i,r in enumerate(k['references'][:20]):
            hi=min(np.searchsorted(h,r['height'],side='right')-1,len(h)-2)
            vi=min(np.searchsorted(v,r['speed'],side='right')-1,len(v)-2)
            a=(r['height']-h[hi])/(h[hi+1]-h[hi]);b=(r['speed']-v[vi])/(v[vi+1]-v[vi])
            weights=np.array([(1-a)*(1-b),(1-a)*b,a*(1-b),a*b])
            value=weights@np.array([t[hi,vi],t[hi,vi+1],t[hi+1,vi],t[hi+1,vi+1]])
            np.testing.assert_allclose(value,new['reference'][i,16:20],rtol=0,atol=1e-12)
            assert new['reference'][i,20]==1.
    # Independent motor envelope check in public nominal model, no bank/FD.
    m,_=sim.load_model(ml.XML,True)
    dofs=[m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
    for x in arrays.values():
        for i in range(25):
            limits=np.array([sim.hw.torque_limit(float('inf'),float(x['v'][i,n]),j<4,0.,.0005)[0] for j,n in enumerate(dofs)])
            assert np.maximum(abs(x['ctrl'][i])-limits,0.).max()<=1e-6
    gaps_old=np.max(abs(old['ctrl_minus_required'][:20]),axis=1)
    gaps_map=np.max(abs(new['ctrl_minus_required'][:20]),axis=1)
    worse=[]
    for i in np.flatnonzero(gaps_map>gaps_old):
        r=k['references'][i]
        worse.append(dict(world=int(i),height=r['height'],speed=r['speed'],
            old_gap_Nm=float(gaps_old[i]),map_gap_Nm=float(gaps_map[i]),
            delta_gap_Nm=float(gaps_map[i]-gaps_old[i]),
            largest_old_component=int(np.argmax(abs(old['ctrl_minus_required'][i]))),
            largest_map_component=int(np.argmax(abs(new['ctrl_minus_required'][i]))),
            old_component_gaps_Nm=old['ctrl_minus_required'][i].tolist(),
            map_component_gaps_Nm=new['ctrl_minus_required'][i].tolist()))
    assert len(worse)==d['validation_worse']==2
    assert int((gaps_map<gaps_old).sum())==d['validation_improved']==18
    assert len(k['baseline_constructor_transitionFD_calls'])==10
    prior=json.loads((OUT/'query_preconstructor_failure.json').read_text())
    assert prior['attempted_individual_queries']==prior['baseline_constructor_transitionFD_calls']==0
    atomic_json(OUT/'query_review.json',dict(
        verified=True,round=283,reviewer_sha256=sha(__file__),query_completion_sha256=sha(OUT/'query_completion.json'),
        table_sha256=sha(OUT/'delta_table.npz'),same_query_inputs=True,zero_runtime_exact=True,
        true_low_speed_commands_verified=True,source_only_declared_changes=True,motor_bounds_passed=True,
        validation_improved=18,validation_worse=2,worse=worse,baseline_transitionFD_calls_confirmed=10,
        fresh_model_compile_only=True,new_queries=0,integration_steps=0,transitionFD_calls=0,new_training_samples=0,
        production_admitted=False,
        decision='Close50static screen, keep frozen table andtwo counterexamples. Worth only a fixed full start/move/stop development task pair; no more statics/refit. Task preservation and benefit must decide continuation.',
        limits='Independent saved data/source/interpolation/envelope review,not freshcontroller query. Twohipdominated worsening points cannotbe explained uniquely by this static evidence. No robust/stability/novelmethod/PPO claim.'))
    print('PASS283 independent50/truecmd/zero/motorbounds;18better/2worse retained; no newqueries',flush=True)


if __name__=='__main__':
    run()
