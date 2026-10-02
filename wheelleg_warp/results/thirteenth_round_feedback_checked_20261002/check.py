"""Verify the causal-history comparison and retained failure without new execution."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.terrain import model,HeightTerrainScenario
from select_braking_common_action import batch_constraint_margins


def check():
    out=Path(__file__).resolve().parent;initial=out.parent/'thirteenth_round_past_state_support_20261002'
    a=json.loads((initial/'verification.json').read_text());z=np.load(initial/'pairing.npz');m=model(HeightTerrainScenario(stand_height_m=.115))
    for row,key in zip(a['rows'],('current_init','past_step_init')):
        p=z[key];t=z['actual'];q=float(abs(p[:,:,:m.nq]-t[:,:,:m.nq]).max());v=float(abs(p[:,:,m.nq:m.nq+m.nv]-t[:,:,m.nq:m.nq+m.nv]).max());c=float(abs(p[:,:,-12:-6]-t[:,:,-12:-6]).max())
        assert (q,v,c)==(row['qpos'],row['qvel'],row['command'])
        assert row['original_gate_pass']==bool(q<=2e-6 and v<=1e-3 and c<=1e-5)
    assert not a['rows'][0]['original_gate_pass'] and a['rows'][1]['original_gate_pass']
    r=json.loads((out/'verification.json').read_text());assert not r['completed'] and r['failure']['step']==95 and len(r['decisions'])==96
    assert all(x['qpos']<=2e-6 and x['qvel']<=1e-3 and x['command']<=1e-5 for x in r['pairing_errors'][:-1])
    pair=np.load(out/'failed_pair.npz');c=float(abs(pair['predicted'][:,:,-12:-6]-pair['actual'][:,:,-12:-6]).max())
    assert c==r['failure']['command'] and c>1e-5
    actual=np.load(out/'actual.npz');u=actual['requests'];du=np.diff(np.concatenate([np.zeros_like(u[:1]),u]),axis=0)
    assert abs(u).max()<=1 and np.sum(abs(du),axis=2).max()<=.1+1e-12
    q=actual['trace'][:,:,:m.nq];ids=[m.joint(n).qposadr[0] for n in ('alphaL','betaL','alphaR','betaR')]
    margin=float(np.min(1.4-abs(q[:,:,ids])));assert margin>=0
    plans=np.stack([np.load(out.parent/'braking_trajectory_slsqp_20261001'/f'world{w}_best.npz')['schedule'][:len(u)] for w in range(2)],axis=1)
    np.testing.assert_array_equal(u,plans)
    b=json.loads((out/'batch_check.json').read_text());assert all(x['pass_original'] for x in b['rows'])
    for folder,record in ((initial,a),(out,r)):
        for name,h in record['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    result=dict(first_window_past_replay_passed=True,first95_controller_windows_pair_passed=True,failed_window=95,original_command_gate_not_relaxed=True,
        actual_active_design_margin_rad=margin,requests_equal_nominal_guide=True,state_feedback_improvement_claimed=False,
        fresh_instance_both_batch_counts_passed_same_failure_state=True,sole_batch_size_cause_claimed=False,full_task_admission=False,learning=False)
    target=out/'checks.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS evidence: causal history fixes first window,95 windows then original rejection, legal requests/actual design, fresh-instance contrast; no admission')


if __name__=='__main__':check()
