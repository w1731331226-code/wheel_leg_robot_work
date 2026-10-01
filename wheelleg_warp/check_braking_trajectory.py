"""Recorded optimized-plan gates, decoder domain and independent actual episode evidence."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.terrain import model,HeightTerrainScenario,HEIGHT_115_GEOMETRIC_MIN as LIMIT
from model_lqr import sagittal_basis
from select_braking_common_action import lqr_cost,batch_scores,batch_constraint_margins,task_forecast
from optimize_braking_trajectory import schedules
from probe_braking_feedback import Forecaster


def check(source):
    result=json.loads((source/'verification.json').read_text());m=model(HeightTerrainScenario(stand_height_m=.115))
    ref,Q,R,P=lqr_cost(m);basis,inputs=sagittal_basis(m);project=np.linalg.pinv(basis);input_project=np.linalg.pinv(inputs)
    for row in result['worlds']:
        w=row['world'];z=np.load(source/f'world{w}_best.npz',allow_pickle=False);steps=int(z['steps']);a=z['trace'][:steps];t=z['task_trace'][:steps]
        action=z['schedule'];delta=np.diff(np.concatenate([np.zeros_like(action[:1]),action]),axis=0)
        assert abs(action).max()<=1.00000000001 and np.sum(abs(delta),axis=1).max()<=.10000000001
        np.testing.assert_allclose(action,schedules(z['parameters'][None],z['common'],len(action))[:,0],atol=1e-12)
        q=a[:,None,:m.nq];v=a[:,None,m.nq:m.nq+m.nv];commands=a[:,None,-12:-6];force=a[:,None,-6:]
        cost,physical=batch_scores(m,q,v,commands,force,z['initial_q'],z['initial_v'],z['initial_controller'],ref,Q,R,P,project,input_project)
        task=task_forecast(q,v,z['initial_q'],z['initial_v'],z['initial_task']);s=z['initial_task'][0]
        assert (s[0]+steps)*.0005-s[1]>=2.-1e-10 and (s[0]+steps-1)*.0005-s[1]<2.-1e-10
        rmse=np.sqrt((s[28]+np.sum((t[:,0]-.115)**2)*.0005)/(s[29]+steps*.0005))
        margins=np.r_[batch_constraint_margins(m,q,v,commands,force,z['initial_v'])[0],.6-task['peak_distance_m'][0],.03-task['peak_tail_speed_m_s'][0],.02-rmse,.02-abs(t[-1,0]-.115),-t[:,1].max()]
        np.testing.assert_allclose(cost[0],row['best_cost'],rtol=1e-10,atol=1e-7)
        np.testing.assert_allclose(margins,z['margins'],atol=1e-12)
        assert bool(np.all(margins>=0))==row['best_feasible']
        if row['best_feasible']:assert physical[0] and task['valid'][0]
    if result['episodes']:
        assert len(result['episodes'])==2 and all(row['best_feasible'] for row in result['worlds'])
        for episode in result['episodes']:
            assert episode['success'] and episode['reason']=='completed'
            assert episode['physical_steps']==episode['physical_evidence_steps']>0 and episode['physical_safety_passed']
            assert min(episode['min_actual_A_leg_m'],episode['min_actual_B_leg_m'])>=LIMIT
            assert episode['min_eight_joint_margin_rad']>=.1  # Preserves active1.4rad inside physical1.5rad.
            assert episode['max_actual_torque_excess_Nm']<=1e-6
            assert episode['stop_distance_m']<=.6 and episode['tail_speed_m_s']<=.03 and episode['height_rmse_m']<=.02
            assert max(episode['peak_deg'])<=5
    assert not result['actual_future_used_for_selection']
    print('PASS frozen trajectory domain, full-deadline cost/gates, and actual2/2 task/physical evidence with0.1rad reserve')


def check_retained(source):
    from types import SimpleNamespace
    d=json.loads((source/'verification.json').read_text());z=np.load(source/'failed_forecast.npz',allow_pickle=False)
    f=d['failure'];assert not d['completed'] and not d['episodes'] and not d['unvalidated_actions_executed']
    assert len(d['decisions'])==f['step']+1 and all(row['valid'] for step in d['decisions'][:-1] for row in step['decisions'])
    m=model(HeightTerrainScenario(stand_height_m=.115));w=f['world'];n=f['remaining_steps'];a=z['trace'][:n,w:w+1]
    margin=batch_constraint_margins(m,a[:,:,:m.nq],a[:,:,m.nq:m.nq+m.nv],a[:,:,-12:-6],a[:,:,-6:],z['initial_v'][w:w+1])[0]
    np.testing.assert_allclose(margin,f['physical_margins'],atol=1e-12)
    assert np.flatnonzero(margin<0).tolist()==[5]  # betaR design cap, not physical1.5rad.
    assert margin[5]>-.1 and z['initial_task'][w,33]>=.1
    assert min(z['initial_task'][w,31:33])>=LIMIT
    # Single retained candidate cannot bypass the original action domain.
    p=Forecaster.__new__(Forecaster);p.n=2;p.arms=1
    assert np.array_equal(p.candidate_changes(None,None),np.zeros((2,6)))
    assert np.array_equal(p.candidate_changes(None,None,np.ones((2,6))),np.ones((2,6)))
    for bad in (np.full((2,6),1.01),np.full((2,6),np.nan),np.zeros((1,6))):
        try:p.candidate_changes(None,None,bad)
        except ValueError:pass
        else:raise AssertionError('保留动作输入未拒绝')
    for env,options in ((SimpleNamespace(stand_heights=np.array([.3])),{}),(SimpleNamespace(),{'candidate_count':2})):
        try:Forecaster(env,**options)
        except ValueError:pass
        else:raise AssertionError('实验预测器误接未核高度/候选域')
    print('PASS retained prediction rejection: design-only crossing, actual current evidence valid, invalid single-candidate inputs rejected')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--retained-source',type=Path);a=p.parse_args();check(a.source)
    if a.retained_source:check_retained(a.retained_source)
