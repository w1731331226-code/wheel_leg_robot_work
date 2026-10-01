"""Bounded full-deadline nominal trajectory solve in the original motor-extra domain."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
from scipy.optimize import minimize
from native.environment import NativeEnv
from native.terrain import bank_height_115,HEIGHT_115_GEOMETRIC_MIN as LIMIT
from probe_height_115_margin import cases
from native.design import current_vmc_table
from probe_braking_feedback import execution_graph,D
from probe_braking_terminal_backup import FullForecaster
from probe_height_115_contact_action_pair import basis
from select_braking_common_action import batch_scores,batch_constraint_margins,task_forecast
from training_contract import STOP_DISTANCE_M,TAIL_SPEED_M_S
from probe_height_115_action_predict_loow import sha

KNOTS=np.array([.005,.1,.5,1.,1.5,2.])


def schedules(parameters,common,chunks):
    """Piecewise-linear desired torques, then the original box and L1 slew limit."""
    parameters=np.asarray(parameters).reshape(-1,6,3);times=(np.arange(chunks)+1)*.005
    desired=np.stack([np.column_stack([np.interp(times,KNOTS,knots[:,axis]) for axis in range(3)]) for knots in parameters],axis=1)@common.T
    desired=np.clip(desired,-1.,1.);out=np.empty_like(desired);previous=np.zeros_like(desired[0])
    for step,target in enumerate(desired):
        change=target-previous;norm=np.sum(abs(change),axis=1)
        action=previous+change*np.minimum(1.,.1/np.maximum(norm,1e-30))[:,None]
        out[step]=action;previous=action
    return out


def solve_world(env,data,world,output):
    q=data['initial_q'][world:world+1];v=data['initial_v'][world:world+1]
    memory=data['initial_controller'][world:world+1];sensor=data['initial_sensor'][world:world+1];state=data['initial_task'][world:world+1]
    assert state[0,37]==state[0,0]>0 and min(state[0,31:33])>=LIMIT and state[0,33]>=0 and max(state[0,35:37])<=1e-6
    p=FullForecaster(env);common=basis(p.nom,q[0],v[0]);common/=np.sum(abs(common),axis=0)
    steps=int(data['end_steps'][world]);chunks=int(np.ceil(steps/10));limit=1/np.max(abs(common),axis=0)
    low=np.tile(-limit,6);high=np.tile(limit,6);x0=np.zeros((6,3));x0[:,2]=limit[2]*(1 if world==0 else -1);x0=x0.ravel()
    scale=np.r_[np.array([1.4,1.4,2.5,2.5]*2),np.full(4,.115),np.tile(200*np.array([1.4,1.4,2.5,2.5]*2),2),np.full(3,np.deg2rad(5)),np.tile([40,40,40,40,4.5,4.5],2),STOP_DISTANCE_M,TAIL_SPEED_M_S,.02,.02,1.]
    assert len(scale)==48
    cache={};log=[];best=None;normalizer=None;started=perf_counter()
    def evaluate(x):
        nonlocal best,normalizer
        if 'x' in cache and np.array_equal(x,cache['x']):return cache
        if len(log)>=120:raise RuntimeError('达到本轮120次整段预测预算')
        points=np.tile(x,(19,1));increments=np.full(18,.01);increments[x+.01>high]*=-1
        for axis in range(18):points[axis+1,axis]+=increments[axis]
        action=schedules(points,common,chunks)
        assert abs(action).max()<=1.00000000001 and np.sum(abs(np.diff(np.concatenate([np.zeros_like(action[:1]),action]),axis=0)),axis=2).max()<=.10000000001
        trace,task_data=p.forecast(q,v,memory,sensor,action);a=trace[:steps];t=task_data[:steps]
        iq=np.repeat(q,19,axis=0);iv=np.repeat(v,19,axis=0);im=np.repeat(memory,19,axis=0);istate=np.repeat(state,19,axis=0)
        costs,physical=batch_scores(p.nom,a[:,:,:p.nq],a[:,:,p.nq:p.nq+p.nv],a[:,:,-12:-6],a[:,:,-6:],iq,iv,im,p.ref,p.Q,p.R,p.P,p.project,p.input_project)
        margins=batch_constraint_margins(p.nom,a[:,:,:p.nq],a[:,:,p.nq:p.nq+p.nv],a[:,:,-12:-6],a[:,:,-6:],iv)
        task=task_forecast(a[:,:,:p.nq],a[:,:,p.nq:p.nq+p.nv],iq,iv,istate)
        rmse=np.sqrt((state[0,28]+np.sum((t[:,:,0]-.115)**2,axis=0)*.0005)/(state[0,29]+steps*.0005))
        contact=np.max(t[:,:,1],axis=0);tailheight=abs(t[-1,:,0]-.115)
        all_margins=np.c_[margins,STOP_DISTANCE_M-task['peak_distance_m'],TAIL_SPEED_M_S-task['peak_tail_speed_m_s'],.02-rmse,.02-tailheight,-contact]
        constraints=all_margins/scale
        assert np.isfinite(costs).all() and np.isfinite(constraints).all()
        valid=physical&task['valid']&(rmse<=.02)&(tailheight<=.02)&(contact==0)
        if normalizer is None:normalizer=costs[0]
        cost=costs/normalizer;jac=(cost[1:]-cost[0])/increments;conjac=((constraints[1:]-constraints[0])/increments[:,None]).T
        violation=float(np.maximum(-constraints[0],0).max())
        row=dict(evaluation=len(log),cost=float(costs[0]),maximum_scaled_violation=violation,stop_distance_m=float(task['peak_distance_m'][0]),tail_speed_m_s=float(task['peak_tail_speed_m_s'][0]),feasible=bool(valid[0]),parameters=x.tolist())
        log.append(row);(output/f'world{world}_progress.json').write_text(json.dumps(log,indent=2)+'\n')
        print('SOLVE',world,len(log),'stop',row['stop_distance_m'],'tail',row['tail_speed_m_s'],'violation',violation,flush=True)
        if best is None or (bool(valid[0]),-violation,-costs[0])>(best['valid'],-best['violation'],-best['cost']):
            best=dict(valid=bool(valid[0]),violation=violation,cost=float(costs[0]),x=x.copy(),action=action[:,0].copy(),trace=trace[:,0].copy(),task_trace=task_data[:,0].copy(),margins=all_margins[0].copy())
        cache.clear();cache.update(x=x.copy(),cost=cost[0],jac=jac,constraints=constraints[0],conjac=conjac)
        return cache
    # ponytail: six existing clock/stop landmarks and one-sided batched FD;
    # refine trajectory discretization/derivatives only after this bounded solve.
    failure=None
    try:
        result=minimize(lambda x:evaluate(x)['cost'],x0,jac=lambda x:evaluate(x)['jac'],method='SLSQP',bounds=list(zip(low,high)),
            constraints={'type':'ineq','fun':lambda x:evaluate(x)['constraints'],'jac':lambda x:evaluate(x)['conjac']},options={'maxiter':40,'ftol':1e-7})
        evaluate(result.x);status=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))
    except RuntimeError as exc:failure=str(exc);status=dict(success=False,message=failure)
    assert best is not None
    np.savez_compressed(output/f'world{world}_best.npz',initial_q=q,initial_v=v,initial_controller=memory,initial_sensor=sensor,initial_task=state,
        common=common,parameters=best['x'],schedule=best['action'],trace=best['trace'],task_trace=best['task_trace'],margins=best['margins'],steps=steps,knots=KNOTS)
    return dict(world=world,solver=status,evaluations=len(log),wall_ms=(perf_counter()-started)*1000,best_feasible=best['valid'],best_cost=best['cost'],best_scaled_violation=best['violation'],failure=failure)


def run(source,output):
    output=output.resolve();source=source.resolve();assert not output.exists();output.mkdir(parents=True)
    data=np.load(source/'prediction.npz',allow_pickle=False);table,_=current_vmc_table();rows=[]
    for world in range(2):
        env=NativeEnv(n=1,scenario=[cases()[4+world]],bank_factory=bank_height_115,height_conditioned=True,height_design='range115',residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=True)
        try:
            env.k['gains'].assign(np.stack([t[0] for t in table]));env.k['feed'].assign(np.stack([t[1] for t in table]));env.k['angles'].assign(np.array([t[2] for t in table]))
            rows.append(solve_world(env,data,world,output))
        finally:env.close()
    episodes=[]
    if all(row['best_feasible'] for row in rows):
        plans=[np.load(output/f'world{world}_best.npz',allow_pickle=False)['schedule'] for world in range(2)]
        env=NativeEnv(n=2,scenario=cases()[4:6],bank_factory=bank_height_115,height_conditioned=True,height_design='range115',residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=True)
        try:
            env.k['gains'].assign(np.stack([t[0] for t in table]));env.k['feed'].assign(np.stack([t[1] for t in table]));env.k['angles'].assign(np.array([t[2] for t in table]));env.reset()
            extra=wp.zeros((2,6),dtype=D);graph=execution_graph(env,extra)
            for _ in range(1600):
                wp.capture_launch(graph)
                if np.all(env.state.numpy()[:,1]>=0):break
                assert not env.done.numpy().any()
            for step in range(max(map(len,plans))):
                extra.assign(np.stack([plan[min(step,len(plan)-1)] for plan in plans]));wp.capture_launch(graph)
            assert env.done.numpy().all()
            episodes=[{k:v for k,v in row.items() if k!='terminal_observation'} for row in env.step_wait()[3]]
        finally:env.close()
    result=dict(role='bounded_full_deadline_nominal_SLSQP_trajectory',worlds=rows,episodes=episodes,actual_future_used_for_selection=False,
        original_costs_and_gates_unchanged=True,knots_s=KNOTS.tolist(),increment_L1_limit_Nm=.1,absolute_per_motor_limit_Nm=1.,finite_difference_step=.01,
        limitations='Offline finite18-parameter nominal trajectory solver; full simulation state, no continuous-domain infeasibility or sensor/robust/recursive/real-time guarantee. Failed plans never executed.',
        input_sha256={str((source/'prediction.npz').relative_to(ROOT)):sha(source/'prediction.npz')},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_braking_terminal_backup.py',ROOT/'wheelleg_warp/select_braking_common_action.py',ROOT/'wheelleg_warp/probe_braking_feedback.py')})
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('RESULT',rows,'actual successes',sum(row['success'] for row in episodes),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)
