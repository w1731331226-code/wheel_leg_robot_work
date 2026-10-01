"""Revalidate the frozen backup suffix at every5ms boundary; never assume stale feasibility."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
from native.environment import NativeEnv
from native.terrain import bank_height_115
from probe_height_115_margin import cases
from probe_current_vmc_design import current_vmc_table
from probe_braking_feedback import D,execution_graph
from probe_braking_terminal_backup import FullForecaster
from select_braking_common_action import batch_scores,batch_constraint_margins,task_forecast
from training_contract import POST_ARRIVAL_S
from probe_height_115_action_predict_loow import sha


def run(source,output):
    source=source.resolve();output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    meta=json.loads((source/'verification.json').read_text());assert all(row['best_feasible'] for row in meta['worlds'])
    plans=[np.load(source/f'world{w}_best.npz',allow_pickle=False)['schedule'] for w in range(2)]
    for plan in plans:
        assert abs(plan).max()<=1. and np.sum(abs(np.diff(np.concatenate([np.zeros_like(plan[:1]),plan]),axis=0)),axis=1).max()<=.10000000001
    env=NativeEnv(n=2,scenario=cases()[4:6],bank_factory=bank_height_115,height_conditioned=True,height_design='range115',
        residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=True)
    try:
        table,_=current_vmc_table();env.k['gains'].assign(np.stack([t[0] for t in table]));env.k['feed'].assign(np.stack([t[1] for t in table]));env.k['angles'].assign(np.array([t[2] for t in table]));env.reset()
        extra=wp.zeros((2,6),dtype=D);graph=execution_graph(env,extra)
        for _ in range(1600):
            wp.capture_launch(graph)
            if np.all(env.state.numpy()[:,1]>=0):break
            assert not env.done.numpy().any()
        else:raise RuntimeError('未在原预算内到达')
        p=FullForecaster(env,candidate_count=1);rows=[];failure=None;previous=np.zeros((2,6));durations=[]
        for step in range(max(map(len,plans))):
            if env.done.numpy().all():break
            started=perf_counter();state=env.state.numpy();active=env.active.numpy()!=0
            q=env.data.qpos.numpy();v=env.data.qvel.numpy();memory=env.k['state'].numpy();sensor=env.data.sensordata.numpy()
            schedule=np.stack([np.stack([plan[min(j,len(plan)-1)] for plan in plans]) for j in range(step,max(map(len,plans)))])
            trace,task_data=p.forecast(q,v,memory,sensor,schedule);decision=[]
            for w in np.flatnonzero(active):
                steps=int(round((state[w,1]+POST_ARRIVAL_S)/.0005-state[w,0]));assert 0<steps<=len(trace)
                a=trace[:steps,w:w+1];t=task_data[:steps,w:w+1];iq=q[w:w+1];iv=v[w:w+1];im=memory[w:w+1]
                cost,physical=batch_scores(p.nom,a[:,:,:p.nq],a[:,:,p.nq:p.nq+p.nv],a[:,:,-12:-6],a[:,:,-6:],iq,iv,im,p.ref,p.Q,p.R,p.P,p.project,p.input_project)
                margins=batch_constraint_margins(p.nom,a[:,:,:p.nq],a[:,:,p.nq:p.nq+p.nv],a[:,:,-12:-6],a[:,:,-6:],iv)
                mission=task_forecast(a[:,:,:p.nq],a[:,:,p.nq:p.nq+p.nv],iq,iv,state[w:w+1])
                rmse=np.sqrt((state[w,28]+np.sum((t[:,:,0]-.115)**2)*.0005)/(state[w,29]+steps*.0005))
                valid=bool(physical[0] and mission['valid'][0] and rmse<=.02 and abs(t[-1,0,0]-.115)<=.02 and not np.any(t[:,:,1]))
                row=dict(world=int(w),remaining_steps=steps,valid=valid,cost=float(cost[0]),stop_distance_m=float(mission['peak_distance_m'][0]),tail_speed_m_s=float(mission['peak_tail_speed_m_s'][0]),
                    height_rmse_m=float(rmse),minimum_physical_margin=float(margins.min()),physical_margins=margins[0].tolist())
                decision.append(row)
                if not valid:failure=dict(step=step,world=int(w),reason='retained_backup_lost_predicted_feasibility',**row)
            duration=(perf_counter()-started)*1000;durations.append(duration);rows.append(dict(step=step,decision_ms=duration,decisions=decision))
            if failure:
                np.savez_compressed(output/'failed_forecast.npz',initial_q=q,initial_v=v,initial_controller=memory,initial_sensor=sensor,initial_task=state,schedule=schedule,trace=trace,task_trace=task_data)
                break  # Diagnostic stop, not a certified physical fallback.
            action=schedule[0];assert abs(action).max()<=1. and np.sum(abs(action-previous),axis=1).max()<=.10000000001
            extra.assign(action);wp.capture_launch(graph);previous=action
            if step%50==0:print('RETAINED',step,'decision ms',duration,flush=True)
        completed=bool(env.done.numpy().all());episodes=[]
        if completed:episodes=[{k:v for k,v in row.items() if k!='terminal_observation'} for row in env.step_wait()[3]]
        result=dict(role='current_state_full_suffix_retained_backup_check',completed=completed,failure=failure,episodes=episodes,decisions=rows,
            decision_ms=dict(median=float(np.median(durations)),p95=float(np.percentile(durations,95)),maximum=float(max(durations))),
            feedback_interval_ms=5.,initial_planning_from_frozen_source=True,new_candidates_generated=False,unvalidated_actions_executed=False,
            limitations='Retains only the supplied frozen plan, revalidating its full remaining nominal trajectory. Full simulator state, fixed7kg flat predictor. Diagnostic stop on loss is not a certified physical fallback; no robust/recursive/real-time claim.',
            input_sha256={str((source/f'world{w}_best.npz').relative_to(ROOT)):sha(source/f'world{w}_best.npz') for w in range(2)},
            source_sha256={str(path.relative_to(ROOT)):sha(path) for path in (Path(__file__),ROOT/'wheelleg_warp/probe_braking_feedback.py',ROOT/'wheelleg_warp/probe_braking_terminal_backup.py',ROOT/'wheelleg_warp/select_braking_common_action.py')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('RETAINED RESULT',completed,failure,'actual passes',sum(e['success'] for e in episodes),flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source,a.output)
