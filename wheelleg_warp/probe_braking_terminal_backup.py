"""Finite causal ramp backups through the original2s task; no gain or action-domain search."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from native.controller import D,fk
from native.environment import NativeEnv,reduce_contacts
from native.terrain import bank_height_115
from probe_height_115_margin import cases
from native.design import current_vmc_table
from probe_braking_feedback import Forecaster,execution_graph,record_forecast
from probe_braking_nominal_rollout import apply_extra
from select_braking_common_action import batch_scores,task_forecast
from training_contract import POST_ARRIVAL_S
from probe_height_115_action_predict_loow import sha


@wp.kernel
def record_task(slot:int,q:wp.array2d[float],ids:wp.array[int],flags:wp.array2d[int],out:wp.array3d[D]):
    w=wp.tid()
    out[slot,w,0]=(fk(D(q[w,ids[0]]),D(q[w,ids[1]]))[3]+fk(D(q[w,ids[2]]),D(q[w,ids[3]]))[3])/D(2)
    out[slot,w,1]=D(flags[w,0]);flags[w,0]=0;flags[w,1]=0


class FullForecaster(Forecaster):
    """Reuse the original predictor for supplied5ms schedules and full task evidence."""
    def __init__(self,env,candidate_count=19):
        super().__init__(env,mixed_signs=True,candidate_count=candidate_count)
        self.flags=wp.zeros((self.count,2),dtype=int);self.task_trace=wp.zeros((10,self.count,2),dtype=D)
        with wp.ScopedCapture() as capture:
            for slot in range(10):
                self.control();wp.launch(apply_extra,self.count,[self.data.qvel,self.ids,self.upper,self.extra,self.data.ctrl]);mjw.step(self.model,self.data)
                wp.launch(reduce_contacts,self.data.naconmax,[self.data.nacon,self.data.contact.worldid,self.data.contact.geom,self.ids,self.flags])
                wp.launch(record_forecast,self.count,[slot,self.data.qpos,self.data.qvel,self.data.ctrl,self.data.actuator_force,self.trace])
                wp.launch(record_task,self.count,[slot,self.data.qpos,self.ids,self.flags,self.task_trace])
        self.full_graph=capture.graph

    def forecast(self,q,v,memory,sensor,schedule):
        self.extra.assign(schedule[0]);self.data.qpos.assign(np.repeat(q,self.arms,axis=0));self.data.qvel.assign(np.repeat(v,self.arms,axis=0))
        self.data.sensordata.assign(np.repeat(sensor,self.arms,axis=0));self.state.assign(np.repeat(memory,self.arms,axis=0));self.data.qacc_warmstart.zero_();self.data.time.zero_();self.flags.zero_()
        wp.capture_launch(self.initialize_graph)
        self.state.assign(np.repeat(memory,self.arms,axis=0));self.data.sensordata.assign(np.repeat(sensor,self.arms,axis=0))
        traces=[];tasks=[]
        for action in schedule:
            self.extra.assign(action);wp.capture_launch(self.full_graph)
            traces.append(self.trace.numpy());tasks.append(self.task_trace.numpy())
            assert not self.data.overflow.numpy().any(),'预测GPU容量溢出'
        return np.concatenate(traces),np.concatenate(tasks)


def run(output):
    output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    env=NativeEnv(n=2,scenario=cases()[4:6],bank_factory=bank_height_115,height_conditioned=True,height_design='range115',
        residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=True)
    try:
        table,_=current_vmc_table()
        env.k['gains'].assign(np.stack([t[0] for t in table]));env.k['feed'].assign(np.stack([t[1] for t in table]))
        env.k['angles'].assign(np.array([t[2] for t in table]))
        env.reset();extra=wp.zeros((2,6),dtype=D);real_graph=execution_graph(env,extra)
        for iteration in range(1600):
            wp.capture_launch(real_graph);state=env.state.numpy()
            if np.all(state[:,1]>=0):break
            assert not env.done.numpy().any(),'任务在到达前结束'
        else:raise RuntimeError('未在原预算内到达')
        q=env.data.qpos.numpy();v=env.data.qvel.numpy();memory=env.k['state'].numpy();sensor=env.data.sensordata.numpy()
        p=FullForecaster(env);direction=p.candidate_changes(q,v)
        end_steps=np.rint((state[:,1]+POST_ARRIVAL_S)/.0005-state[:,0]).astype(int)
        chunks=int(np.ceil(end_steps.max()/10));schedule=[];previous=np.zeros_like(direction)
        # ponytail: nineteen frozen current-state directions; no continuous-domain
        # feasibility claim. Optimize trajectories only if this backup family fails.
        for chunk in range(chunks):
            action=np.clip(previous+direction,-1.,1.)
            assert abs(action).max()<=1. and np.sum(abs(action-previous),axis=1).max()<=.10000000001
            schedule.append(action);previous=action
        schedule=np.array(schedule);started=perf_counter();trace,task_data=p.forecast(q,v,memory,sensor,schedule);duration=(perf_counter()-started)*1000
        rows=[];selected=[]
        for world in range(2):
            sl=slice(world*p.arms,(world+1)*p.arms);steps=end_steps[world];a=trace[:steps,sl];t=task_data[:steps,sl]
            iq=np.repeat(q[world:world+1],p.arms,axis=0);iv=np.repeat(v[world:world+1],p.arms,axis=0);im=np.repeat(memory[world:world+1],p.arms,axis=0)
            cost,physical=batch_scores(p.nom,a[:,:,:p.nq],a[:,:,p.nq:p.nq+p.nv],a[:,:,-12:-6],a[:,:,-6:],iq,iv,im,p.ref,p.Q,p.R,p.P,p.project,p.input_project)
            mission=task_forecast(a[:,:,:p.nq],a[:,:,p.nq:p.nq+p.nv],iq,iv,np.repeat(state[world:world+1],p.arms,axis=0))
            rmse=np.sqrt((state[world,28]+np.sum((t[:,:,0]-.115)**2,axis=0)*.0005)/(state[world,29]+steps*.0005))
            contact=np.any(t[:,:,1]!=0,axis=0);height=(rmse<=.02)&(abs(t[-1,:,0]-.115)<=.02)
            valid=physical&mission['valid']&height&~contact
            choice=int(np.argmin(np.where(valid,cost,np.inf))) if valid.any() else None;selected.append(choice)
            rows.append(dict(world=world,elapsed_at_start_s=float(state[world,0]*.0005-state[world,1]),steps=int(steps),selected_arm=choice,
                valid=valid.tolist(),physical=physical.tolist(),task=mission['valid'].tolist(),nonwheel_contact=contact.tolist(),height_pass=height.tolist(),
                cost=cost.tolist(),stop_distance_m=mission['peak_distance_m'].tolist(),tail_speed_m_s=mission['peak_tail_speed_m_s'].tolist(),height_rmse_m=rmse.tolist()))
        # Freeze model choices before any actual future execution is observed.
        np.savez_compressed(output/'prediction.npz',initial_q=q,initial_v=v,initial_controller=memory,initial_sensor=sensor,initial_task=state,
            direction=direction,schedule=schedule,trace=trace,task_trace=task_data,end_steps=end_steps)
        episodes=[]
        if all(choice is not None for choice in selected):
            indices=np.arange(2)*p.arms+np.array(selected)
            for chunk in range(chunks):extra.assign(schedule[chunk,indices]);wp.capture_launch(real_graph)
            assert env.done.numpy().all()
            episodes=[{k:v for k,v in row.items() if k!='terminal_observation'} for row in env.step_wait()[3]]
        result=dict(role='fixed_current_state_ramp_backup_until_original_deadline',rows=rows,selected_arms=selected,episodes=episodes,
            nominal_full_task_candidates=sum(sum(row['valid']) for row in rows),actual_future_used_for_selection=False,forecast_ms=duration,
            increment_L1_limit_Nm=.1,absolute_per_motor_limit_Nm=1.,control_interval_ms=5.,original_task_and_physical_gates_unchanged=True,
            limitations='Finite19 frozen directions with causal Nom feedback and saturated ramps. Full simulator state, fixed7kg flat predictor. Neither continuous-domain infeasibility nor sensor-only, robust, recursive or real-time guarantee. Actual replay occurs only after each world has a frozen feasible candidate.',
            source_sha256={str(path.relative_to(ROOT)):sha(path) for path in (Path(__file__),ROOT/'wheelleg_warp/probe_braking_feedback.py',ROOT/'wheelleg_warp/select_braking_common_action.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/training_contract.py')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('BACKUP',selected,'actual task passes',sum(row['success'] for row in episodes),flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
