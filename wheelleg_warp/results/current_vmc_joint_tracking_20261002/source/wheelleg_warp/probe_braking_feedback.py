"""Independent5ms finite-candidate feedback pilot; original cost and small extra-action domain."""
import argparse,json,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from native.controller import D,V6,command_bounds,control_physical
from native.environment import NativeEnv,begin,command_step,reduce_contacts,collect_physical,after
from native.models import batch
from native.terrain import model,HeightTerrainScenario,bank_height_115
from probe_height_115_margin import cases
from probe_height_115_contact_action_pair import basis
from probe_braking_nominal_rollout import apply_extra
from probe_current_vmc_design import current_vmc_table
from select_braking_common_action import lqr_cost,batch_scores,task_forecast
from model_lqr import sagittal_basis
from probe_height_115_action_predict_loow import sha


@wp.kernel
def record_forecast(slot:int,q:wp.array2d[float],v:wp.array2d[float],ctrl:wp.array2d[float],force:wp.array2d[float],out:wp.array3d[D]):
    w=wp.tid();nq=q.shape[1];nv=v.shape[1]
    for j in range(nq):out[slot,w,j]=D(q[w,j])
    for j in range(nv):out[slot,w,nq+j]=D(v[w,j])
    for j in range(6):
        out[slot,w,nq+nv+j]=D(ctrl[w,j]);out[slot,w,nq+nv+6+j]=D(force[w,j])


@wp.kernel
def execute_extra(v:wp.array2d[float],ids:wp.array[int],upper:wp.array2d[D],extra:wp.array2d[D],active:wp.array[int],ctrl:wp.array2d[float],diag:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    speed=V6();gain=V6()
    for j in range(6):speed[j]=D(v[w,ids[4+j]]);gain[j]=upper[w,j]
    bound=command_bounds(speed,gain)
    for j in range(6):
        old=D(ctrl[w,j]);new=wp.clamp(old+extra[w,j],-bound[j],bound[j]);ctrl[w,j]=float(new);diag[w,6+j]=new-old


def execution_graph(env,extra,extra_update=None):
    if env.height_safety!='physical_v1':raise ValueError('独立执行图需要真实物理契约')
    n=env.num_envs;upper=env.control_extra[0]
    with wp.ScopedCapture() as capture:
        wp.launch(begin,n,[env.reward])
        for _ in range(10):
            d=env.data;wp.launch(command_step,n,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(env.control_kernel,n,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
                env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
            if extra_update is not None:wp.launch(extra_update[0],n,extra_update[1])
            wp.launch(execute_extra,n,[d.qvel,env.ids,upper,extra,env.active,d.ctrl,env.diag]);mjw.step(env.model,d)
            wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags]);wp.launch(collect_physical,n,env.physical_args)
            wp.launch(after,n,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,
                env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    return capture.graph


class Forecaster:
    # ponytail: finite fixed candidates; continuous optimization only if this
    # verified grid is the limiting factor. This is not a real-time controller.
    def __init__(self,env,mixed_signs=False,candidate_count=None):
        if candidate_count is not None and (type(candidate_count) is not int or candidate_count not in (1,13,19)):raise ValueError('预测候选数须为1、13或19')
        if not np.all(np.asarray(env.stand_heights)==.115):raise ValueError('当前独立预测器仅核过0.115m，不能用于其他高度')
        self.n=env.num_envs;self.arms=candidate_count if candidate_count is not None else 19 if mixed_signs else 13;self.count=self.n*self.arms;self.nom=model(HeightTerrainScenario(stand_height_m=.115))
        _,self.model,self.data,_=batch([self.nom]*self.count,[HeightTerrainScenario(stand_height_m=.115)]*self.count)
        self.ids=wp.array(env.ids.numpy(),dtype=int);self.upper=wp.array(np.tile(env.actuator_gain_upper,(self.count,1)),dtype=D)
        self.state=wp.zeros((self.count,env.k['state'].shape[1]),dtype=D);self.reference=wp.array(np.repeat(env.k['reference'].numpy(),self.arms,axis=0),dtype=D)
        self.fixed={k:wp.array(env.k[k].numpy(),dtype=D) for k in ('heights','gains','feed','angles','yaw')}
        self.targets=wp.zeros((self.count,3));self.command=wp.zeros(self.count,dtype=D);self.active=wp.ones(self.count,dtype=int)
        self.diag=wp.zeros((self.count,38),dtype=D);self.extra=wp.zeros((self.count,6),dtype=D)
        self.ref,self.Q,self.R,self.P=lqr_cost(self.nom);b,u=sagittal_basis(self.nom);self.project=np.linalg.pinv(b);self.input_project=np.linalg.pinv(u)
        self.nq=self.nom.nq;self.nv=self.nom.nv;self.width=self.nq+self.nv+12
        self.trace=wp.zeros((10,self.count,self.width),dtype=D)
        with wp.ScopedCapture() as initialize:
            self.control();wp.launch(apply_extra,self.count,[self.data.qvel,self.ids,self.upper,self.extra,self.data.ctrl])
            mjw.forward(self.model,self.data);wp.copy(self.data.qacc_warmstart,self.data.qacc)
        self.initialize_graph=initialize.graph
        with wp.ScopedCapture() as capture:
            for slot in range(10):
                self.control();wp.launch(apply_extra,self.count,[self.data.qvel,self.ids,self.upper,self.extra,self.data.ctrl]);mjw.step(self.model,self.data)
                wp.launch(record_forecast,self.count,[slot,self.data.qpos,self.data.qvel,self.data.ctrl,self.data.actuator_force,self.trace])
        self.graph=capture.graph

    def control(self):
        wp.launch(control_physical,self.count,[self.data.qpos,self.data.qvel,self.data.sensordata,self.targets,self.command,self.active,self.state,self.ids,
            self.fixed['heights'],self.fixed['gains'],self.fixed['feed'],self.fixed['angles'],self.reference,self.fixed['yaw'],self.data.ctrl,self.diag,0,0,self.upper],block_dim=32)

    def candidate_changes(self,q,v,current=None):
        if self.arms==1:
            if current is None:return np.zeros((self.n,6))
            if current.shape!=(self.n,6) or not np.isfinite(current).all() or np.any(abs(current)>1.):raise ValueError('保留动作须为有限的原1Nm盒内批量')
            return current.copy()
        changes=[]
        for i in range(self.n):
            b=basis(self.nom,q[i],v[i]);b=b/np.sum(abs(b),axis=0)*.1
            dirs=[b[:,j] for j in range(3)]+[(b[:,0]+b[:,1])/2,(b[:,1]+b[:,2])/2,(b[:,0]+b[:,2])/2]
            if self.arms==19:dirs += [(b[:,0]-b[:,1])/2,(b[:,1]-b[:,2])/2,(b[:,0]-b[:,2])/2]
            changes.append(np.array([np.zeros(6)]+[signed for d in dirs for signed in (d,-d)]))
        changes=np.concatenate(changes);assert np.max(np.sum(abs(changes),axis=1))<=.10000000001
        if current is not None:
            changes=np.clip(changes+np.repeat(current,self.arms,axis=0),-1.,1.)
            assert np.max(abs(changes))<=1. and np.max(np.sum(abs(changes-np.repeat(current,self.arms,axis=0)),axis=1))<=.10000000001
        return changes

    def choose(self,env,stopped,task_state,current=None):
        timing_start=perf_counter()
        q=env.data.qpos.numpy();v=env.data.qvel.numpy();memory=env.k['state'].numpy();sensor=env.data.sensordata.numpy()
        timing_read=perf_counter();changes=self.candidate_changes(q,v,current)
        self.extra.assign(changes);self.data.qpos.assign(np.repeat(q,self.arms,axis=0));self.data.qvel.assign(np.repeat(v,self.arms,axis=0))
        self.data.sensordata.assign(np.repeat(sensor,self.arms,axis=0));self.data.qacc_warmstart.zero_();self.state.assign(np.repeat(memory,self.arms,axis=0))
        wp.synchronize_device();timing_upload=perf_counter()
        # Calculate current Nom using known past sensor/filter state. Own forward
        # solve initializes numerical warmstart without advancing q/v.
        wp.capture_launch(self.initialize_graph)
        # Restore BEFORE-current-control memory/sensors; rollout computes it once.
        self.state.assign(np.repeat(memory,self.arms,axis=0));self.data.sensordata.assign(np.repeat(sensor,self.arms,axis=0))
        wp.synchronize_device();timing_initialize=perf_counter()
        wp.capture_launch(self.graph);wp.synchronize_device();timing_gpu=perf_counter()
        trace=self.trace.numpy();force=trace[:,:,-6:]
        timing_forecast=perf_counter()
        all_q=trace[:,:,:self.nq];all_v=trace[:,:,self.nq:self.nq+self.nv];all_cmd=trace[:,:,-12:-6]
        all_cost,all_valid=batch_scores(self.nom,all_q,all_v,all_cmd,force,np.repeat(q,self.arms,axis=0),np.repeat(v,self.arms,axis=0),np.repeat(memory,self.arms,axis=0),self.ref,self.Q,self.R,self.P,self.project,self.input_project)
        task=task_forecast(all_q,all_v,np.repeat(q,self.arms,axis=0),np.repeat(v,self.arms,axis=0),np.repeat(task_state,self.arms,axis=0))
        out=np.zeros((self.n,6));rows=[]
        for i in np.flatnonzero(stopped):
            sl=slice(i*self.arms,(i+1)*self.arms);values=all_cost[sl];physical=all_valid[sl];valid=physical&task['valid'][sl]
            choices=np.flatnonzero(valid)
            if not len(choices):return None,dict(world=int(i),reason='no_task_valid_candidate' if physical.any() else 'no_physical_valid_candidate',
                values=values.tolist(),physical_valid_candidates=int(physical.sum()),task_state=task_state[i,:8].tolist(),
                task={key:value[sl].tolist() for key,value in task.items()})
            arm=int(choices[np.argmin(np.array(values)[choices])]);out[i]=changes[i*self.arms+arm]
            rows.append(dict(world=int(i),arm=arm,valid_candidates=len(choices),physical_valid_candidates=int(physical.sum()),
                distinct_candidates=len(np.unique(changes[sl],axis=0)),cost=float(values[arm]),zero_cost=float(values[0]),
                task={key:float(value[i*self.arms+arm]) for key,value in task.items() if key!='valid'}))
        timing_end=perf_counter()
        self.last_timing=dict(read_inputs_ms=(timing_read-timing_start)*1000,candidates_and_upload_ms=(timing_upload-timing_read)*1000,
            own_initialization_ms=(timing_initialize-timing_upload)*1000,rollout_and_download_ms=(timing_forecast-timing_initialize)*1000,
            rollout_sync_ms=(timing_gpu-timing_initialize)*1000,download_ms=(timing_forecast-timing_gpu)*1000,
            checks_and_cost_ms=(timing_end-timing_forecast)*1000)
        return out,rows


def run(output,incremental=False,current_vmc=False,mixed_signs=False):
    output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    scenes=cases()[4:6];env=NativeEnv(n=2,scenario=scenes,bank_factory=bank_height_115,height_conditioned=True,height_design='range115',
        residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=True)
    try:
        if current_vmc:
            table,_=current_vmc_table();env.k['gains'].assign(np.stack([t[0] for t in table]));env.k['feed'].assign(np.stack([t[1] for t in table]));env.k['angles'].assign(np.array([t[2] for t in table]))
        env.reset();predictor=Forecaster(env,mixed_signs);extra=wp.zeros((2,6),dtype=D);graph=execution_graph(env,extra)
        rows=[];durations=[];failure=None;current=np.zeros((2,6))
        for iteration in range(1600):
            state=env.state.numpy();stopped=(state[:,1]>=0)&(env.active.numpy()!=0)
            if np.any(stopped):
                before=perf_counter();chosen,decision=predictor.choose(env,stopped,state,current if incremental else None);durations.append((perf_counter()-before)*1000)
                if chosen is None:failure=dict(iteration=iteration,**decision);break
                extra.assign(chosen);current=chosen;rows.append(dict(iteration=iteration,decisions=decision,extra=chosen.tolist(),timing=predictor.last_timing))
            else:extra.zero_()
            wp.capture_launch(graph)
            if np.all(env.done.numpy()!=0):break
            if iteration%200==0:print('PROGRESS',iteration,'planning',len(rows),flush=True)
        completed=bool(np.all(env.done.numpy()!=0));infos=[]
        if completed:infos=[{k:v for k,v in r.items() if k!='terminal_observation'} for r in env.step_wait()[3]]
        result=dict(role='independent_5ms_finite_candidate_feedback_highspeed_pilot',completed=completed,prediction_failure=failure,
            episodes=infos,final_task_state=env.state.numpy()[:,:8].tolist(),decisions=rows,decision_ms=dict(median=float(np.median(durations)),p95=float(np.percentile(durations,95)),maximum=float(max(durations))) if durations else None,
            incremental=incremental,current_vmc=current_vmc,mixed_signs=mixed_signs,candidate_count=predictor.arms,predictor_geom_count=predictor.nom.ngeom,extra_L1_increment_limit_Nm=.1,absolute_per_motor_extra_limit_Nm=1. if incremental else .1,feedback_interval_ms=5.,forecast_horizon_ms=5.,original_cost_and_gates_unchanged=True,
            task_contract_screened=True,terminal_task_feasibility_verified=False,
            limitations=f'Fixed{predictor.arms}-candidate greedy feedback, two nominal highspeed cases only. Original arrival-distance and tail-speed history/deadline gates are screened over5ms, with remaining unverified time recorded; no feasible terminal/backup guarantee. Predictor reads current full simulator state and known internal memory; not sensor-only or real-time. No default, CPU or PPO controller changes. Incremental mode uses0.1Nm L1 changes inside existing1Nm absolute motor-extra box; new visited states still need independent model support.',
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/select_braking_common_action.py',ROOT/'wheelleg_warp/probe_braking_phase_chart.py',ROOT/'wheelleg_warp/probe_current_vmc_design.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/training_contract.py')})
        (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('COMPLETED',completed,'success',sum(r['success'] for r in infos),'failure',failure,flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--incremental',action='store_true');parser.add_argument('--current-vmc',action='store_true');parser.add_argument('--mixed-signs',action='store_true')
    args=parser.parse_args();run(args.output,args.incremental,args.current_vmc,args.mixed_signs)
