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
from probe_height_115_local_states import snapshot
from probe_braking_nominal_rollout import apply_extra
from probe_braking_phase_chart import physical_metrics
from select_braking_common_action import lqr_cost,trajectory_cost,constraint_rejections
from model_lqr import sagittal_basis
from probe_height_115_action_predict_loow import sha


@wp.kernel
def record_force(slot:int,f:wp.array2d[float],out:wp.array3d[D]):
    w=wp.tid()
    for j in range(6):out[slot,w,j]=D(f[w,j])


@wp.kernel
def execute_extra(v:wp.array2d[float],ids:wp.array[int],upper:wp.array2d[D],extra:wp.array2d[D],active:wp.array[int],ctrl:wp.array2d[float],diag:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    speed=V6();gain=V6()
    for j in range(6):speed[j]=D(v[w,ids[4+j]]);gain[j]=upper[w,j]
    bound=command_bounds(speed,gain)
    for j in range(6):
        old=D(ctrl[w,j]);new=wp.clamp(old+extra[w,j],-bound[j],bound[j]);ctrl[w,j]=float(new);diag[w,6+j]=new-old


class Forecaster:
    # ponytail: thirteen fixed candidates; continuous optimization only if this
    # verified grid is the limiting factor. This is not a real-time controller.
    def __init__(self,env):
        self.n=env.num_envs;self.count=self.n*13;self.nom=model(HeightTerrainScenario(stand_height_m=.115))
        _,self.model,self.data,_=batch([self.nom]*self.count,[HeightTerrainScenario(stand_height_m=.115)]*self.count)
        self.ids=wp.array(env.ids.numpy(),dtype=int);self.upper=wp.array(np.tile(env.actuator_gain_upper,(self.count,1)),dtype=D)
        self.state=wp.zeros((self.count,env.k['state'].shape[1]),dtype=D);self.reference=wp.array(np.repeat(env.k['reference'].numpy(),13,axis=0),dtype=D)
        self.fixed={k:wp.array(env.k[k].numpy(),dtype=D) for k in ('heights','gains','feed','angles','yaw')}
        self.targets=wp.zeros((self.count,3));self.command=wp.zeros(self.count,dtype=D);self.active=wp.ones(self.count,dtype=int)
        self.diag=wp.zeros((self.count,38),dtype=D);self.extra=wp.zeros((self.count,6),dtype=D)
        self.ref,self.Q,self.R,self.P=lqr_cost(self.nom);b,u=sagittal_basis(self.nom);self.project=np.linalg.pinv(b);self.input_project=np.linalg.pinv(u)
        self.nq=self.nom.nq;self.nv=self.nom.nv;self.width=1+self.nq+2*self.nv+6
        self.trace=wp.zeros((10,self.count,self.width),dtype=D);self.force=wp.zeros((10,self.count,6),dtype=D)
        with wp.ScopedCapture() as capture:
            for slot in range(10):
                self.control();wp.launch(apply_extra,self.count,[self.data.qvel,self.ids,self.upper,self.extra,self.data.ctrl]);mjw.step(self.model,self.data)
                wp.launch(snapshot,self.count,[slot,self.data.time,self.data.qpos,self.data.qvel,self.data.qacc_warmstart,self.data.ctrl,self.trace])
                wp.launch(record_force,self.count,[slot,self.data.actuator_force,self.force])
        self.graph=capture.graph

    def control(self):
        wp.launch(control_physical,self.count,[self.data.qpos,self.data.qvel,self.data.sensordata,self.targets,self.command,self.active,self.state,self.ids,
            self.fixed['heights'],self.fixed['gains'],self.fixed['feed'],self.fixed['angles'],self.reference,self.fixed['yaw'],self.data.ctrl,self.diag,0,0,self.upper],block_dim=32)

    def choose(self,env,stopped,current=None):
        q=env.data.qpos.numpy();v=env.data.qvel.numpy();memory=env.k['state'].numpy();sensor=env.data.sensordata.numpy()
        changes=[]
        for i in range(self.n):
            b=basis(self.nom,q[i],v[i]);b=b/np.sum(abs(b),axis=0)*.1
            dirs=[b[:,j] for j in range(3)]+[(b[:,0]+b[:,1])/2,(b[:,1]+b[:,2])/2,(b[:,0]+b[:,2])/2]
            changes.append(np.array([np.zeros(6)]+[signed for d in dirs for signed in (d,-d)]))
        changes=np.concatenate(changes);assert np.max(np.sum(abs(changes),axis=1))<=.10000000001
        if current is not None:
            increments=changes.copy();changes=np.clip(changes+np.repeat(current,13,axis=0),-1.,1.)
            assert np.max(abs(changes))<=1. and np.max(np.sum(abs(changes-np.repeat(current,13,axis=0)),axis=1))<=.10000000001
        self.extra.assign(changes);self.data.qpos.assign(np.repeat(q,13,axis=0));self.data.qvel.assign(np.repeat(v,13,axis=0))
        self.data.sensordata.assign(np.repeat(sensor,13,axis=0));self.data.qacc_warmstart.zero_();self.state.assign(np.repeat(memory,13,axis=0))
        # Calculate current Nom using known past sensor/filter state. Own forward
        # solve initializes numerical warmstart without advancing q/v.
        self.control();wp.launch(apply_extra,self.count,[self.data.qvel,self.ids,self.upper,self.extra,self.data.ctrl])
        mjw.forward(self.model,self.data);wp.copy(self.data.qacc_warmstart,self.data.qacc)
        # Restore BEFORE-current-control memory/sensors; rollout computes it once.
        self.state.assign(np.repeat(memory,13,axis=0));self.data.sensordata.assign(np.repeat(sensor,13,axis=0))
        wp.capture_launch(self.graph);trace=self.trace.numpy();force=self.force.numpy()
        out=np.zeros((self.n,6));rows=[]
        for i in np.flatnonzero(stopped):
            values=[];valid=[]
            for arm in range(13):
                index=i*13+arm;p=trace[:,index,1:1+self.nq];vel=trace[:,index,1+self.nq:1+self.nq+self.nv];cmd=trace[:,index,-6:]
                reasons=constraint_rejections(self.nom,p,vel);prior=np.r_[v[i:i+1],vel[:-1]]
                if physical_metrics(self.nom,p,vel,force[:,index],prior,cmd)['safe_arms']!=10:reasons.append('physical_gate')
                values.append(trajectory_cost(self.nom,p,vel,cmd,q[i],v[i],memory[i],self.ref,self.Q,self.R,self.P,self.project,self.input_project));valid.append(not reasons)
            choices=np.flatnonzero(valid)
            if not len(choices):return None,dict(world=int(i),reason='no_valid_candidate',values=values)
            arm=int(choices[np.argmin(np.array(values)[choices])]);out[i]=changes[i*13+arm]
            rows.append(dict(world=int(i),arm=arm,valid_candidates=len(choices),cost=values[arm],zero_cost=values[0]))
        return out,rows


def run(output,incremental=False):
    output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    scenes=cases()[4:6];env=NativeEnv(n=2,scenario=scenes,bank_factory=bank_height_115,height_conditioned=True,height_design='range115',
        residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=True)
    try:
        env.reset();predictor=Forecaster(env);extra=wp.zeros((2,6),dtype=D);upper=wp.array(np.tile(env.actuator_gain_upper,(2,1)),dtype=D)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,2,[env.reward])
            for _ in range(10):
                d=env.data;wp.launch(command_step,2,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
                wp.launch(env.control_kernel,2,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
                    env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
                wp.launch(execute_extra,2,[d.qvel,env.ids,upper,extra,env.active,d.ctrl,env.diag]);mjw.step(env.model,d)
                wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags]);wp.launch(collect_physical,2,env.physical_args)
                wp.launch(after,2,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,
                    env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        rows=[];durations=[];failure=None;current=np.zeros((2,6))
        for iteration in range(1600):
            state=env.state.numpy();stopped=(state[:,1]>=0)&(env.active.numpy()!=0)
            if np.any(stopped):
                before=perf_counter();chosen,decision=predictor.choose(env,stopped,current if incremental else None);durations.append((perf_counter()-before)*1000)
                if chosen is None:failure=dict(iteration=iteration,**decision);break
                extra.assign(chosen);current=chosen;rows.append(dict(iteration=iteration,decisions=decision,extra=chosen.tolist()))
            else:extra.zero_()
            wp.capture_launch(capture.graph)
            if np.all(env.done.numpy()!=0):break
            if iteration%200==0:print('PROGRESS',iteration,'planning',len(rows),flush=True)
        completed=bool(np.all(env.done.numpy()!=0));infos=[]
        if completed:infos=[{k:v for k,v in r.items() if k!='terminal_observation'} for r in env.step_wait()[3]]
        result=dict(role='independent_5ms_finite_candidate_feedback_highspeed_pilot',completed=completed,prediction_failure=failure,
            episodes=infos,decisions=rows,decision_ms=dict(median=float(np.median(durations)),p95=float(np.percentile(durations,95)),maximum=float(max(durations))) if durations else None,
            incremental=incremental,extra_L1_increment_limit_Nm=.1,absolute_per_motor_extra_limit_Nm=1. if incremental else .1,feedback_interval_ms=5.,forecast_horizon_ms=5.,original_cost_and_gates_unchanged=True,
            limitations='Fixed13-candidate greedy feedback, two nominal highspeed cases only. Predictor reads current full simulator state and known internal memory; not sensor-only or real-time. No default, CPU or PPO changes. Incremental mode uses0.1Nm L1 changes inside existing1Nm absolute motor-extra box; new visited states still need independent model support.',
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/select_braking_common_action.py',ROOT/'wheelleg_warp/probe_braking_phase_chart.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
        (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('COMPLETED',completed,'success',sum(r['success'] for r in infos),'failure',failure,flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--incremental',action='store_true')
    args=parser.parse_args();run(args.output,args.incremental)
