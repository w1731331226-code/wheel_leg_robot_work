"""Offline full-episode scoring of one shared plan across registered plants.

No future predictor or plant-state checkpoint: every evaluation resets and runs
the actual candidate environment from standing through the original deadline.
"""
from dataclasses import asdict, replace
from pathlib import Path
import json
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from native.controller import D,fk
from native.environment import NativeEnv,begin,command_step,reduce_contacts,collect_physical,after
from native.terrain import model,HeightTerrainScenario
from probe_height_115_margin import cases
from select_braking_common_action import lqr_cost,batch_scores,batch_constraint_margins,task_forecast
from model_lqr import sagittal_basis
from optimize_braking_trajectory import schedules,KNOTS
from probe_height_115_action_predict_loow import sha

OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'braking_trajectory_slsqp_20261001'
VARIATIONS=[('nominal',{}),('mass_only',dict(mass=7.5)),('friction_only',dict(mu_l=.6,mu_r=1.)),
    ('drive_only',dict(drive_difference=.03)),('training_corner',dict(mass=7.5,mu_l=.6,mu_r=1.,drive_difference=.03,delay_ms=10.))]
SCALE=np.r_[np.array([1.4,1.4,2.5,2.5]*2),np.full(4,.115),np.tile(200*np.array([1.4,1.4,2.5,2.5]*2),2),
    np.full(3,np.deg2rad(5)),np.tile([40,40,40,40,4.5,4.5],2),.6,.03,.02,.02,1.]


@wp.kernel
def update_plan(arms:int,plan:wp.array3d[D],state:wp.array2d[D],active:wp.array[int],q:wp.array2d[float],v:wp.array2d[float],
                memory:wp.array2d[D],extra:wp.array2d[D],starts:wp.array[int],iq:wp.array2d[D],iv:wp.array2d[D],
                im:wp.array2d[D],initial:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0 or state[w,1]<D(0):return
    if starts[w]<0:
        starts[w]=int(state[w,0])
        for j in range(q.shape[1]):iq[w,j]=D(q[w,j])
        for j in range(v.shape[1]):iv[w,j]=D(v[w,j])
        for j in range(memory.shape[1]):im[w,j]=memory[w,j]
        for j in range(state.shape[1]):initial[w,j]=state[w,j]
    slot=wp.min((int(state[w,0])-starts[w])/10,399)
    for j in range(6):extra[w,j]=plan[slot,w%arms,j]


@wp.kernel
def record(q:wp.array2d[float],v:wp.array2d[float],ctrl:wp.array2d[float],force:wp.array2d[float],
           ids:wp.array[int],flags:wp.array2d[int],state:wp.array2d[D],active:wp.array[int],starts:wp.array[int],
           lengths:wp.array[int],trace:wp.array3d[D]):
    w=wp.tid()
    if active[w]==0 or starts[w]<0:return
    slot=int(state[w,0])-starts[w];nq=q.shape[1];nv=v.shape[1]
    if slot>=4000:return
    for j in range(nq):trace[slot,w,j]=D(q[w,j])
    for j in range(nv):trace[slot,w,nq+j]=D(v[w,j])
    for j in range(6):
        trace[slot,w,nq+nv+j]=D(ctrl[w,j]);trace[slot,w,nq+nv+6+j]=D(force[w,j])
    trace[slot,w,nq+nv+12]=(fk(D(q[w,ids[0]]),D(q[w,ids[1]]))[3]+fk(D(q[w,ids[2]]),D(q[w,ids[3]]))[3])/D(2)
    trace[slot,w,nq+nv+13]=D(flags[w,0])
    lengths[w]=slot+1


class Episodes:
    # ponytail: actual episodes are rerun from reset; cache prefix physics only
    # after a measured bottleneck and a verified full-state restore exist.
    def __init__(self,direction,arms=1):
        self.direction=direction;self.arms=arms
        base=cases()[4+direction]
        self.scenes=[replace(base,**change) for _,change in VARIATIONS for _ in range(arms)]
        self.env=env=NativeEnv.height115_candidate(n=len(self.scenes),scenario=self.scenes,residual_scale=0,nominal_correction=True)
        n=len(self.scenes);d=env.data;self.nq=env.cpu.nq;self.nv=env.cpu.nv
        self.plan=wp.zeros((400,arms,6),dtype=D);self.starts=wp.full(n,-1,dtype=int);self.lengths=wp.zeros(n,dtype=int)
        self.iq=wp.zeros((n,self.nq),dtype=D);self.iv=wp.zeros((n,self.nv),dtype=D)
        self.im=wp.zeros(env.k['state'].shape,dtype=D);self.initial=wp.zeros(env.state.shape,dtype=D)
        self.trace=wp.zeros((4000,n,self.nq+self.nv+14),dtype=D)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,n,[env.reward])
            for slot in range(40):
                if slot%10==0:
                    wp.launch(update_plan,n,[arms,self.plan,env.state,env.active,d.qpos,d.qvel,env.k['state'],env.nominal_correction,
                        self.starts,self.iq,self.iv,self.im,self.initial])
                wp.launch(command_step,n,[env.state,env.param,env.command,env.active,d.qpos,d.qvel,d.qacc_warmstart,env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
                wp.launch(env.control_kernel,n,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
                    env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
                mjw.step(env.model,d)
                wp.launch(reduce_contacts,d.naconmax,[d.nacon,d.contact.worldid,d.contact.geom,env.ids,env.contact_flags])
                wp.launch(collect_physical,n,env.physical_args)
                wp.launch(record,n,[d.qpos,d.qvel,d.ctrl,d.actuator_force,env.ids,env.contact_flags,env.state,env.active,self.starts,self.lengths,self.trace])
                wp.launch(after,n,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,
                    env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        self.graph=capture.graph
        self.nom=model(HeightTerrainScenario(stand_height_m=.115))
        self.ref,self.Q,self.R,self.P=lqr_cost(self.nom)
        b,u=sagittal_basis(self.nom);self.project=np.linalg.pinv(b);self.input_project=np.linalg.pinv(u)

    def close(self):self.env.close()

    def evaluate(self,plan,save=None):
        assert plan.shape==(400,self.arms,6) and np.isfinite(plan).all() and abs(plan).max()<=1.+1e-12
        assert abs(np.diff(np.concatenate([np.zeros_like(plan[:1]),plan]),axis=0)).sum(axis=2).max()<=.1+1e-12
        started=perf_counter();env=self.env;n=len(self.scenes)
        env.reset();self.starts.fill_(-1);self.lengths.zero_();self.trace.zero_();self.plan.assign(plan)
        for step in range(700):
            wp.capture_launch(self.graph)
            if step%10==0 and env.done.numpy().all():break
        assert env.done.numpy().all() and not env.data.overflow.numpy().any()
        states=env.state.numpy();lengths=self.lengths.numpy();start_steps=self.starts.numpy()
        assert np.all(start_steps>=0) and np.all((lengths>0)&(lengths<=4000))
        assert np.array_equal(lengths,states[:,0].astype(int)-start_steps)
        initial=self.initial.numpy();iq=self.iq.numpy();iv=self.iv.numpy();im=self.im.numpy();trace=self.trace.numpy()
        costs=np.zeros(n);margins=np.zeros((n,48))
        for count in np.unique(lengths):
            ix=np.flatnonzero(lengths==count);a=trace[:count,ix,:-2];task_data=trace[:count,ix,-2:]
            q=a[:,:,:self.nq];v=a[:,:,self.nq:self.nq+self.nv];commands=a[:,:,-12:-6];force=a[:,:,-6:]
            cost,_=batch_scores(self.nom,q,v,commands,force,iq[ix],iv[ix],im[ix],self.ref,self.Q,self.R,self.P,self.project,self.input_project)
            physical=batch_constraint_margins(self.nom,q,v,commands,force,iv[ix])
            task=task_forecast(q,v,iq[ix],iv[ix],initial[ix])
            rmse=np.sqrt((initial[ix,28]+np.sum((task_data[:,:,0]-.115)**2,axis=0)*.0005)/(initial[ix,29]+count*.0005))
            height_error=abs(task_data[-1,:,0]-.115)
            margins[ix]=np.c_[physical,.6-task['peak_distance_m'],.03-task['peak_tail_speed_m_s'],.02-rmse,.02-height_error,-task_data[:,:,1].max(axis=0)]
            costs[ix]=cost
            np.testing.assert_allclose(task['peak_distance_m'],states[ix,6],rtol=0,atol=1e-12)
            np.testing.assert_allclose(task['peak_tail_speed_m_s'],states[ix,7],rtol=0,atol=1e-12)
            np.testing.assert_allclose(rmse,np.sqrt(states[ix,28]/states[ix,29]),rtol=0,atol=1e-12)
        rows=[{k:v for k,v in row.items() if k!='terminal_observation'} for row in env.step_wait()[3]]
        assert all(r['physical_steps']==r['physical_evidence_steps'] and r['reason']=='completed' for r in rows)
        valid=(margins>=0).all(axis=1)&np.array([r['success'] for r in rows])
        result=dict(direction=self.direction,arms=self.arms,episodes=rows,scenarios=[asdict(s) for s in self.scenes],
            costs=costs.tolist(),margins=margins.tolist(),valid=valid.tolist(),wall_s=perf_counter()-started,
            start_steps=start_steps.tolist(),braking_steps=lengths.tolist(),learning=False,full_admission=False)
        if save is not None:
            assert not save.with_suffix('.json').exists()
            np.savez_compressed(save.with_suffix('.npz'),trace=trace,lengths=lengths,initial_q=iq,initial_v=iv,initial_controller=im,initial_task=initial,schedule=plan)
            result['trace_sha256']=sha(save.with_suffix('.npz'))
            result['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/select_braking_common_action.py')}
            save.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
        return result


def preflight():
    assert not (OUT/'preflight.json').exists()
    registration=dict(variations=VARIATIONS,knots=KNOTS.tolist(),cost='original fixed7kg fifteen-state Q/R/P, nominal arm objective',
        constraints='original48 full-horizon constraints and full episode success for every registered plant',
        selection='same plan per speed sign for all physical parameters; no parameter label in control',
        optimizer_budget=dict(maxiter=40,max_evaluations=120,finite_difference=.01),
        scope='Offline full simulation from standing, no plant-state prefix restore and no online predictor; no final holdout or learning.')
    (OUT/'registered_protocol.json').write_text(json.dumps(registration,indent=2)+'\n')
    records=[]
    for direction in range(2):
        z=np.load(SOURCE/f'world{direction}_best.npz',allow_pickle=False)
        plan=schedules(z['parameters'][None,:],z['common'],400)
        np.testing.assert_allclose(plan[:,0],z['schedule'],rtol=0,atol=1e-15)
        evaluator=Episodes(direction)
        try:
            result=evaluator.evaluate(plan,OUT/f'preflight_world{direction}')
            records.append(dict(direction=direction,valid=result['valid'],wall_s=result['wall_s']))
            print('PREFLIGHT',records[-1],flush=True)
        finally:evaluator.close()
    (OUT/'preflight.json').write_text(json.dumps(dict(records=records,host_gpu_task_metrics_agree=True,optimization_started=False),indent=2)+'\n')


if __name__=='__main__':preflight()
