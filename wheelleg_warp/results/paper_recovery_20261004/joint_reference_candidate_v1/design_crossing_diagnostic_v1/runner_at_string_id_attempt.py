"""One registered20-episode forensic queue, keeping learned models frozen."""
import json
import time
import numpy as np
import torch
import mujoco
import warp as wp
from stable_baselines3 import PPO
from run_joint_reference_zero_pair import evaluate
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
from smoke_reward_training import weight_digest

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/design_crossing_diagnostic_v1'


class FrozenActor:
    def __init__(self,model,deterministic):
        self.model=model;self.policy=model.policy;self.deterministic=deterministic
        self.observation_space=model.observation_space;self.action_space=model.action_space
    def predict(self,obs,deterministic=True):return self.model.predict(obs,deterministic=self.deterministic)


def point(data,index,addresses,row,directory):
    t,pre,post,contacts,steps,columns=data;nq=17;nv=16
    qids,vids=addresses;q=post[index,qids];v=post[index,nq+vids]
    names=('alphaL','betaL','alphaR','betaR');joint=int(np.argmax(abs(q[:4])))
    result=dict(index=int(index),physical_step=int(t[index,1]),pre_time_s=float(t[index,2]),post_time_s=float(t[index,3]),
        active_q_rad=q[:4].tolist(),active_velocity_rad_s=v[:4].tolist(),largest_active_joint=names[joint],
        design_margin_rad=float(t[index,columns['design_margin']]),pre_ctrl_Nm=pre[index,nq+nv:].tolist(),post_force_Nm=post[index,nq+nv:].tolist(),
        current_command=float(t[index,columns['speed_command']]),body_attitude_rad=t[index,[columns[k] for k in ('roll','pitch','yaw')]].tolist(),
        pre_wheel_normal_N=t[index,[columns[k] for k in ('left_normal_N','right_normal_N')]].tolist(),
        physical_step_contacts=contacts[steps==index+1].tolist(),original_lambda=float(t[index,columns['original_lambda']]),
        contact_timing='archived physical-step contact/force;postq andrecordedcontact geometry haveintegration timing difference,notunique causalproof')
    if 'joint_reference_trace' in row:
        with np.load(directory/row['joint_reference_trace']['path']) as z:
            ref=z['trace'][index];result['reference']=dict(nominal_height=float(ref[2]),tracking_mean=float(ref[3]),half_difference=float(ref[4]),
                targets=ref[5:7].tolist(),radial_anchor=float(ref[7]),requested=ref[10:16].tolist(),effective=ref[16:22].tolist(),filtered_reference=ref[22:24].tolist(),filtered_motor=ref[24:30].tolist())
    return result


def analyze(directory,rows,addresses):
    records=[];qids,_=addresses
    for row in rows:
        f=directory/row['complete_trace']['path'];assert sha(f)==row['complete_trace']['sha256']
        with np.load(f) as z:
            t,pre,post,contacts,steps=[z[k].copy() for k in ('trace','pre','post','contacts','contact_steps')]
            assert z['state_sizes'].tolist()==[17,16,6];columns={str(k):i for i,k in enumerate(z['columns'])}
        margin=1.4-abs(post[:,qids[:4]]).max(axis=1)
        np.testing.assert_array_equal(margin,t[:,columns['design_margin']])
        np.testing.assert_allclose(margin.min(),row['min_active_design_margin_rad'],rtol=0,atol=1e-15)
        cross=np.flatnonzero(margin<0);first=None
        data=(t,pre,post,contacts,steps,columns)
        if len(cross):first=point(data,int(cross[0]),addresses,row,directory)
        records.append(dict(case=row['seed'],physical=row['physical_safety_passed'],design=row['design_joint_passed'],success=row['success'],
            first_crossing=first,minimum_point=point(data,int(margin.argmin()),addresses,row,directory),dense_sha256=sha(f)))
    return records


def run():
    q=json.loads((OUT/'proposal.json').read_text());a=json.loads((OUT/'source_admission.json').read_text())
    assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json')
    assert not any((OUT/f).exists() for f in ('started.json','completion.json','failure.json'))
    assert q['episodes']==20 and len(q['cases'])==4 and len(q['conditions'])==5
    assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    torch.set_num_threads(1);model=PPO.load(ROOT/q['model'],device='cuda')
    initial=weight_digest(model);x=np.zeros((4,481),np.float32)
    det=FrozenActor(model,True);random=FrozenActor(model,False)
    np.testing.assert_array_equal(det.predict(x)[0],model.predict(x,deterministic=True)[0])
    torch.manual_seed(q['stochastic_action_seed']);u=random.predict(x)[0]
    torch.manual_seed(q['stochastic_action_seed']);np.testing.assert_array_equal(u,model.predict(x,deterministic=False)[0])
    assert np.isfinite(u).all() and np.max(abs(u))<=1 and weight_digest(model)==initial
    atomic_json(OUT/'unit.json',dict(verified=True,frozen_actor_mode_and_rng_delegation=True,policy_unchanged=True,physics_graph_launches=0))
    cpu=mujoco.MjModel.from_xml_path(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'))
    joint_names=('alphaL','betaL','alphaR','betaR','wheel1','wheel2')
    addresses=tuple(np.array([getattr(cpu,attr)[cpu.joint(name).id] for name in joint_names]) for attr in ('jnt_qposadr','jnt_dofadr'))
    assert cpu.nq==17 and cpu.nv==16
    calls=[];steps=0;worlds=4;fd=mujoco.mjd_transitionFD;graph=wp.capture_launch;reports={};records=[];current=None;start=time.perf_counter();world=None
    def counted(*args,**kwargs):
        calls.append(float(args[2]));assert len(calls)<=q['constructor_FD_budget'];return fd(*args,**kwargs)
    def capture(*args,**kwargs):
        nonlocal steps
        steps+=40*worlds;assert steps<=q['world_step_budget'];return graph(*args,**kwargs)
    mujoco.mjd_transitionFD=counted;wp.capture_launch=capture
    atomic_json(OUT/'started.json',dict(proposal_sha256=sha(OUT/'proposal.json'),source_admission_sha256=sha(OUT/'source_admission.json')))
    try:
        for condition in q['conditions']:
            assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items());current=condition['label']
            directory=OUT/current;directory.mkdir(exist_ok=False);torch.manual_seed(q['stochastic_action_seed'])
            torch.save(dict(RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),directory/'initial_RNG.pt')
            actor=FrozenActor(model,condition['deterministic']) if condition['learned'] else None
            before=steps;result,checked=evaluate(q['cases'],condition['arm'],directory,q['world_step_budget']-steps,q['yaw_config'],model=actor,normalization=ROOT/q['normalization'] if actor else None)
            assert steps-before==result['actual_graph_world_steps'] and weight_digest(model)==initial
            with np.load(directory/'initial.npz') as z:initials={k:z[k].copy() for k in z.files}
            if world is None:world=initials
            else:
                for k in initials:np.testing.assert_array_equal(initials[k],world[k])
            report=analyze(directory,result['runs'],addresses);reports[current]=report
            atomic_json(directory/'crossings.json',dict(records=report,addresses=[v.tolist() for v in addresses],joint_names=joint_names))
            records.append(dict(condition=current,episodes=4,design=sum(r['design'] for r in report),physical=sum(r['physical'] for r in report),
                success=sum(r['success'] for r in report),actual_graph_world_steps=result['actual_graph_world_steps'],first_episode_world_steps=result['first_episode_world_steps'],result_sha256=sha(directory/'result.json'),checked=checked))
            atomic_json(OUT/'progress.json',dict(records=records,actual_graph_world_steps=steps))
        assert len(records)==5 and sum(r['episodes'] for r in records)==20
        baseline_fail=any(not r['design'] for k in ('old_B0','joint_zero','joint_yaw') for r in reports[k])
        learned_fail=any(not r['design'] for k in ('frozen_M_det','frozen_M_stochastic') for r in reports[k])
        decision='baseline_design_boundary_observed' if baseline_fail else ('frozen_learned_boundary_observed_only' if learned_fail else 'original_nonstationary_failure_not_reproduced')
        atomic_json(OUT/'completion.json',dict(verified=True,records=records,decision=decision,actual_graph_world_steps=steps,first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),
            constructor_FD_calls=calls,wall_seconds=time.perf_counter()-start,learning_samples=0,original_qualification_remains_failed=True,formal5_admitted=False))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),condition=current,records=records,actual_graph_world_steps=steps,FD_calls=calls,implicit_retry=False));raise
    finally:mujoco.mjd_transitionFD=fd;wp.capture_launch=graph


if __name__=='__main__':run()
