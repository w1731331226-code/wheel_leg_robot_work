"""平地三例首越线前20ms：六电机恒定力矩增量的局部线性可行性诊断。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
from scipy.optimize import linprog
import wheelleg_sim as sim
from native.terrain import HEIGHT_115_GEOMETRIC_MIN,HeightTerrainScenario,model
from probe_height_115_passive import JOINTS,geometry
from probe_height_115_cpu_step_pair import forces_cpu

DT=.0005
HORIZON=40
EPS=.1
GAMMA=.001  # 仅覆盖已测CPU/Warp 20ms配对数值误差，不是硬件安全裕量。
EVENTS=(1,9,10)  # 平地 +0.5/+1/−1 m/s 各自最早的几何或关节门。


def metric(model,data,joint_ids,joint_range):
    actual=geometry(model,data.qpos)[1]
    angles=data.qpos[joint_ids]
    pose=np.asarray(sim.euler(data))
    raw=np.r_[actual-HEIGHT_115_GEOMETRIC_MIN,
              angles-joint_range[:,0],joint_range[:,1]-angles,
              np.deg2rad(5)-pose,np.deg2rad(5)+pose]
    scale=np.r_[np.full(2,.001),np.full(16,.1),np.full(6,np.deg2rad(5))]
    return raw/scale,actual,angles,pose


def rollout(model,pre,start,columns,delta,joint_ids,joint_range):
    d=mujoco.MjData(model);first=pre[start]
    d.time=float(first[0]);d.qpos[:]=first[columns['qpos_start']:columns['qpos_start']+model.nq]
    d.qvel[:]=first[columns['qvel_start']:columns['qvel_start']+model.nv]
    d.qacc_warmstart[:]=first[columns['warmstart_start']:columns['warmstart_start']+model.nv]
    values=[];contacts=[];clipped=0;peak_motor=0.
    for step in range(HORIZON):
        row=pre[start+step]
        base=row[columns['ctrl_start']:columns['ctrl_start']+model.nu]
        for aid in range(model.nu):
            dof=model.jnt_dofadr[model.actuator_trnid[aid,0]]
            speed=float(d.qvel[dof])
            bound,_=sim.hw.torque_limit(float('inf'),speed,aid<4,0.,DT)
            bound=min(bound,float(model.actuator_ctrlrange[aid,1]))
            requested=float(base[aid]+delta[aid]);d.ctrl[aid]=np.clip(requested,-bound,bound)
            clipped+=abs(requested-d.ctrl[aid])>1e-9
        peak_motor=max(peak_motor,float(np.max(abs(d.ctrl))))
        mujoco.mj_step(model,d)
        value,_,_,_=metric(model,d,joint_ids,joint_range)
        values.append(value)
        contacts.append(set(forces_cpu(model,d)))
    g=np.asarray(values)
    return g,contacts,dict(min_normalized_margin=float(g.min()),
                           min_actual_leg_m=float(HEIGHT_115_GEOMETRIC_MIN+.001*np.min(g[:,:2])),
                           min_joint_normalized_margin=float(np.min(g[:,2:18])),
                           min_attitude_normalized_margin=float(np.min(g[:,18:])),
                           clipped_motor_physical_commands=int(clipped),peak_motor_torque_Nm=peak_motor)


def bounds_on_baseline(model,pre,start,columns):
    lower=np.full(model.nu,-np.inf);upper=np.full(model.nu,np.inf)
    for step in range(HORIZON):
        row=pre[start+step];base=row[columns['ctrl_start']:columns['ctrl_start']+model.nu]
        velocity=row[columns['qvel_start']:columns['qvel_start']+model.nv]
        for aid in range(model.nu):
            dof=model.jnt_dofadr[model.actuator_trnid[aid,0]]
            bound,_=sim.hw.torque_limit(float('inf'),float(velocity[dof]),aid<4,0.,DT)
            bound=min(bound,float(model.actuator_ctrlrange[aid,1]))
            lower[aid]=max(lower[aid],-bound-base[aid]);upper[aid]=min(upper[aid],bound-base[aid])
    assert np.all(lower<=0) and np.all(upper>=0)
    return lower,upper


def run(output):
    assert not output.exists()
    folder=ROOT/'wheelleg_warp/results/height_115_local_states_20260928'
    source=folder/'verification.json';capture=json.loads(source.read_text())
    pair=json.loads((ROOT/'wheelleg_warp/results/height_115_cpu_horizon_pair_20260928/verification.json').read_text())
    assert pair['summary']['horizon_pass'] and pair['source_windows_sha256']==capture['windows_sha256']
    archive=np.load(folder/'windows.npz');columns=capture['columns'];rows=[];linear={}
    for event_id in EVENTS:
        event=capture['events'][event_id];w=event['world']
        m=model(HeightTerrainScenario(**event['scenario']))
        pre=archive[f'event_{event_id}_pre'];post=archive[f'event_{event_id}_post']
        start=int(np.argmin(abs(pre[:,0]-(event['reference_s']-.02))))
        assert start+HORIZON<=len(pre)
        joint_ids=np.array([m.joint(name).qposadr[0] for name in JOINTS])
        joint_range=np.asarray([m.jnt_range[m.joint(name).id] for name in JOINTS])
        baseline,baseline_contact,base_summary=rollout(m,pre,start,columns,np.zeros(6),joint_ids,joint_range)
        warp_metrics=[];scratch=mujoco.MjData(m)
        for step in range(HORIZON):
            sample=post[start+step]
            scratch.qpos[:]=sample[columns['qpos_start']:columns['qpos_start']+m.nq]
            warp_metrics.append(metric(m,scratch,joint_ids,joint_range)[0])
        pair_metric_error=float(np.max(abs(baseline-np.asarray(warp_metrics))))
        assert pair_metric_error<GAMMA,(event_id,pair_metric_error)
        assert any(row['world']==w and row['event']==event['kind'] for row in pair['rows'])
        low,high=bounds_on_baseline(m,pre,start,columns)
        sensitivity=np.empty((HORIZON,baseline.shape[1],6));mode_changes=0;perturb_clipped=0
        for aid in range(6):
            delta=np.zeros(6);delta[aid]=EPS
            plus,cplus,pdetail=rollout(m,pre,start,columns,delta,joint_ids,joint_range)
            delta[aid]=-EPS
            minus,cminus,mdetail=rollout(m,pre,start,columns,delta,joint_ids,joint_range)
            sensitivity[:,:,aid]=(plus-minus)/(2*EPS)
            mode_changes+=sum(a!=b or a!=c for a,b,c in zip(baseline_contact,cplus,cminus))
            perturb_clipped+=pdetail['clipped_motor_physical_commands']+mdetail['clipped_motor_physical_commands']
        g=baseline.reshape(-1);s=sensitivity.reshape(-1,6)
        full_box=linprog(np.zeros(6),A_ub=-s,b_ub=g,bounds=list(zip(low,high)),method='highs')
        trust_low=np.maximum(low,-EPS);trust_high=np.minimum(high,EPS)
        assert np.all(trust_low<=trust_high)
        # A zero-cost full-box LP can select a saturated vertex far outside the derivative's local range.
        # Within ±0.1 Nm, minimize the peak added torque while keeping a small numerical pairing margin.
        safe=np.column_stack((-s,np.zeros(len(g))))
        positive=np.column_stack((np.eye(6),-np.ones(6)))
        negative=np.column_stack((-np.eye(6),-np.ones(6)))
        result=linprog(np.r_[np.zeros(6),1.],A_ub=np.vstack((safe,positive,negative)),
                       b_ub=np.r_[g-GAMMA,np.zeros(12)],
                       bounds=list(zip(trust_low,trust_high))+[(0.,EPS)],method='highs')
        candidate=None
        if result.success:
            exact,contact,details=rollout(m,pre,start,columns,result.x[:6],joint_ids,joint_range)
            candidate=dict(delta_torque_Nm=result.x[:6].tolist(),minimum_peak_delta_torque_Nm=float(result.x[6]),
                           linear_min_normalized_margin=float(np.min(g+s@result.x[:6])),**details,
                           contact_pair_steps_equal=sum(a==b for a,b in zip(contact,baseline_contact)),
                           constant_delta_applied=bool(details['clipped_motor_physical_commands']==0),
                           nonlinear_all_constraints_passed=bool(np.min(exact)>=-1e-9),
                           nonlinear_numerical_margin_passed=bool(np.min(exact)>=GAMMA-1e-7))
        linear[f'event_{event_id}_baseline']=baseline
        linear[f'event_{event_id}_sensitivity']=sensitivity
        linear[f'event_{event_id}_delta_bounds']=np.stack((low,high),axis=1)
        linear[f'event_{event_id}_trust_bounds']=np.stack((trust_low,trust_high),axis=1)
        rows.append(dict(world=w,event=event['kind'],reference_s=event['reference_s'],
                         start_s=float(pre[start,0]),baseline=base_summary,
                         cpu_warp_pair_max_normalized_diff=pair_metric_error,
                         delta_bounds_Nm=np.stack((low,high),axis=1).tolist(),
                         trust_bounds_Nm=np.stack((trust_low,trust_high),axis=1).tolist(),
                         finite_difference_eps_Nm=EPS,contact_mode_changed_perturbed_steps=mode_changes,
                         finite_difference_clipped_motor_commands=perturb_clipped,
                         linearization_contact_stable=bool(mode_changes==0 and perturb_clipped==0),
                         full_box_linear_feasible=bool(full_box.success),
                         linear_feasible=bool(result.success),linear_status=result.message,
                         candidate=candidate))
    output.mkdir(parents=True)
    np.savez_compressed(output/'linearization.npz',**linear)
    summary=dict(states=len(rows),linear_feasible=sum(row['linear_feasible'] for row in rows),
                 nonlinear_verified_same_contact=sum(bool(row['candidate'] and row['candidate']['nonlinear_numerical_margin_passed']
                     and row['candidate']['contact_pair_steps_equal']==HORIZON and row['candidate']['constant_delta_applied']
                     and row['linearization_contact_stable']) for row in rows),
                 max_cpu_warp_pair_normalized_diff=max(row['cpu_warp_pair_max_normalized_diff'] for row in rows),
                 all_finite_difference_contacts_stable=all(row['contact_mode_changed_perturbed_steps']==0 for row in rows))
    sources=('wheelleg_warp/probe_height_115_local_lp.py','wheelleg_warp/probe_height_115_local_states.py',
             'wheelleg_warp/probe_height_115_cpu_horizon_pair.py','wheelleg_warp/probe_height_115_passive.py',
             'wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py',
             'wheelleg_ppo/tools/wheelleg_sim.py','wheelleg_ppo/tools/hardware_profile.py',
             'wheelleg_ppo/tools/state_estimation.py','wheelleg_ppo/xml/wheelleg.xml')
    payload=dict(role='public_flat_20ms_constant_six_motor_delta_local_feasibility',training=False,final_holdout_opened=False,
                 horizon_s=.02,finite_difference_eps_Nm=EPS,local_trust_box_Nm=EPS,
                 numerical_pairing_margin_normalized=GAMMA,
                 action_set='constant_six_motor_delta_on_replayed_baseline_ctrl',
                 not_evaluated_action_set='filtered_three_dimensional_M3_residual',summary=summary,rows=rows,
                 source_windows_sha256=capture['windows_sha256'],
                 linearization_sha256=hashlib.sha256((output/'linearization.npz').read_bytes()).hexdigest(),
                 source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources})
    (output/'verification.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
