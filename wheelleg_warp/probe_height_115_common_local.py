"""0.115 m三处平地局部安全增量的三维共模虚拟动作可达性。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from scipy.optimize import linprog
from native.terrain import HeightTerrainScenario,model
from probe_height_115_local_lp import EPS,GAMMA,HORIZON,rollout
from probe_height_115_passive import JOINTS
from state_estimation import leg_kinematics


def run(output):
    assert not output.exists()
    folder=ROOT/'wheelleg_warp/results/height_115_local_lp_20260928'
    lp_path=folder/'verification.json';lp=json.loads(lp_path.read_text())
    state_path=ROOT/'wheelleg_warp/results/height_115_local_states_20260928/verification.json'
    state=json.loads(state_path.read_text());windows=np.load(state_path.parent/'windows.npz')
    linear=np.load(folder/'linearization.npz')
    for record in (lp,state):
        assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
                   for name,digest in record['source_sha256'].items())
    assert hashlib.sha256((folder/'linearization.npz').read_bytes()).hexdigest()==lp['linearization_sha256']
    assert hashlib.sha256((state_path.parent/'windows.npz').read_bytes()).hexdigest()==state['windows_sha256']
    rows=[]
    for event_id,row in zip((1,9,10),lp['rows']):
        event=state['events'][event_id]
        assert event['world']==row['world'] and event['kind']==row['event']
        m=model(HeightTerrainScenario(**event['scenario']))
        pre=windows[f'event_{event_id}_pre'];cols=state['columns']
        start=int(np.argmin(abs(pre[:,0]-row['start_s'])))
        q=pre[start,cols['qpos_start']:cols['qpos_start']+m.nq]
        v=pre[start,cols['qvel_start']:cols['qvel_start']+m.nv]
        basis=np.zeros((6,3))
        for side in ('L','R'):
            joints=[m.joint(name+side).id for name in ('alpha','beta')]
            jac=leg_kinematics(q[m.jnt_qposadr[joints]],v[m.jnt_dofadr[joints]])[3]
            motors=[m.actuator('motor_'+name+side).id for name in ('alpha','beta')]
            basis[motors,0]=jac[:,0]
            basis[motors,1]=jac[:,1]
            basis[m.actuator('motor_wheel'+side).id,2]=1.
        assert np.linalg.matrix_rank(basis)==3
        chosen=np.asarray(row['candidate']['delta_torque_Nm'])
        projection=basis@np.linalg.lstsq(basis,chosen,rcond=None)[0]
        g=linear[f'event_{event_id}_baseline'].reshape(-1)
        sensitivity=linear[f'event_{event_id}_sensitivity'].reshape(-1,6)
        mapped=sensitivity@basis
        safe=np.column_stack((-mapped,np.zeros(len(g))))
        upper=np.column_stack((basis,-np.ones(6)))
        lower=np.column_stack((-basis,-np.ones(6)))
        result=linprog(np.r_[np.zeros(3),1.],A_ub=np.vstack((safe,upper,lower)),
                       b_ub=np.r_[g-GAMMA,np.zeros(12)],bounds=[(None,None)]*3+[(0.,EPS)],method='highs')
        candidate=None
        if result.success:
            delta=basis@result.x[:3]
            ids=np.array([m.joint(name).qposadr[0] for name in JOINTS])
            ranges=np.asarray([m.jnt_range[m.joint(name).id] for name in JOINTS])
            baseline,base_contacts,_=rollout(m,pre,start,cols,np.zeros(6),ids,ranges)
            actual,contacts,details=rollout(m,pre,start,cols,delta,ids,ranges)
            candidate=dict(common_virtual_action=result.x[:3].tolist(),
                           delta_motor_Nm=delta.tolist(),minimum_peak_motor_delta_Nm=float(result.x[3]),
                           linear_min_normalized_margin=float(np.min(g+mapped@result.x[:3])),
                           **details,
                           contact_pair_steps_equal=sum(a==b for a,b in zip(base_contacts,contacts)),
                           nonlinear_numerical_margin_passed=bool(np.min(actual)>=GAMMA-1e-7),
                           constant_delta_applied=bool(details['clipped_motor_physical_commands']==0))
        rows.append(dict(world=row['world'],event=row['event'],start_s=row['start_s'],
            basis_common_motor_per_unit=basis.tolist(),
            six_motor_candidate_projection_relative_l2_error=float(np.linalg.norm(chosen-projection)/np.linalg.norm(chosen)),
            common_three_linear_feasible=bool(result.success),linear_status=result.message,candidate=candidate))
    summary=dict(states=len(rows),linear_feasible=sum(r['common_three_linear_feasible'] for r in rows),
                 nonlinear_verified=sum(bool(r['candidate'] and r['candidate']['nonlinear_numerical_margin_passed']
                     and r['candidate']['constant_delta_applied'] and r['candidate']['contact_pair_steps_equal']==HORIZON)
                     for r in rows),
                 max_projection_relative_l2_error=max(r['six_motor_candidate_projection_relative_l2_error'] for r in rows))
    sources=('wheelleg_warp/probe_height_115_common_local.py','wheelleg_warp/probe_height_115_local_lp.py',
             'wheelleg_warp/probe_height_115_local_states.py','wheelleg_warp/probe_height_115_passive.py',
             'wheelleg_warp/native/terrain.py',
             'wheelleg_warp/native/models.py','wheelleg_ppo/tools/state_estimation.py',
             'wheelleg_ppo/tools/wheelleg_sim.py','wheelleg_ppo/tools/hardware_profile.py',
             'wheelleg_ppo/xml/wheelleg.xml')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(dict(role='public_fixed_start_jacobian_common_three_local_test',training=False,final_holdout_opened=False,
            nominal_target_m=.115,horizon_s=.02,local_motor_delta_trust_Nm=EPS,
            numerical_pairing_margin_normalized=GAMMA,
            action_channels=('common_radial_force_N','common_hip_virtual_torque_Nm','common_wheel_torque_Nm'),
            fixed_start_jacobian_then_constant_motor_delta=True,
            actor_and_online_controller_tested=False,summary=summary,rows=rows,
            source_lp_verification_sha256=hashlib.sha256(lp_path.read_bytes()).hexdigest(),
            source_linearization_sha256=lp['linearization_sha256'],
            source_state_windows_sha256=state['windows_sha256'],
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}),
            stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
