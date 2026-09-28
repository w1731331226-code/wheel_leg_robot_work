"""三处平地局部可行解与原M3差模动作子空间的受限线性比较。"""
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
from native.terrain import HeightTerrainScenario,model
from ppo_env import Residual
from state_estimation import leg_kinematics


def max_min_margin(g,s,basis,trust):
    projected=s@basis
    constraints=[np.column_stack((-projected,np.ones(len(g))))]
    upper=[g]
    if trust is not None:
        constraints.extend((np.column_stack((basis,np.zeros(6))),
                            np.column_stack((-basis,np.zeros(6)))))
        upper.extend((np.full(6,trust),np.full(6,trust)))
    result=linprog(np.array([0.,0.,0.,-1.]),A_ub=np.vstack(constraints),b_ub=np.concatenate(upper),
                   bounds=[(-1.,1.)]*3+[(None,None)],method='highs')
    assert result.success,result.message
    return dict(action=result.x[:3].tolist(),max_min_normalized_margin=float(result.x[3]),
                zero_margin_feasible=bool(result.x[3]>=-1e-9),
                numerical_margin_0p001_feasible=bool(result.x[3]>=.001-1e-9))


def run(output):
    assert not output.exists()
    folder=ROOT/'wheelleg_warp/results/height_115_local_lp_20260928'
    lp_path=folder/'verification.json';lp=json.loads(lp_path.read_text())
    assert lp['summary']['nonlinear_verified_same_contact']==3
    state_path=ROOT/'wheelleg_warp/results/height_115_local_states_20260928/verification.json'
    state=json.loads(state_path.read_text());windows=np.load(state_path.parent/'windows.npz')
    lin=np.load(folder/'linearization.npz')
    for record in (lp,state):
        assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
                   for name,digest in record['source_sha256'].items())
    assert hashlib.sha256((folder/'linearization.npz').read_bytes()).hexdigest()==lp['linearization_sha256']
    assert hashlib.sha256((state_path.parent/'windows.npz').read_bytes()).hexdigest()==state['windows_sha256']
    rows=[]
    for event_id,row in zip((1,9,10),lp['rows']):
        assert state['events'][event_id]['world']==row['world'] and state['events'][event_id]['kind']==row['event']
        scenario=HeightTerrainScenario(**state['events'][event_id]['scenario'])
        m=model(scenario);d=mujoco.MjData(m)
        pre=windows[f'event_{event_id}_pre'];start=int(np.argmin(abs(pre[:,0]-row['start_s'])))
        cols=state['columns']
        d.qpos[:]=pre[start,cols['qpos_start']:cols['qpos_start']+m.nq]
        d.qvel[:]=pre[start,cols['qvel_start']:cols['qvel_start']+m.nv]
        residual=Residual(mode='diff3')
        basis=np.column_stack([residual.map(m,d,np.eye(3)[j]) for j in range(3)])
        assert basis.shape==(6,3)
        expected=np.zeros((6,3))
        for side,sign in (('L',1),('R',-1)):
            joints=[m.joint(name+side).id for name in ('alpha','beta')]
            jac=leg_kinematics(d.qpos[m.jnt_qposadr[joints]],d.qvel[m.jnt_dofadr[joints]])[3]
            actuators=[m.actuator('motor_'+name+side).id for name in ('alpha','beta')]
            expected[actuators,0]=sign*jac[:,0]*(.1*sim.hw.DESIGN_MASS*9.81/2)
            expected[actuators,1]=sign*jac[:,1]
            expected[m.actuator('motor_wheel'+side).id,2]=sign*.3
        assert np.max(abs(basis-expected))<1e-12
        g=lin[f'event_{event_id}_baseline'].reshape(-1)
        s=lin[f'event_{event_id}_sensitivity'].reshape(-1,6)
        chosen=np.asarray(row['candidate']['delta_torque_Nm'])
        projected=basis@np.linalg.lstsq(basis,chosen,rcond=None)[0]
        result=dict(world=row['world'],event=row['event'],start_s=row['start_s'],
                    basis_motor_Nm_per_unit=basis.tolist(),
                    six_motor_candidate_Nm=chosen.tolist(),
                    projection_relative_l2_error=float(np.linalg.norm(chosen-projected)/np.linalg.norm(chosen)),
                    unfiltered_M3_no_motor_trust=max_min_margin(g,s,basis,None),
                    unfiltered_M3_with_0p1Nm_motor_trust=max_min_margin(g,s,basis,.1))
        rows.append(result)
    summary=dict(states=len(rows),all_three_M3_zero_margin_infeasible=all(
        not row['unfiltered_M3_no_motor_trust']['zero_margin_feasible'] for row in rows),
        min_candidate_projection_relative_error=min(row['projection_relative_l2_error'] for row in rows))
    names=('wheelleg_warp/probe_height_115_m3_local.py','wheelleg_warp/probe_height_115_local_lp.py',
           'wheelleg_warp/probe_height_115_local_states.py','wheelleg_warp/native/terrain.py',
           'wheelleg_warp/native/models.py','wheelleg_ppo/tools/ppo_env.py',
           'wheelleg_ppo/tools/state_estimation.py','wheelleg_ppo/xml/wheelleg.xml')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(dict(role='public_fixed_jacobian_constant_M3_motor_subspace_local_linear_test',
            training=False,final_holdout_opened=False,actor_filter_and_common_lambda_included=False,
            action_space='unfiltered M3 map frozen at selected start q; constant 20ms motor increment',
            summary=summary,rows=rows,
            lp_linearization_sha256=lp['linearization_sha256'],
            state_windows_sha256=state['windows_sha256'],
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}),
            stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
