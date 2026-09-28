"""2Nm局部标定后一次独立组合动作首步核验，仅诊断oracle候选。"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from probe_height_115_action_predict_loow import ROOT, DT, sha, compare_start
from probe_height_115_braking_budget import ACTIVE, barriers
from probe_height_115_live_common_local import simulate, common_basis
from probe_height_115_radial_authority import torque_box, pose
from probe_height_115_passive import geometry
from probe_height_115_contact_action_pair import margins
from native.terrain import model, HeightTerrainScenario, HEIGHT_115_GEOMETRIC_MIN


def run():
    folder=ROOT/'wheelleg_warp/results';source=folder/'height_115_contact_response_2nm_20260929'
    output=folder/'height_115_contact_candidate_20260929';assert not output.exists()
    fitpath=folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    archivepath=folder/'height_115_local_states_20260928/verification.json'
    report=json.loads((source/'verification.json').read_text());row=report['rows'][2];assert row['world']==3
    fit=json.loads(fitpath.read_text());archive=json.loads(archivepath.read_text())
    scenario=HeightTerrainScenario(**archive['events'][fit['selected_event_ids'][3]]['scenario'])
    m=model(scenario);z=np.load(source/'world_3.npz');assert sha(source/'world_3.npz')==row['trace_sha256']
    q=z['pre'][0,0,:m.nq];v=z['pre'][0,0,m.nq:m.nq+m.nv];nom=z['pre'][0,0,m.nq+m.nv:m.nq+m.nv+6]
    qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    reserve=fit['folds'][3]['reserves'];gain=np.array(row['local_gain']);B=common_basis(m,q,v);lim=torque_box(m,v)
    bs,_=barriers(q[qa],v[va],np.array(row['measured_active_acceleration'][0]),gain,
        (reserve['actual_A_length_m'],reserve['eight_joint_margin_rad']))
    for j in range(2):bs[j]['a0']=row['measured_radial_acceleration'][0][j];bs[j]['g']=np.array(row['local_radial_gain'][j])
    A=[];rhs=[]
    for b in bs:
        if b['rate']<0:
            margin=row['local_affine_radial_max_error_m_s2'] if 'leg' in b['name'] else row['local_affine_response_max_error_rad_s2']
            A.append(np.r_[-b['g'],0]);rhs.append(b['a0']-b['rate']**2/(2*b['h'])-margin-1e-5)
    for j in range(6):
        A.extend([np.r_[B[j],-1],np.r_[-B[j],-1],np.r_[B[j],0],np.r_[-B[j],0]])
        rhs.extend([0,0,lim[j]-nom[j],lim[j]+nom[j]])
    sol=linprog([0,0,0,1],A_ub=A,b_ub=rhs,bounds=[(None,None)]*3+[(0,2)],method='highs')
    assert sol.success and np.max(np.array(A)@sol.x-rhs)<1e-6
    coeff=np.zeros((7,3));coeff[1]=sol.x[:3];coeff[2]=z['coefficients'][1]
    schedule=np.zeros((40,7,3));schedule[0]=coeff
    rec,starts,_,_=simulate([scenario],[row['start_step']],np.zeros((7,3)),schedule=schedule)
    checks=[]
    wheels={m.geom('wheel_collide_'+s).id for s in ('L','R')}
    for arm in range(3):
        r=rec[arm];compare_start(starts[arm],starts[0],m.nq,m.nv,f'candidate arm{arm}')
        assert np.array_equal(r['pre'][0,:m.nq],q) and np.array_equal(r['pre'][0,m.nq:m.nq+m.nv],v)
        assert np.max(abs(r['applied'][0]-nom-B@coeff[arm]))<1e-5
        assert np.max(abs(r['applied'][0])-lim)<1e-6
        acc=(r['post_velocity'][0,va]-v[va])/DT
        nxt,_=barriers(r['post'][0,qa],r['post_velocity'][0,va],np.zeros(4),gain,(0,0))
        actual=np.r_[[(nxt[j]['rate']-bs[j]['rate'])/DT for j in range(2)],np.stack([-acc,acc],axis=1).ravel()]
        slacks={b['name']:float(actual[k]-b['rate']**2/(2*b['h'])) for k,b in enumerate(bs) if b['rate']<0}
        leg=geometry(m,r['post'][0])[1];joint=margins(m,r['post'][0]);att=pose(r['post'][0],scenario)
        nonwheel=any(not wheels.intersection(pair) for pair in r['contacts'][0])
        checks.append(dict(arm=arm,action=coeff[arm].tolist(),
            actual_constant_acceleration_slack=slacks,
            braking_gate_pass=bool(min(slacks.values())>=-1e-5),
            actual_geometry_pose_gate_pass=bool(leg.min()>=HEIGHT_115_GEOMETRIC_MIN and joint.min()>=0 and max(att['world_abs_deg'])<=5 and not nonwheel),
            true_A_margin_m=(leg-HEIGHT_115_GEOMETRIC_MIN).tolist(),
            joint_min_margin_rad=float(joint.min()),peak_extra_motor_Nm=float(abs(r['applied'][0]-nom).max()),
            contacts_equal_zero=r['contacts'][0]==rec[0]['contacts'][0]))
    output.mkdir();np.savez_compressed(output/'trace.npz',coefficients=coeff,starts=starts,
        **{k:np.stack([r[k] for r in rec]) for k in ('pre','post','post_velocity','applied')},contact_raw=rec[0]['contact_raw'])
    result=dict(role='independent_one_step_local_oracle_combination_validation',world=3,checks=checks,
        candidate_predicted_peak_motor_Nm=float(sol.x[3]),
        margin_rule='Axis-fit absolute residual for each barrier family plus fixed 1e-5; empirical, not certified bound.',
        limitations='Candidate uses future paired-state calibration; only first 0.5ms scored. No online, continuous or full-task safety claim.',
        trace_sha256=sha(output/'trace.npz'),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (source/'verification.json',source/'world_3.npz',fitpath,archivepath)},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_live_common_local.py',ROOT/'wheelleg_warp/probe_height_115_braking_budget.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(checks,flush=True)


if __name__=='__main__':run()
