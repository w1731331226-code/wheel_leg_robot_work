"""晚期2Nm轴向标定后的单个组合首步独立CPU/Warp复验。"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from probe_height_115_late_response import paired_step
from probe_height_115_action_predict_loow import ROOT,DT,ACTIVE,sha
from probe_height_115_live_common_local import common_basis
from probe_height_115_radial_authority import torque_box,pose
from probe_height_115_contact_action_pair import margins
from probe_height_115_passive import geometry
from probe_height_115_braking_budget import barriers
from native.terrain import model,HeightTerrainScenario,HEIGHT_115_GEOMETRIC_MIN


def run():
    folder=ROOT/'wheelleg_warp/results';source=folder/'height_115_late_response_20260929'
    output=folder/'height_115_late_candidate_20260929';assert not output.exists()
    paths=[source/'verification.json',folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json',
        folder/'height_115_local_states_20260928/verification.json']
    report,fit,archive=[json.loads(p.read_text()) for p in paths];assert report['status']=='paired_gate_pass'
    assert sha(source/'trace.npz')==report['trace_sha256']
    with np.load(source/'trace.npz') as raw:z={k:raw[k] for k in raw.files}
    calibration=next(x for x in report['models'] if x['backend']=='warp' and x['calibration_peak_Nm']==2)
    zero=next(x for x in report['records'] if x['backend']=='warp' and x['arm']=='zero')
    scenario=HeightTerrainScenario(**archive['events'][fit['selected_event_ids'][3]]['scenario']);m=model(scenario)
    q=z['initial_q'];v=z['initial_v'];nom=z['nominal'];qa=np.array([m.joint(x).qposadr[0] for x in ACTIVE]);va=np.array([m.joint(x).dofadr[0] for x in ACTIVE])
    G=np.array(calibration['gain']);rg=np.array(calibration['radial_gain']);B=common_basis(m,q,v);box=torque_box(m,v)
    rs=fit['folds'][3]['reserves'];reserve=(rs['actual_A_length_m'],rs['eight_joint_margin_rad'])
    bs,_=barriers(q[qa],v[va],np.array(zero['active_acceleration']),G,reserve)
    for j in range(2):bs[j]['a0']=zero['radial_acceleration'][j];bs[j]['g']=rg[j]
    A=[];rhs=[]
    for b in bs:
        assert b['h']>0
        if b['rate']<0:
            margin=calibration['axis_fit_max_radial_error'] if 'leg' in b['name'] else calibration['axis_fit_max_joint_error']
            A.append(np.r_[-b['g'],0.]);rhs.append(b['a0']-b['rate']**2/(2*b['h'])-margin-1e-5)
    for j in range(6):
        A.extend([np.r_[B[j],-1.],np.r_[-B[j],-1.],np.r_[B[j],0.],np.r_[-B[j],0.]])
        rhs.extend([0,0,box[j]-nom[j],box[j]+nom[j]])
    sol=linprog([0,0,0,1],A_ub=A,b_ub=rhs,bounds=[(None,None)]*3+[(0,2)],method='highs')
    assert sol.success and np.max(np.array(A)@sol.x-rhs)<=1e-6
    actions=np.zeros((7,3));actions[1]=sol.x[:3];actions[2]=z['actions'][7]
    issued=(nom+actions@B.T).astype(np.float32)
    assert np.max(abs(issued)-box)<=1e-6 and np.max(abs(issued-nom))<=2+1e-5
    cq,cv,gq,gv,cp,gp,cforce,gforce=paired_step(m,scenario,q,v,issued)
    output.mkdir();np.savez_compressed(output/'trace.npz',initial_q=q,initial_v=v,nominal=nom,actions=actions,issued=issued,
        cpu_q=cq,cpu_v=cv,warp_q=gq,warp_v=gv,cpu_actuator_force=cforce,warp_actuator_force=gforce)
    qerr=float(abs(cq-gq).max());verr=float(abs(cv-gv).max());pair=all(a==b for a,b in zip(cp,gp))
    replay_q=float(abs(gq[0]-z['warp_q'][0]).max());replay_v=float(abs(gv[0]-z['warp_v'][0]).max())
    agreement=bool(qerr<=2e-6 and verr<=1e-3 and pair and replay_q<=2e-6 and replay_v<=1e-3)
    rows=[];wheels={m.geom('wheel_collide_'+s).id for s in ('L','R')}
    for backend,Q,V,pairs in [('cpu',cq,cv,cp),('warp',gq,gv,gp)]:
        for arm in (0,1,2):
            acc=(V[arm,va]-v[va])/DT;nxt,_=barriers(Q[arm,qa],V[arm,va],np.zeros(4),G,reserve)
            actual=np.r_[[(nxt[j]['rate']-bs[j]['rate'])/DT for j in range(2)],np.stack([-acc,acc],axis=1).ravel()]
            slack={b['name']:float(actual[j]-b['rate']**2/(2*b['h'])) for j,b in enumerate(bs) if b['rate']<0}
            L=geometry(m,Q[arm])[1];J=float(margins(m,Q[arm]).min());att=max(pose(Q[arm],scenario)['world_abs_deg']);nonwheel=any(not wheels.intersection(x) for x in pairs[arm])
            rows.append(dict(backend=backend,arm=('zero','combination','Fplus2Nm')[arm],braking_slack=slack,
                braking_gate=bool(min(slack.values())>=-1e-5),physical_gate=bool(L.min()>=HEIGHT_115_GEOMETRIC_MIN and J>=0 and att<=5 and not nonwheel),
                true_A_margin_m=(L-HEIGHT_115_GEOMETRIC_MIN).tolist(),joint_margin_rad=J,attitude_deg=att,
                same_contacts_as_zero=pairs[arm]==pairs[0]))
    accepted=agreement and all(x['braking_gate'] and x['physical_gate'] for x in rows if x['arm']=='combination')
    result=dict(role='independent_late_combination_one_step_validation',status='single_step_pass' if accepted else 'failed_fixed_gate',
        candidate_action=sol.x[:3].tolist(),predicted_peak_Nm=float(sol.x[3]),actual_peak_Nm=float(abs(issued[1]-nom).max()),
        pairing=dict(passed=agreement,q_error=qerr,v_error=verr,contacts_equal=pair,zero_replay_q_error=replay_q,zero_replay_v_error=replay_v),rows=rows,
        margin_rule='Use 2Nm axis-fit maximum absolute joint/radial errors plus1e-5, frozen before independent combination; not a certified error bound.',
        limitations='Cold numerical state and frozen nominal torque for one0.5ms step. Fitted gain and zero-next-step baseline are offline oracle. No domain expansion of online control or sustained/latency success claim.',
        trace_sha256=sha(output/'trace.npz'),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[source/'trace.npz']},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_late_response.py',ROOT/'wheelleg_warp/probe_height_115_braking_budget.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result['status'],result['candidate_action'],result['actual_peak_Nm'],result['pairing'],rows,flush=True)


if __name__=='__main__':run()
