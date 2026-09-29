"""失败附近实际状态的冷数值起点CPU/Warp单步响应，非闭环域扩展。"""
import json
from pathlib import Path
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from scipy.optimize import linprog
from probe_height_115_action_predict_loow import ROOT,DT,ACTIVE,sha
from probe_height_115_live_common_local import common_basis
from probe_height_115_radial_authority import torque_box,pose
from probe_height_115_contact_action_pair import margins,contact_sets,forces_cpu
from probe_height_115_passive import geometry
from probe_height_115_braking_budget import barriers
from native.terrain import model,bank_height_115,HeightTerrainScenario,HEIGHT_115_GEOMETRIC_MIN


def run():
    folder=ROOT/'wheelleg_warp/results';source=folder/'height_115_scheduled_guard_20260929'
    output=folder/'height_115_late_response_20260929';assert not output.exists()
    paths=[source/'verification.json',folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json',
           folder/'height_115_local_states_20260928/verification.json']
    log,fit,archive=[json.loads(p.read_text()) for p in paths]
    selected=next(s for s in log['summaries'] if s['arm']==3);step=selected['failure']['step']
    assert selected['failure']['kind']=='model' and selected['method']=='instant_05ms_reference'
    assert sha(source/'trace.npz')==log['trace_sha256']
    with np.load(source/'trace.npz') as raw:z={k:raw[k] for k in raw.files}
    q=z['pre_q'][step,3].copy();v=z['pre_v'][step,3].copy();nom=z['nominal'][step,3].copy()
    assert np.all(z['actions'][step,3]==0)
    scenario=HeightTerrainScenario(**archive['events'][fit['selected_event_ids'][3]]['scenario']);m=model(scenario)
    assert m.na==0 and abs(m.opt.timestep-DT)<1e-12
    B=common_basis(m,q,v);box=torque_box(m,v);scale=1/np.max(abs(B),axis=0)
    actions=[np.zeros(3)];labels=['zero']
    for peak in (1.,2.):
        for k,name in enumerate(('F','H','W')):
            for sign in (1,-1):actions.append(sign*peak*scale[k]*np.eye(3)[k]);labels.append(f'{name}{sign:+d}_{peak:g}Nm')
    actions=np.array(actions);request=nom+actions@B.T;issued=request.astype(np.float32)
    assert np.max(abs(issued)-box)<=1e-6 and np.max(abs(issued-request))<=1e-5
    n=len(actions);wp.init();wp.set_device('cuda:0')
    _,wm,wd,_=bank_height_115(n,scenario=[scenario]*n)
    wd.qpos.assign(np.tile(q,(n,1)).astype(np.float32));wd.qvel.assign(np.tile(v,(n,1)).astype(np.float32))
    wd.qacc_warmstart.assign(np.zeros((n,m.nv),np.float32));wd.ctrl.assign(issued)
    cpu=[]
    for arm in range(n):
        data=mujoco.MjData(m);data.qpos[:]=q;data.qvel[:]=v;data.qacc_warmstart[:]=0.;data.ctrl[:]=issued[arm]
        mujoco.mj_step(m,data);cpu.append(data)
    mjw.step(wm,wd)
    cq=np.array([d.qpos.copy() for d in cpu]);cv=np.array([d.qvel.copy() for d in cpu]);gq=wd.qpos.numpy().astype(float);gv=wd.qvel.numpy().astype(float)
    cp=[set(forces_cpu(m,d)) for d in cpu];gp=contact_sets(wd,n)
    output.mkdir();np.savez_compressed(output/'trace.npz',initial_q=q,initial_v=v,nominal=nom,actions=actions,issued=issued,
        cpu_q=cq,cpu_v=cv,warp_q=gq,warp_v=gv,cpu_actuator_force=np.array([d.actuator_force for d in cpu]),warp_actuator_force=wd.actuator_force.numpy())
    qerr=float(abs(cq-gq).max());verr=float(abs(cv-gv).max());pair_equal=[a==b for a,b in zip(cp,gp)]
    source_q_error=float(abs(gq[0]-z['post_q'][step,3]).max())
    source_v_error=float(abs(gv[0]-z['post_v'][step,3]).max())
    agreement=bool(qerr<=2e-6 and verr<=1e-3 and all(pair_equal) and source_q_error<=2e-6 and source_v_error<=1e-3)
    qa=np.array([m.joint(k).qposadr[0] for k in ACTIVE]);va=np.array([m.joint(k).dofadr[0] for k in ACTIVE]);G=np.array(fit['folds'][3]['gain'])
    rs=fit['folds'][3]['reserves'];reserve=(rs['actual_A_length_m'],rs['eight_joint_margin_rad'])
    base,_=barriers(q[qa],v[va],np.zeros(4),G,reserve);wheels={m.geom('wheel_collide_'+s).id for s in ('L','R')}
    records=[];models=[]
    for backend,Q,V,pairs in [('cpu',cq,cv,cp),('warp',gq,gv,gp)]:
        acceleration=(V[:,va]-v[va])/DT;radial=[]
        for arm in range(n):
            nxt,_=barriers(Q[arm,qa],V[arm,va],np.zeros(4),G,reserve)
            rad=[(nxt[j]['rate']-base[j]['rate'])/DT for j in range(2)];radial.append(rad)
            L=geometry(m,Q[arm])[1];J=float(margins(m,Q[arm]).min());att=max(pose(Q[arm],scenario)['world_abs_deg']);nonwheel=any(not wheels.intersection(x) for x in pairs[arm])
            records.append(dict(backend=backend,arm=labels[arm],radial_acceleration=rad,active_acceleration=acceleration[arm].tolist(),
                true_A_margin_m=(L-HEIGHT_115_GEOMETRIC_MIN).tolist(),joint_margin_rad=J,attitude_deg=att,
                physical_gate=bool(L.min()>=HEIGHT_115_GEOMETRIC_MIN and J>=0 and att<=5 and not nonwheel),
                contact_pairs=[list(x) for x in sorted(pairs[arm])],same_contacts_as_zero=pairs[arm]==pairs[0]))
        radial=np.array(radial)
        if not agreement:continue
        for peak,start in ((1.,1),(2.,7)):
            gain=np.stack([(acceleration[start+2*k]-acceleration[start+2*k+1])/(2*peak*scale[k]) for k in range(3)],axis=1)
            rg=np.stack([(radial[start+2*k]-radial[start+2*k+1])/(2*peak*scale[k]) for k in range(3)],axis=1)
            bs,_=barriers(q[qa],v[va],acceleration[0],gain,reserve)
            for j in range(2):bs[j]['a0']=radial[0,j];bs[j]['g']=rg[j]
            closing=[b for b in bs if b['rate']<0];assert all(b['h']>0 for b in closing)
            A=[];rhs=[]
            for b in closing:A.append(np.r_[-b['g'],0.]);rhs.append(b['a0']-b['rate']**2/(2*b['h']))
            for j in range(6):
                A.extend([np.r_[B[j],-1.],np.r_[-B[j],-1.],np.r_[B[j],0.],np.r_[-B[j],0.]])
                rhs.extend([0.,0.,box[j]-nom[j],box[j]+nom[j]])
            sol=linprog([0,0,0,1],A_ub=A,b_ub=rhs,bounds=[(None,None)]*3+[(0,peak)],method='highs')
            if sol.success:assert np.max(np.array(A)@sol.x-rhs)<=1e-6
            indices=np.r_[0,np.arange(start,start+6)]
            models.append(dict(backend=backend,calibration_peak_Nm=peak,coupled_feasible=bool(sol.success),
                minimum_model_peak_Nm=float(sol.x[3]) if sol.success else None,action=sol.x[:3].tolist() if sol.success else None,
                axis_fit_max_joint_error=float(abs(acceleration[indices]-acceleration[0]-actions[indices]@gain.T).max()),
                axis_fit_max_radial_error=float(abs(radial[indices]-radial[0]-actions[indices]@rg.T).max()),gain=gain.tolist(),radial_gain=rg.tolist()))
    result=dict(role='late_state_one_step_paired_action_calibration',status='paired_gate_pass' if agreement else 'failed_backend_pairing',
        source_arm=3,source_step=step,source_time_ms=step*.5,numerical_warmstart='zero on both backends; source warmstart unavailable',
        cpu_warp_q_error=qerr,cpu_warp_v_error=verr,contact_pairs_equal=pair_equal,records=records,models=models,
        source_zero_q_error=source_q_error,source_zero_v_error=source_v_error,
        limitations='Archived actual q/v and nominal torque held for one step; controller memory not evolved. Cold numerical solver start, not exact archived trajectory replay. Central axis fits and zero-next-step baseline are offline oracle; combinations and sustained control unvalidated.',
        trace_sha256=sha(output/'trace.npz'),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[source/'trace.npz']},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_contact_action_pair.py',
            ROOT/'wheelleg_warp/probe_height_115_braking_budget.py',ROOT/'wheelleg_warp/native/terrain.py',ROOT/'wheelleg_warp/native/models.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result['status'],qerr,verr,all(pair_equal),flush=True);print(models,flush=True)


if __name__=='__main__':run()
