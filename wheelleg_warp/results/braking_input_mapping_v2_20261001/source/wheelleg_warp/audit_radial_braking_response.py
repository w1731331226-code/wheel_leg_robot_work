"""One fixed nominal two-input braking LP; diagnostic only, never an online controller."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import wheelleg_sim as sim
from state_estimation import leg_kinematics
from model_lqr import design,sagittal_basis,vmc_coordinates,check_linearization
from probe_height_115_action_predict_loow import sha
from native.terrain import HEIGHT_115_GEOMETRIC_MIN as LIMIT


def mapped_input(model,reference,delta,virtual):
    """Actual current-J VMC plus radial PD, expressed as three common motor inputs."""
    basis,inputs=sagittal_basis(model);_,g,_=vmc_coordinates(reference)
    feed=np.linalg.solve(g,np.linalg.pinv(inputs)@reference.ctrl)
    full=basis@delta;q=reference.qpos.copy();mujoco.mj_integratePos(model,q,full[:model.nv],1.)
    v=reference.qvel+full[model.nv:]
    joints=[model.joint(name) for name in ('alphaL','betaL')]
    leg,_,_,jac=leg_kinematics(q[[j.qposadr[0] for j in joints]],v[[j.dofadr[0] for j in joints]])
    height=sim.fk_joints(reference.qpos[joints[0].qposadr[0]],reference.qpos[joints[1].qposadr[0]])['leg_len']
    rate=(jac.T@v[[j.dofadr[0] for j in joints]])[0]
    force=feed[2]+sim.hw.DESIGN_MASS/sim.hw.BASELINE_MASS*(sim.KP_LEG*(height-np.linalg.norm(leg))-sim.KD_LEG*rate)
    return np.r_[feed[0]+virtual[0],jac@np.array([force,feed[1]+virtual[1]])]


def nominal_input_jacobian(model,reference,eps=1e-6):
    eye=np.eye(15)*eps;zero=np.zeros(2)
    return np.column_stack([(mapped_input(model,reference,d,zero)-mapped_input(model,reference,-d,zero))/(2*eps) for d in eye])


def run(source):
    target=source/'response_analysis.json';assert not target.exists()
    meta=json.loads((source/'verification.json').read_text());col=meta['columns'];rows=[]
    with np.load(source/'windows.npz',allow_pickle=False) as archive:
        for w in range(6):
            key=f'w{w}_braking';p=archive[key+'_pre'];z=archive[key+'_post'];r=archive[key+'_radial'];d=archive[key+'_diagnostic']
            active=r[:,2]==0;assert active.any() and np.isfinite(d).all()
            pre=p[active];post=z[active];diag=d[active];rad=r[active]
            v=pre[:,col['qvel_start']];a=(post[:,col['qvel_start']]-v)/.0005
            steady=(rad[:,0]-rad[:,3]>=.1)&(rad[:,0]-rad[:,3]<=.75)
            requested=np.mean(diag[:,23:25],axis=1);safe=np.mean(diag[:,25:27],axis=1)
            rows.append(dict(world=w,projected_fraction=float(np.mean(diag[:,28]>0)),
                max_requested_angle_deg=float(np.rad2deg(abs(requested).max())),max_safe_angle_deg=float(np.rad2deg(abs(safe).max())),
                steady_mean_deceleration_m_s2=float(np.mean(-np.sign(v[steady])*a[steady])),
                steady_peak_wheel_Nm=float(abs(rad[steady,12]).max()),
                motor_clipped_steps=int(np.sum(np.max(abs(diag[:,15:21]-pre[:,col['ctrl_start']:]),axis=1)>1e-6))))
            if w==4:initial=pre[0].copy()
    model,_=sim.load_model(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'),True)
    reference,a,b,_=design(model,.115,min_height=.115)
    check_linearization(model,reference,a,b)
    basis,_=sagittal_basis(model);c,g,_=vmc_coordinates(reference)
    torque_state=nominal_input_jacobian(model,reference)
    A=a+b@torque_state;B=b@g[:,:2]
    q=initial[1:col['qvel_start']];v=initial[col['qvel_start']:col['warmstart_start']]
    assert len(q)==model.nq and len(v)==model.nv
    refq=reference.qpos.copy();refq[0]=q[0]
    delta=np.empty(model.nv);mujoco.mj_differentiatePos(model,delta,1.,refq,q)
    x0=np.linalg.pinv(basis)@np.r_[delta,v];assert abs(x0[0])<1e-12
    # Exact 0.5ms model inside each fixed 5ms zero-order hold. No grid search.
    hold=10;N=400;nx=15;nu=2;SX=(N+1)*nx;SU=N*nu;size=SX+SU+1
    feed=np.linalg.pinv(sagittal_basis(model)[1])@reference.ctrl
    torque_input=g[:,:2];torque_box=np.array([4.5/1.05,40.,40.])
    state_rows=[];lower=[];upper=[]
    for index,name in enumerate(('alphaL','passA_L','betaL','passC_L'),3):
        state_rows.append(np.eye(nx)[index]);j=model.joint(name).id;cap=1.4 if name in ('alphaL','betaL') else 2.5
        ref=reference.qpos[model.jnt_qposadr[j]];lower.append(-cap-ref);upper.append(cap-ref)
    state_rows += [np.eye(nx)[2],c[6],np.eye(nx)[10],np.eye(nx)[12],np.eye(nx)[14]]
    lower += [-np.deg2rad(5),LIMIT-.115,-sim.hw.HIP_RATED_RPM*np.pi/30,-sim.hw.HIP_RATED_RPM*np.pi/30,-sim.hw.MOTOR_RATED_RPM*np.pi/30]
    upper += [np.deg2rad(5),.38-.115,sim.hw.HIP_RATED_RPM*np.pi/30,sim.hw.HIP_RATED_RPM*np.pi/30,sim.hw.MOTOR_RATED_RPM*np.pi/30]
    C=np.array(state_rows);lo=np.array(lower);hi=np.array(upper)
    ds=[];du=[];rhs=[];peak=[];P=np.eye(nx);Q=np.zeros((nx,nu))
    for _ in range(hold):
        ds.append(np.r_[C@P,-C@P,torque_state@P,-torque_state@P,P[0:1],-P[0:1]])
        du.append(np.r_[C@Q,-C@Q,torque_state@Q+torque_input,-torque_state@Q-torque_input,Q[0:1],-Q[0:1]])
        rhs.append(np.r_[hi,-lo,torque_box-feed,torque_box+feed,0.,0.])
        peak.append(np.r_[np.zeros(2*len(C)+6),-1.,-1.])
        Q=A@Q+B;P=A@P
    ds=np.concatenate(ds);du=np.concatenate(du);rhs=np.concatenate(rhs);peak=np.concatenate(peak)
    # Sparse multiple shooting keeps the full fifteen states and omitted modes.
    eq=sparse.lil_matrix(((N+1)*nx,size));eq[:nx,:nx]=np.eye(nx)
    for k in range(N):
        sl=slice((k+1)*nx,(k+2)*nx);eq[sl,k*nx:(k+1)*nx]=-P;eq[sl,(k+1)*nx:(k+2)*nx]=np.eye(nx);eq[sl,SX+k*nu:SX+(k+1)*nu]=-Q
    ub=sparse.hstack([sparse.hstack([sparse.kron(sparse.eye(N),ds),sparse.csr_matrix((N*len(rhs),nx))]),sparse.kron(sparse.eye(N),du),sparse.csr_matrix(np.tile(peak,N)[:,None])],format='csr')
    final=np.r_[C,-C,np.eye(nx)[7:8],-np.eye(nx)[7:8],np.eye(nx)[0:1],-np.eye(nx)[0:1]]
    tail=sparse.lil_matrix((len(final),size));tail[:,N*nx:SX]=final;tail[-2:,-1]=-1
    ub=sparse.vstack([ub,tail.tocsr()],format='csr');rhs=np.r_[np.tile(rhs,N),hi,-lo,.03,.03,0.,0.]
    # Original tail gate is a maximum from1.5s onward, not just the endpoint.
    tail_rows=sparse.lil_matrix((2*100*hold,size));row=0
    for k in range(300,N):
        T=np.eye(nx);U=np.zeros((nx,nu))
        for _ in range(hold):
            tail_rows[row:row+2,k*nx:(k+1)*nx]=np.r_[T[7:8],-T[7:8]]
            tail_rows[row:row+2,SX+k*nu:SX+(k+1)*nu]=np.r_[U[7:8],-U[7:8]]
            row+=2;U=A@U+B;T=A@T
    assert row==tail_rows.shape[0]
    ub=sparse.vstack([ub,tail_rows.tocsr()],format='csr');rhs=np.r_[rhs,np.full(row,.03)]
    objective=np.zeros(size);objective[-1]=1
    bounds=[(None,None)]*(size-1)+[(0,.6)]
    solved=linprog(objective,A_ub=ub,b_ub=rhs,A_eq=eq.tocsr(),b_eq=np.r_[x0,np.zeros(N*nx)],bounds=bounds,method='highs')
    report=dict(role='fixed_nominal_linear_braking_feasibility',telemetry=rows,linear_status=int(solved.status),linear_message=solved.message,
        horizon_s=2.,input_hold_s=.005,model_check_hz=2000,inputs=['per_wheel_torque','per_leg_hub_torque'],
        limits=dict(distance_m=.6,tail_speed_m_s=.03,tail_start_s=1.5,pitch_deg=5,active_joint_rad=1.4,passive_joint_rad=2.5,radial_min_m=LIMIT),
        prediction_velocity_pitch_response_per_Nm=(c[[3,5]]@b@g[:,:2]/.0005).tolist(),
        current_vmc_jacobian_derivative_included=True,
        limitations='Nominal linear model at0.115m, sagittal common mode, measured arrival initial state. Current-J VMC and radial PD differentiated at equilibrium; experimental radial guard excluded. This remains first order and does not model arbitrary input-state products. Linear joint/radial outputs are not nonlinear A/B closure, contact or invariance certificates. Not an online or full task result.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_coordinated_radial_events.py',ROOT/'wheelleg_ppo/tools/model_lqr.py')})
    if solved.success:
        states=solved.x[:SX].reshape(N+1,nx);inputs=solved.x[SX:SX+SU].reshape(N,nu)
        eqerr=float(abs(eq@solved.x-np.r_[x0,np.zeros(N*nx)]).max());ineqerr=float(np.max(ub@solved.x-rhs))
        assert eqerr<1e-7 and ineqerr<1e-7
        report.update(linear_minimum_max_distance_m=float(solved.fun),equality_max_error=eqerr,inequality_max_excess=ineqerr)
        np.savez_compressed(source/'linear_braking_lp.npz',states=states,inputs=inputs,A=A,B=B,x0=x0,reference_q=reference.qpos,reference_ctrl=reference.ctrl)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    run(parser.parse_args().source)
