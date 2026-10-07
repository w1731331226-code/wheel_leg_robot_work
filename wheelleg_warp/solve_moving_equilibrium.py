"""Bounded nominal inverse/forward relative equilibrium;not a controller."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from review_yaw_sector import ROOT,sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import model_lqr as ml
import wheelleg_sim as sim
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/moving_equilibrium_contract_v1'


def problem(m,reference,height,speed,counter):
    left=[m.joint(n).id for n in ('alphaL','passA_L','betaL','passC_L')]
    right=[m.joint(n).id for n in ('alphaR','passA_R','betaR','passC_R')]
    qa=m.jnt_qposadr[left];qb=m.jnt_qposadr[right]
    wheel=np.array([m.jnt_dofadr[m.joint(n).id] for n in ('wheel1','wheel2')])
    basis,inputs=ml.sagittal_basis(m);velocity=basis[m.nv:,7:]
    limits=np.min(np.where(inputs.T!=0,m.actuator_ctrlrange[:,1],np.inf),axis=1)
    lo=m.jnt_range[left,0].copy();hi=m.jnt_range[left,1].copy()
    lo[[0,2]]=np.maximum(lo[[0,2]],-1.4);hi[[0,2]]=np.minimum(hi[[0,2]],1.4)
    omega_limit=sim.hw.MOTOR_NO_LOAD_RPM*2*math.pi/60
    lower=np.r_[.02,-math.pi/2,lo,-limits,-omega_limit]
    upper=np.r_[sim.L1+sim.L2+2*sim.hw.WHEEL_RADIUS,math.pi/2,hi,limits,omega_limit]
    initial=np.r_[reference['q'][2],sim.euler(type('Pose',(),{'qpos':reference['q']})())[1],
                  reference['q'][qa],np.linalg.pinv(inputs)@reference['ctrl'],speed/sim.hw.WHEEL_RADIUS]
    initial=np.clip(initial,np.nextafter(lower,np.inf),np.nextafter(upper,-np.inf))
    data=mujoco.MjData(m)
    def residual(x):
        x=np.asarray(x,float)
        if x.shape!=(10,) or not np.isfinite(x).all():raise ValueError('Finite10variable vector required')
        if counter['used']>=counter['limit']:raise RuntimeError('Registeredresidualcallbudget exhausted')
        counter['used']+=1
        mujoco.mj_resetData(m,data)
        data.qpos[2]=x[0];data.qpos[3:7]=[math.cos(x[1]/2),0,math.sin(x[1]/2),0]
        data.qpos[qa]=data.qpos[qb]=x[2:6]
        data.qvel[0]=speed;data.qvel[wheel]=x[9];data.ctrl[:]=inputs@x[6:9]
        data.qacc_warmstart[:]=0
        mujoco.mj_forward(m,data);data.qacc[:]=0;mujoco.mj_inverse(m,data)
        forces=velocity.T@(data.qfrc_inverse-data.qfrc_actuator)
        length=sim.fk_joints(x[2],x[4])['leg_len']
        result=np.r_[forces,100*(length-height),x[1]]
        if not np.isfinite(result).all():raise ValueError('Nonfinite generalized force residual')
        return result
    return initial,lower,upper,residual,data


def setup():
    p=json.loads((OUT/'proposal.json').read_text())
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    m,_=sim.load_model(ml.XML,True)
    assert (m.nq,m.nv,m.nu,m.na,m.nmocap,m.nplugin)==(17,16,6,0,0,0)
    assert m.opt.timestep==.0005 and m.opt.iterations==100 and int(m.opt.integrator)==3
    assert abs(m.body_mass.sum()-7.)<1e-12
    assert mujoco.get_mjcb_control() is None and mujoco.get_mjcb_passive() is None
    refs=[]
    for ref in p['static_reference_inputs']:
        f=ROOT/ref['path'];assert sha(f)==ref['sha256']
        with np.load(f,allow_pickle=False) as z:refs.append(dict(height=ref['height_m'],q=z['reference_q'].copy(),ctrl=z['reference_ctrl'].copy()))
    return p,m,refs


def unit():
    assert not (OUT/'source_admission.json').exists()
    p,m,refs=setup();counter=dict(used=0,limit=12000);checks=[]
    for ref in refs:
        x,lo,hi,f,data=problem(m,ref,ref['height'],0.,counter);zero=f(x)
        assert zero.shape==(10,) and abs(zero).max()<=1e-6
        mujoco.mj_forward(m,data);assert abs(data.qacc).max()<=1e-4
        moving=[]
        for speed in p['speeds_m_s']:
            x,lo,hi,f,data=problem(m,ref,ref['height'],speed,counter)
            assert np.all((x>lo)&(x<hi)) and np.isclose(x[9],speed/sim.hw.WHEEL_RADIUS)
            value=f(x);assert value.shape==(10,) and np.isfinite(value).all()
            moving.append(dict(speed=speed,residual_max_abs=float(abs(value).max()),initial_variables=x.tolist()))
        checks.append(dict(height=ref['height'],zero_speed_residual_max_abs=float(abs(zero).max()),moving_initial_residuals=moving))
    assert counter['used']==15
    x,lo,hi,f,data=problem(m,refs[0],refs[0]['height'],.7,dict(used=0,limit=0))
    try:f(x)
    except RuntimeError:pass
    else:raise AssertionError('Budget guardfailed')
    x,lo,hi,f,data=problem(m,refs[0],refs[0]['height'],.7,counter)
    try:f(np.full(10,np.nan))
    except ValueError:pass
    else:raise AssertionError('NaNaccepted')
    atomic_json(OUT/'source_admission.json',dict(verified=True,checks=checks,unit_residual_calls=15,
        proposal_sha256=sha(OUT/'proposal.json'),solver_source_sha256=sha(__file__),source_sha256=p['source_sha256'],
        unit_static_forward_inverse_only=True,optimization_calls=0,integration_steps=0,transitionFD_calls=0,
        solve_batch_admitted=True,max_batch_residual_calls=12000,max_nfev_each=100,
        limits='Staticoriginalreference pipeline identity only;movingbalance not solved andno model/controller/stability admission.'))
    print('PASS261 staticforcepipeline/options/bounds/counter/source;0optimization/integration/FD/PPO',flush=True)


def batch():
    assert not any((OUT/n).exists() for n in ('batch_started.json','completion.json','failure.json'))
    a=json.loads((OUT/'source_admission.json').read_text());assert a['verified'] and a['solve_batch_admitted']
    assert a['solver_source_sha256']==sha(__file__) and a['proposal_sha256']==sha(OUT/'proposal.json')
    p,m,refs=setup();counter=dict(used=0,limit=12000);records=[]
    atomic_json(OUT/'batch_started.json',dict(admission_sha256=sha(OUT/'source_admission.json'),residual_limit=12000))
    try:
        for i,ref in enumerate(refs):
            for j,speed in enumerate(p['speeds_m_s']):
                x,lo,hi,f,data=problem(m,ref,ref['height'],speed,counter);before=counter['used']
                result=least_squares(f,x,bounds=(lo,hi),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=100,x_scale='jac')
                error=f(result.x);mujoco.mj_forward(m,data)
                q=data.qpos.copy();v=data.qvel.copy();ctrl=data.ctrl.copy();acc=data.qacc.copy()
                active=np.array([m.jnt_qposadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
                alljoint=[m.joint(n).id for n in ('alphaL','betaL','alphaR','betaR','passA_L','passC_L','passA_R','passC_R')]
                qa=m.jnt_qposadr[alljoint];margin=float(np.minimum(q[qa]-m.jnt_range[alljoint,0],m.jnt_range[alljoint,1]-q[qa]).min())
                motors=[m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
                bounds=np.array([sim.hw.torque_limit(float('inf'),float(v[n]),k<4,0.,.0005)[0] for k,n in enumerate(motors)])
                floor=m.geom('floor').id;wheels={m.geom('wheel_collide_L').id,m.geom('wheel_collide_R').id}
                supported=data.ncon==2 and all(floor in set(c.geom) and (set(c.geom)-{floor}).issubset(wheels) for c in data.contact)
                lengths=[];loops=[]
                for side in ('L','R'):
                    hip=(data.xanchor[m.joint('alpha'+side).id]+data.xanchor[m.joint('beta'+side).id])/2
                    wheelpos=data.xpos[m.body('wheel'+side).id];end=data.site_xpos[m.site('couplerB_'+side+'_end').id]
                    lengths.extend([float(np.linalg.norm(wheelpos-hip)),float(np.linalg.norm(end-hip))])
                    loops.append(float(np.linalg.norm(wheelpos-end)))
                path=OUT/f'height_{i}_speed_{j}.npz';np.savez_compressed(path,initial=x,solution=result.x,lower=lo,upper=hi,
                    residual=error,q=q,v=v,ctrl=ctrl,qacc=acc,qfrc_inverse=data.qfrc_inverse,qfrc_actuator=data.qfrc_actuator,qfrc_passive=data.qfrc_passive)
                record=dict(height=ref['height'],speed=speed,path=path.name,sha256=sha(path),solver_success=bool(result.success),
                    solver_status=int(result.status),nfev=int(result.nfev),actual_residual_calls=counter['used']-before,
                    force_residual_max_abs=float(abs(error).max()),forward_qacc_max_abs=float(abs(acc).max()),
                    active_margin_rad=float((1.4-abs(q[active])).min()),eight_margin_rad=margin,
                    command_excess_Nm=float(np.maximum(abs(ctrl)-bounds,0).max()),
                    actual_torque_excess_Nm=float(np.maximum(abs(data.actuator_force)-bounds,0).max()),
                    bilateral_only_floor_support=bool(supported),
                    minimum_actual_leg_m=min(lengths),maximum_loop_error_m=max(loops),
                    passed=bool(result.success and abs(error).max()<=1e-6 and abs(acc).max()<=1e-4 and margin>=0 and (1.4-abs(q[active])).min()>=0
                                and np.maximum(abs(ctrl)-bounds,0).max()<=1e-6 and np.maximum(abs(data.actuator_force)-bounds,0).max()<=1e-6 and supported
                                and min(lengths)>=.1147044660616607))
                records.append(record);atomic_json(OUT/'progress.json',dict(residual_calls=counter['used'],completed_states=len(records)))
                print('BALANCE',ref['height'],speed,record['passed'],record['force_residual_max_abs'],record['forward_qacc_max_abs'],flush=True)
        assert len(records)==10 and counter['used']<=12000
        atomic_json(OUT/'completion.json',dict(verified=True,records=records,residual_calls=counter['used'],optimization_states=10,
            integration_steps=0,transitionFD_calls=0,new_training_samples=0,admission_sha256=sha(OUT/'source_admission.json'),
            controller_admitted=False,limits='Forcebalance staticforward check;no nonlinear trajectory,actualNomclosedloop oruncertaincontact proof. Independentreview required.'))
    except BaseException as e:
        atomic_json(OUT/'failure.json',dict(error=repr(e),residual_calls=counter['used'],records=records,implicit_retry=False));raise


if __name__=='__main__':{'unit':unit,'batch':batch}[sys.argv[1]]()
