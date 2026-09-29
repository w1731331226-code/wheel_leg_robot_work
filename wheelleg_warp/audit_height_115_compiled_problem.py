"""已安装Warp的CPU装配核对照与同步GPU读写热点基准。"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import warp as wp
from native.controller import D
from native.safety_problem import assemble
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
from probe_height_115_continuous_common_1nm import GAMMA, LENGTH_SCALE_M, JOINT_SCALE_RAD
from probe_height_115_action_predict_loow import ROOT, DT, forecast_acceleration, sha
from audit_height_115_control_timing import load_cases, direct, stats, _core
import probe_height_115_live_braking as reference


def make_builder(m,G,reserve,qa,va):
    motor_dofs=m.jnt_dofadr[m.actuator_trnid[:,0]]
    wp.init();shapes={'q':(1,4),'v':(1,4),'a0':(1,4),'speeds':(1,6),'nominal':(1,6),
        'A':(1,34,4),'b':(1,34),'B':(1,6,3),'box':(1,6),'info':(1,10,3),'diagnostics':(1,3)}
    buffers={k:wp.zeros(shape,dtype=D,device='cpu') for k,shape in shapes.items()}
    views={k:x.numpy() for k,x in buffers.items()};flag=wp.zeros(1,dtype=wp.int32,device='cpu')
    fixed=[wp.array(G,dtype=D,device='cpu'),wp.array(np.array(reserve),dtype=D,device='cpu'),
        wp.array(m.actuator_ctrlrange[:,1].copy(),dtype=D,device='cpu'),D(HEIGHT_115_GEOMETRIC_MIN),
        D(GAMMA*LENGTH_SCALE_M),D(GAMMA*JOINT_SCALE_RAD)]
    args=[buffers[k] for k in ('q','v','a0','speeds','nominal')]+fixed+[buffers[k] for k in ('A','b','B','box','info','diagnostics')]+[flag]
    def build(q,v,a0,nom):
        views['q'][0]=q[qa];views['v'][0]=v[va];views['a0'][0]=a0
        views['speeds'][0]=v[motor_dofs];views['nominal'][0]=nom
        wp.launch(assemble,1,args,device='cpu')
        assert flag.numpy()[0]==0
        return views['A'][0],views['b'][0]
    return build,views


def run():
    out=ROOT/'wheelleg_warp/results/height_115_compiled_problem_20260929';assert not out.exists()
    m,G,reserve,limits,qa,va,samples,inputs=load_cases();motor_dofs=m.jnt_dofadr[m.actuator_trnid[:,0]]
    build,views=make_builder(m,G,reserve,qa,va)
    h=_core._Highs()
    for key,value in [('output_flag',False),('log_to_console',False),('presolve','on'),('solver','choose'),
                      ('simplex_strategy',int(_core.simplex_constants.SimplexStrategy.kSimplexStrategyDual))]:
        assert h.setOptionValue(key,value)==_core.HighsStatus.kOk
    original=reference.linprog;captured=None
    def capture(c,**kwargs):
        nonlocal captured
        result=original(c,**kwargs);captured=(np.array(c),np.array(kwargs['A_ub']),np.array(kwargs['b_ub']),result)
        return result
    comparisons=[];assembly_times=[];problem_times=[]
    try:
        reference.linprog=capture
        for i,(kind,t,q,v,a0,nom,expected) in enumerate(samples):
            reference.solve_from_acceleration(m,q,v,a0,G,reserve,nom)
            c,oldA,oldb,sol=captured
            bs,_=reference.barriers(q[qa],v[va],a0,G,reserve)
            mask=np.array([b['rate']<0 for b in bs]);info=np.array([[b['h'],b['rate'],b['a0']] for b in bs])
            A,b=build(q,v,a0,nom)
            assert np.array_equal(views['info'][0,:,1]<0,mask)
            compactA=np.r_[A[:10][mask],A[10:]];compactb=np.r_[b[:10][mask],b[10:]]
            matrix_error=float(max(abs(compactA-oldA).max(),abs(compactb-oldb).max()))
            geometry_error=float(abs(views['info'][0,:,0]-info[:,0]).max())
            info_error=float(abs(views['info'][0]-info).max())
            basis_error=float(abs(views['B'][0]-reference.basis(m,q,v)).max())
            box_error=float(abs(views['box'][0]-reference.torque_box(m,v)).max())
            pl,pj=forecast_acceleration(q[qa],v[va],a0[None,:],limits)
            pred=np.array([pl.min()-HEIGHT_115_GEOMETRIC_MIN-reserve[0]-GAMMA*LENGTH_SCALE_M,
                           pj.min()-reserve[1]-GAMMA*JOINT_SCALE_RAD])
            prediction_error=float(abs(views['diagnostics'][0,1:]-pred).max())
            assert geometry_error<=1e-12 and matrix_error<=1e-6 and info_error<=1e-6 and basis_error<=1e-6 and box_error<=1e-6 and prediction_error<=1e-12,(i,geometry_error,matrix_error,info_error,basis_error,box_error,prediction_error)
            status,x=direct(h,c,A,b)
            residual=None;objective=None;motor_error=None
            if sol.success:
                assert status==_core.HighsModelStatus.kOptimal
                residual=float(max((oldA@x-oldb).max(),-x[3],x[3]-1));objective=float(abs(c@x-sol.fun))
                motor_error=float(abs(reference.basis(m,q,v)@(x[:3]-sol.x[:3])).max())
                assert residual<=1e-6 and objective<=1e-6 and motor_error<=1e-6,(i,residual,objective,motor_error)
            else:assert status==_core.HighsModelStatus.kInfeasible
            comparisons.append(dict(case=kind,step=t,matrix_error=matrix_error,geometry_error=geometry_error,
                info_error=info_error,prediction_error=prediction_error,original_residual=residual,
                objective_error=objective,motor_error=motor_error))
        for _ in range(3):
            for _,_,q,v,a0,nom,_ in samples:
                tic=perf_counter();A,b=build(q,v,a0,nom);assembly_times.append(perf_counter()-tic)
                tic=perf_counter();A,b=build(q,v,a0,nom);direct(h,[0.,0.,0.,1.],A,b);problem_times.append(perf_counter()-tic)
    finally:reference.linprog=original
    # Same synchronous hotspot scope as the preceding benchmark, but compiled geometry/assembly.
    qgpu=wp.zeros((1,m.nq),dtype=wp.float32,device='cuda:0');vgpu=wp.zeros((1,m.nv),dtype=wp.float32,device='cuda:0')
    ngpu=wp.zeros((1,6),dtype=wp.float32,device='cuda:0');cgpu=wp.zeros((1,6),dtype=wp.float32,device='cuda:0');pipeline=[]
    for repeat in range(4):
        for _,_,q,v,a0,nom,expected in samples[:44]:
            qgpu.assign(q[None].astype(np.float32));vgpu.assign(v[None].astype(np.float32));ngpu.assign(nom[None].astype(np.float32));wp.synchronize()
            tic=perf_counter()
            cq=qgpu.numpy()[0].astype(float);cv=vgpu.numpy()[0].astype(float);cn=ngpu.numpy()[0].astype(float)
            # a0 is already available; time its equivalent four-element history arithmetic explicitly.
            ca=(cv[va]-(v[va]-DT*a0))/DT
            A,b=build(cq,cv,ca,cn);status,x=direct(h,[0.,0.,0.,1.],A,b);assert status==_core.HighsModelStatus.kOptimal
            assert np.max(A@x-b)<=1e-6 and -.000001<=x[3]<=1.000001
            delta=views['B'][0]@x[:3];issued=cn+delta
            assert np.max(abs(delta))<=1+1e-6 and np.max(abs(issued)-views['box'][0])<=1e-6
            cgpu.assign(issued[None].astype(np.float32));wp.synchronize()
            elapsed=perf_counter()-tic
            assert np.max(abs(reference.basis(m,cq,cv)@(x[:3]-expected)))<=1e-6
            if repeat:pipeline.append(elapsed)
    result=dict(role='compiled_CPU_Warp_safety_problem_equivalence_and_timing',cases=comparisons,
        assembly_and_prediction=stats(assembly_times),assembly_and_direct_LP=stats(problem_times),
        synchronous_hotspot=stats(pipeline),hotspot_over_05ms=int(sum(x>.0005 for x in pipeline)),
        limits=dict(geometry_and_prediction=1e-12,matrix_and_info=1e-6,original_residual=1e-6,objective=1e-6,motor_difference_Nm=1e-6),
        limitations=['CPU compiled Warp kernel reuses native fk/Jacobian; same finite-difference steps and half-step check retained.',
            'Inactive barrier rows are zero <= zero, preserving the feasible set.',
            'Timing is a microbenchmark, excludes physics/nominal control/real sensors and uses ready host history; not a deadline guarantee.',
            'No new physics or default-controller integration; only the archived 44 feasible and two infeasible cases checked.'],
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/safety_problem.py',
            ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/audit_height_115_control_timing.py',ROOT/'wheelleg_warp/probe_height_115_live_braking.py')})
    out.mkdir();(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print({k:result[k] for k in ('assembly_and_prediction','assembly_and_direct_LP','synchronous_hotspot','hotspot_over_05ms')},flush=True)


if __name__=='__main__':run()
