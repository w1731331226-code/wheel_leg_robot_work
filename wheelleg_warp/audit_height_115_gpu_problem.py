"""GPU驻留装配、单包下载及GPU执行映射的同LP微基准。"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import warp as wp
from native.controller import D
from native.safety_problem import assemble, gather_current, pack_problem, execute_solution
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
from probe_height_115_continuous_common_1nm import GAMMA,LENGTH_SCALE_M,JOINT_SCALE_RAD
from probe_height_115_action_predict_loow import ROOT,sha
from audit_height_115_control_timing import load_cases,stats,_core
from audit_height_115_persistent_lp import UpdatingLP
import probe_height_115_live_braking as reference


def run():
    out=ROOT/'wheelleg_warp/results/height_115_gpu_problem_20260929';assert not out.exists()
    m,G,reserve,limits,qa,va,samples,inputs=load_cases();wp.init();device='cuda:0'
    shapes={'q':(1,4),'v':(1,4),'a0':(1,4),'speeds':(1,6),'nominal':(1,6),
        'A':(1,34,4),'b':(1,34),'B':(1,6,3),'box':(1,6),'info':(1,10,3),'diagnostics':(1,3)}
    b={k:wp.zeros(s,dtype=D,device=device) for k,s in shapes.items()};flag=wp.zeros(1,dtype=wp.int32,device=device)
    gain=wp.array(G,dtype=D,device=device);res=wp.array(np.array(reserve),dtype=D,device=device)
    caps=wp.array(m.actuator_ctrlrange[:,1].copy(),dtype=D,device=device);motor=m.jnt_dofadr[m.actuator_trnid[:,0]]
    args=[b[k] for k in ('q','v','a0','speeds','nominal')]+[gain,res,caps,D(HEIGHT_115_GEOMETRIC_MIN),D(GAMMA*LENGTH_SCALE_M),D(GAMMA*JOINT_SCALE_RAD)]+[b[k] for k in ('A','b','B','box','info','diagnostics')]+[flag]
    packet=wp.zeros((1,174),dtype=D,device=device);solution=wp.zeros((1,4),dtype=D,device=device)
    qfull=wp.zeros((1,m.nq),dtype=wp.float32,device=device);vfull=wp.zeros((1,m.nv),dtype=wp.float32,device=device)
    nom=wp.zeros((1,6),dtype=wp.float32,device=device);pv=wp.zeros((1,4),dtype=D,device=device);pu=wp.zeros((1,3),dtype=D,device=device)
    ids=[wp.array(x.astype(np.int32),dtype=wp.int32,device=device) for x in (qa,va,motor)]
    gather_args=[qfull,vfull,nom,pv,pu,gain]+ids+[b[k] for k in ('q','v','a0','speeds','nominal')]
    ctrl=wp.zeros((1,6),dtype=wp.float32,device=device);execution=wp.zeros(1,dtype=wp.int32,device=device)
    pack_args=[b['A'],b['b'],b['diagnostics'],flag,packet];execute_args=[b['B'],b['box'],b['nominal'],solution,flag,ctrl,execution]
    wp.launch(gather_current,1,gather_args);wp.launch(assemble,1,args);wp.launch(pack_problem,1,pack_args);wp.launch(execute_solution,1,execute_args);wp.synchronize()
    with wp.ScopedCapture() as assembly_graph:
        wp.launch(assemble,1,args);wp.launch(pack_problem,1,pack_args)
    with wp.ScopedCapture() as prepare_graph:
        wp.launch(gather_current,1,gather_args);wp.launch(assemble,1,args);wp.launch(pack_problem,1,pack_args)
    with wp.ScopedCapture() as execute_graph:wp.launch(execute_solution,1,execute_args)
    engine=UpdatingLP();checks=[];old_solver=reference.linprog;captured=None
    def capture(c,**kw):
        nonlocal captured
        r=old_solver(c,**kw);captured=(np.array(kw['A_ub']),np.array(kw['b_ub']),r);return r
    try:
        reference.linprog=capture
        for kind,t,q,v,a0,n,expected in samples:
            reference.solve_from_acceleration(m,q,v,a0,G,reserve,n);oldA,oldb,ref=captured
            for key,value in [('q',q[qa]),('v',v[va]),('a0',a0),('speeds',v[motor]),('nominal',n)]:b[key].assign(value[None])
            wp.capture_launch(assembly_graph.graph);p=packet.numpy()[0];assert p[170]==0 and p[171]<1e-4
            A=p[:136].reshape(34,4);rhs=p[136:170];bs,_=reference.barriers(q[qa],v[va],a0,G,reserve)
            mask=np.array([x['rate']<0 for x in bs]);matrix_error=float(max(abs(np.r_[A[:10][mask],A[10:]]-oldA).max(),abs(np.r_[rhs[:10][mask],rhs[10:]]-oldb).max()))
            assert matrix_error<=1e-6
            status,x=engine.solve(A,rhs);check=dict(case=kind,step=t,matrix_error=matrix_error)
            if ref.success:
                assert status==_core.HighsModelStatus.kOptimal
                residual=float(max((oldA@x-oldb).max(),-x[3],x[3]-1));obj=float(abs(x[3]-ref.fun))
                solution.assign(x[None]);wp.capture_launch(execute_graph.graph)
                assert execution.numpy()[0]==0
                error=float(abs(ctrl.numpy()[0]-(n+reference.basis(m,q,v)@ref.x[:3])).max())
                assert residual<=1e-6 and obj<=1e-6 and error<=1e-6
                check.update(original_residual=residual,objective_error=obj,executed_motor_error=error)
            else:assert status==_core.HighsModelStatus.kInfeasible
            checks.append(check)
    finally:reference.linprog=old_solver
    source=ROOT/'wheelleg_warp/results/height_115_recovery_feedback_v2_20260929/trace.npz'
    with np.load(source) as raw:history_v=raw['pre_v'];history_u=raw['actions']
    times=[];records=[]
    for repeat in range(4):
        for i,(_,t,q,v,a0,n,expected) in enumerate(samples[:44]):
            qfull.assign(q[None].astype(np.float32));vfull.assign(v[None].astype(np.float32));nom.assign(n[None].astype(np.float32))
            pv.assign(history_v[t-1,1,va][None]);pu.assign(history_u[t-1,1][None]);wp.synchronize()
            tic=perf_counter();wp.capture_launch(prepare_graph.graph);p=packet.numpy()[0]
            assert p[170]==0 and p[171]<1e-4 and min(p[172:174])<0
            A=p[:136].reshape(34,4);rhs=p[136:170];status,x=engine.solve(A,rhs)
            assert status==_core.HighsModelStatus.kOptimal and np.max(A@x-rhs)<=1e-6
            solution.assign(x[None]);wp.capture_launch(execute_graph.graph);assert execution.numpy()[0]==0
            elapsed=perf_counter()-tic
            assert np.max(abs(ctrl.numpy()[0]-(n+reference.basis(m,q,v)@expected)))<=1e-6
            if repeat:times.append(elapsed);records.append(dict(repeat=repeat,case=i,seconds=elapsed))
    result=dict(role='GPU_assembly_single_packet_CPU_LP_GPU_execution_hotspot',equivalence=checks,
        hotspot=stats(times),over_05ms=int(sum(t>.0005 for t in times)),records=records,
        previous_CPU_version='d35ab5b',
        scope='GPU gather/current a0/forecast/assembly/pack; one packet download; host incremental LP and primal check; solution upload; GPU current mapping/motor gate and execution-status download. Command verification readback is outside timing.',
        exclusions='Input/history fixture uploads, compilation, physics, nominal controller and real sensor link are outside timing. Single robot; no OS worst-case deadline or multiworld training-throughput claim.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/safety_problem.py',
            ROOT/'wheelleg_warp/audit_height_115_persistent_lp.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/audit_height_115_control_timing.py')})
    out.mkdir();(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result['hotspot'],result['over_05ms'],flush=True)


if __name__=='__main__':run()
