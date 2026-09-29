"""固定结构HiGHS更新的等价、顺序依赖及配对热点计时。"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import warp as wp
from audit_height_115_control_timing import load_cases, direct, stats, _core
from audit_height_115_compiled_problem import make_builder
from probe_height_115_action_predict_loow import ROOT, DT, sha
import probe_height_115_live_braking as reference


def solver():
    h=_core._Highs()
    for key,value in [('output_flag',False),('log_to_console',False),('presolve','on'),('solver','choose'),
                      ('simplex_strategy',int(_core.simplex_constants.SimplexStrategy.kSimplexStrategyDual))]:
        assert h.setOptionValue(key,value)==_core.HighsStatus.kOk
    return h


class UpdatingLP:
    def __init__(self):
        self.h=solver();self.A=None;self.b=None

    def solve(self,A,b):
        assert A.shape==(34,4) and b.shape==(34,) and np.isfinite(A).all() and np.isfinite(b).all()
        if self.A is None:
            status,x=direct(self.h,[0.,0.,0.,1.],A,b)
        else:
            for i,j in np.argwhere(A!=self.A):
                assert self.h.changeCoeff(int(i),int(j),float(A[i,j]))==_core.HighsStatus.kOk
            for i in np.flatnonzero(b!=self.b):
                assert self.h.changeRowBounds(int(i),-_core.kHighsInf,float(b[i]))==_core.HighsStatus.kOk
            assert self.h.run()==_core.HighsStatus.kOk
            status=self.h.getModelStatus()
            x=np.array(self.h.getSolution().col_value) if status==_core.HighsModelStatus.kOptimal else None
        self.A=A.copy();self.b=b.copy()
        return status,x


def run():
    out=ROOT/'wheelleg_warp/results/height_115_persistent_lp_20260929';assert not out.exists()
    m,G,reserve,limits,qa,va,samples,inputs=load_cases();build,views=make_builder(m,G,reserve,qa,va)
    problems=[];original=reference.linprog;captured=None
    def capture(c,**kw):
        nonlocal captured
        result=original(c,**kw);captured=(np.array(kw['A_ub']),np.array(kw['b_ub']),result)
        return result
    try:
        reference.linprog=capture
        for _,_,q,v,a0,nom,_ in samples:
            reference.solve_from_acceleration(m,q,v,a0,G,reserve,nom);A,b=build(q,v,a0,nom)
            problems.append((A.copy(),b.copy(),captured,reference.basis(m,q,v)))
    finally:reference.linprog=original
    engine=UpdatingLP();checks=[];times=[]
    orders=[np.arange(46),np.arange(45,-1,-1),np.random.default_rng(115202609).permutation(46)]
    for order_id,order in enumerate(orders):
        for i in order:
            A,b,(oldA,oldb,ref),B=problems[i];tic=perf_counter();status,x=engine.solve(A,b);duration=perf_counter()-tic
            check=dict(order=order_id,case=int(i),seconds=duration,reference_success=bool(ref.success))
            if ref.success:
                assert status==_core.HighsModelStatus.kOptimal
                residual=float(max((oldA@x-oldb).max(),-x[3],x[3]-1));objective=float(abs(x[3]-ref.fun))
                motor=float(abs(B@(x[:3]-ref.x[:3])).max())
                check.update(original_residual=residual,objective_error=objective,motor_error=motor)
                assert residual<=1e-6 and objective<=1e-6 and motor<=1e-6,check
            else:assert status==_core.HighsModelStatus.kInfeasible
            checks.append(check)
            if len(checks)>1:times.append(duration)
    source=ROOT/'wheelleg_warp/results/height_115_recovery_feedback_v2_20260929/trace.npz'
    with np.load(source) as data:history_v=data['pre_v'];history_u=data['actions']
    qgpu=wp.zeros((1,m.nq),dtype=wp.float32,device='cuda:0');vgpu=wp.zeros((1,m.nv),dtype=wp.float32,device='cuda:0')
    ngpu=wp.zeros((1,6),dtype=wp.float32,device='cuda:0');cgpu=wp.zeros((1,6),dtype=wp.float32,device='cuda:0')
    fresh=solver();updated=UpdatingLP();pipeline={'reload':[],'update':[]};records=[]
    for repeat in range(4):
        for i,(_,step,q,v,_,nom,expected) in enumerate(samples[:44]):
            qgpu.assign(q[None].astype(np.float32));vgpu.assign(v[None].astype(np.float32));ngpu.assign(nom[None].astype(np.float32))
            pv=history_v[step-1,1,va];pa=history_u[step-1,1]
            methods=('reload','update') if (repeat+i)%2==0 else ('update','reload')
            for name in methods:
                wp.synchronize();tic=perf_counter()
                cq=qgpu.numpy()[0].astype(float);cv=vgpu.numpy()[0].astype(float);cn=ngpu.numpy()[0].astype(float)
                ca=(cv[va]-pv)/DT-G@pa;A,b=build(cq,cv,ca,cn)
                status,x=direct(fresh,[0.,0.,0.,1.],A,b) if name=='reload' else updated.solve(A,b)
                assert status==_core.HighsModelStatus.kOptimal and np.max(A@x-b)<=1e-6 and -1e-6<=x[3]<=1+1e-6
                delta=views['B'][0]@x[:3];issued=cn+delta
                assert np.max(abs(delta))<=1+1e-6 and np.max(abs(issued)-views['box'][0])<=1e-6
                cgpu.assign(issued[None].astype(np.float32));wp.synchronize();elapsed=perf_counter()-tic
                assert np.max(abs(reference.basis(m,cq,cv)@(x[:3]-expected)))<=1e-6
                if repeat:
                    pipeline[name].append(elapsed);records.append(dict(repeat=repeat,case=i,method=name,seconds=elapsed))
    result=dict(role='persistent_fixed_matrix_LP_equivalence_and_paired_hotspot',equivalence_checks=checks,
        update_solve_excluding_first_load=stats(times),paired_hotspot={k:stats(v) for k,v in pipeline.items()},
        over_05ms={k:sum(t>.0005 for t in v) for k,v in pipeline.items()},records=records,
        limitations=['Private installed HiGHS API; no new dependency or controller deployment.',
            'Fresh load and incremental update alternate order on identical archived state inputs; actual recorded previous velocity/action used.',
            'Observed timing excludes physics, nominal controller and real sensors, and does not establish worst-case execution time.',
            'Equivalence gate includes forward, reverse and fixed shuffled sequences, including infeasible-to-feasible transitions.'],
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/audit_height_115_compiled_problem.py',
            ROOT/'wheelleg_warp/native/safety_problem.py',ROOT/'wheelleg_warp/audit_height_115_control_timing.py')})
    out.mkdir();(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print({k:result[k] for k in ('update_solve_excluding_first_load','paired_hotspot','over_05ms')},flush=True)


if __name__=='__main__':run()
