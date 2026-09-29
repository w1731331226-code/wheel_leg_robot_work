"""归档LP的分项时序与等价HiGHS直接调用；不修改闭环控制。"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.sparse import csc_matrix
from scipy.optimize._highspy import _core
import warp as wp
import probe_height_115_live_braking as braking
from probe_height_115_recovery_feedback import predict_active
from probe_height_115_action_predict_loow import ROOT, DT, ACTIVE, forecast_acceleration, sha
from native.terrain import model, HeightTerrainScenario


def stats(values):
    x=np.asarray(values)*1000
    return dict(count=len(x),median_ms=float(np.median(x)),p95_ms=float(np.quantile(x,.95)),max_ms=float(x.max()))


def direct(h,c,A,b):
    matrix=csc_matrix(A);lp=_core.HighsLp();lp.num_col_=4;lp.num_row_=len(b)
    lp.col_cost_=np.array(c,float);lp.col_lower_=np.array([-_core.kHighsInf]*3+[0.]);lp.col_upper_=np.array([_core.kHighsInf]*3+[1.])
    lp.row_lower_=np.full(len(b),-_core.kHighsInf);lp.row_upper_=np.array(b,float)
    lp.a_matrix_.format_=_core.MatrixFormat.kColwise;lp.a_matrix_.num_col_=4;lp.a_matrix_.num_row_=len(b)
    lp.a_matrix_.start_=matrix.indptr;lp.a_matrix_.index_=matrix.indices;lp.a_matrix_.value_=matrix.data
    assert h.passModel(lp)==_core.HighsStatus.kOk
    assert h.run()==_core.HighsStatus.kOk
    status=h.getModelStatus()
    x=np.array(h.getSolution().col_value) if status==_core.HighsModelStatus.kOptimal else None
    return status,x


def load_cases():
    folder=ROOT/'wheelleg_warp/results'
    fitpath=folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    arcpath=folder/'height_115_local_states_20260928/verification.json'
    fit=json.loads(fitpath.read_text());arc=json.loads(arcpath.read_text())
    m=model(HeightTerrainScenario(**arc['events'][fit['selected_event_ids'][3]]['scenario']))
    va=np.array([m.joint(n).dofadr[0] for n in ACTIVE]);qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE])
    limits=np.array([m.jnt_range[m.joint(n).id] for n in ACTIVE]);G=np.array(fit['folds'][3]['gain'])
    r=fit['folds'][3]['reserves'];reserve=(r['actual_A_length_m'],r['eight_joint_margin_rad']);inputs=[fitpath,arcpath];samples=[]
    for name,kind in [('height_115_recovery_feedback_v2_20260929','feasible'),
                      ('height_115_recovery_feedback_delay2_20260929','aged_failure'),
                      ('height_115_recovery_feedback_delay2_predict_20260929','predicted_failure')]:
        p=folder/name;d=json.loads((p/'verification.json').read_text());assert sha(p/'trace.npz')==d['trace_sha256']
        inputs.extend((p/'verification.json',p/'trace.npz'))
        with np.load(p/'trace.npz') as raw:z={k:raw[k] for k in raw.files}
        if kind=='feasible':
            for row in d['rows'][41:]:
                if not np.any(row['action']):continue
                t=row['step'];q=z['pre_q'][t,1];v=z['pre_v'][t,1]
                a0=(v[va]-z['pre_v'][t-1,1,va])/DT-G@z['actions'][t-1,1]
                samples.append((kind,t,q,v,a0,z['nominal'][t,1],np.array(row['action'])))
        else:
            t=d['failure']['step'];k=t-4;q=z['pre_q'][k,1].copy();v=z['pre_v'][k,1].copy()
            pv=z['pre_v'][k-1,1,va];pa=z['actions'][k-1,1];a0=(v[va]-pv)/DT-G@pa
            if kind=='predicted_failure':q[qa],v[va],a0=predict_active(q[qa],v[va],pv,pa,G,z['actions'][k:t,1])
            samples.append((kind,t,q,v,a0,z['nominal'][k,1],None))
    assert len(samples)==46
    return m,G,reserve,limits,qa,va,samples,inputs


def run():
    folder=ROOT/'wheelleg_warp/results';out=folder/'height_115_control_timing_20260929';assert not out.exists()
    m,G,reserve,limits,qa,va,samples,inputs=load_cases()
    originals={n:getattr(braking,n) for n in ('barriers','basis','torque_box','linprog')}
    measured=[];problems=[];bucket={};latest=None
    def wrap(name,func):
        def call(*args,**kwargs):
            nonlocal latest
            tic=perf_counter();result=func(*args,**kwargs);bucket[name]+=perf_counter()-tic
            if name=='linprog':latest=(np.array(args[0]),np.array(kwargs['A_ub']),np.array(kwargs['b_ub']),result)
            return result
        return call
    try:
        for name,func in originals.items():setattr(braking,name,wrap(name,func))
        for repeat in range(4):
            for kind,t,q,v,a0,nom,expected in samples:
                bucket={name:0. for name in originals};latest=None
                tic=perf_counter();braking.solve_from_acceleration(m,q,v,a0,G,reserve,nom);total=perf_counter()-tic
                assert latest is not None;c,A,b,result=latest
                assert result.success==(expected is not None)
                if result.success:assert np.allclose(result.x[:3],expected,atol=1e-8,rtol=0)
                if repeat==0:problems.append((c,A,b,result))
                measured.append(dict(repeat=repeat,case=kind,step=t,total=total,
                    residual_assembly_and_checks=total-sum(bucket.values()),**bucket))
    finally:
        for name,func in originals.items():setattr(braking,name,func)
    h=_core._Highs()
    for key,value in [('output_flag',False),('log_to_console',False),('presolve','on'),('solver','choose'),
                      ('simplex_strategy',int(_core.simplex_constants.SimplexStrategy.kSimplexStrategyDual))]:
        assert h.setOptionValue(key,value)==_core.HighsStatus.kOk
    direct_times=[];max_residual=0.;max_objective_error=0.;max_action_difference=0.
    for repeat in range(4):
        for c,A,b,reference in problems:
            tic=perf_counter();status,x=direct(h,c,A,b);elapsed=perf_counter()-tic
            if reference.success:
                assert status==_core.HighsModelStatus.kOptimal
                residual=float(max((A@x-b).max(),-x[3],x[3]-1))
                objective_error=float(abs(c@x-reference.fun));assert residual<=1e-6 and objective_error<=1e-8
                max_residual=max(max_residual,residual);max_objective_error=max(max_objective_error,objective_error)
                max_action_difference=max(max_action_difference,float(abs(x-reference.x).max()))
            else:assert status==_core.HighsModelStatus.kInfeasible
            if repeat:direct_times.append(elapsed)
    prediction=[]
    for _ in range(3):
        for _,_,q,v,a0,_,_ in samples:
            tic=perf_counter();forecast_acceleration(q[qa],v[va],a0[None,:],limits);prediction.append(perf_counter()-tic)
    transfers=[];wp.init()
    for n in (1,7):
        buffers=[wp.array(np.zeros((n,width),np.float32),device='cuda:0') for width in (m.nq,m.nv,6)]
        download=[];upload=[]
        for repeat in range(51):
            wp.synchronize();tic=perf_counter();host=[x.numpy().astype(float) for x in buffers];read=perf_counter()-tic
            tic=perf_counter();buffers[2].assign(host[2].astype(np.float32));wp.synchronize();write=perf_counter()-tic
            if repeat:download.append(read);upload.append(write)
        transfers.append(dict(worlds=n,download_three_arrays=stats(download),upload_control=stats(upload)))
    keys=('total','barriers','basis','torque_box','linprog','residual_assembly_and_checks')
    result=dict(role='fixed_LP_timing_and_direct_binding_equivalence',scipy_version=scipy.__version__,
        highs_version=h.version(),cases=46,feasible_cases=44,infeasible_cases=2,
        first_cycle={k:stats([r[k] for r in measured if r['repeat']==0]) for k in keys},
        warm_cycles={k:stats([r[k] for r in measured if r['repeat']>0]) for k in keys},
        prediction_only=stats(prediction),direct_binding_including_matrix_load=stats(direct_times),transfers=transfers,
        equivalence=dict(max_primal_residual=max_residual,max_objective_difference=max_objective_error,max_solution_difference=max_action_difference),
        limitations=['Profiling wrappers add overhead; residual includes assembly, validation and wrapper overhead.',
            'Direct path is an experiment using a private installed SciPy API, not a production replacement or a new solver.',
            'Idle GPU transfer microbenchmark excludes waiting for physics/controller kernels and any real sensor link.',
            'Separate component medians are not an end-to-end deadline bound; observed maxima are not worst-case execution guarantees.',
            'No controller thresholds, objective or mathematical constraints changed; no physical rollout or training.'],
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_live_braking.py',ROOT/'wheelleg_warp/probe_height_115_action_predict_loow.py')})
    out.mkdir();(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('warm_cycles','direct_binding_including_matrix_load','prediction_only','transfers','equivalence')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
