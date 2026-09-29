"""同一LP直接接口的完整安全层微基准，包含同步GPU读写；不跑物理。"""
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import numpy as np
import warp as wp
import probe_height_115_live_braking as braking
from audit_height_115_control_timing import direct, stats, _core
from probe_height_115_action_predict_loow import ROOT, DT, ACTIVE, forecast_acceleration, sha
from native.terrain import model, HeightTerrainScenario


def run():
    folder=ROOT/'wheelleg_warp/results';out=folder/'height_115_direct_pipeline_20260929';assert not out.exists()
    paths=[folder/n/'verification.json' for n in ('height_115_recovery_feedback_v2_20260929',
        'height_115_action_predict_1nm_single_graph_20260929','height_115_local_states_20260928','height_115_control_timing_20260929')]
    log,fit,arc,profile=[json.loads(p.read_text()) for p in paths]
    trace=paths[0].parent/'trace.npz';assert sha(trace)==log['trace_sha256']
    with np.load(trace) as raw:z={k:raw[k] for k in raw.files}
    m=model(HeightTerrainScenario(**arc['events'][fit['selected_event_ids'][3]]['scenario']))
    qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    limits=np.array([m.jnt_range[m.joint(n).id] for n in ACTIVE]);G=np.array(fit['folds'][3]['gain'])
    r=fit['folds'][3]['reserves'];reserve=(r['actual_A_length_m'],r['eight_joint_margin_rad'])
    indices=[r['step'] for r in log['rows'][41:] if np.any(r['action'])];assert len(indices)==44
    h=_core._Highs()
    for key,value in [('output_flag',False),('log_to_console',False),('presolve','on'),('solver','choose'),
                      ('simplex_strategy',int(_core.simplex_constants.SimplexStrategy.kSimplexStrategyDual))]:
        assert h.setOptionValue(key,value)==_core.HighsStatus.kOk
    def adapter(c,*,A_ub,b_ub,bounds,method):
        assert bounds==[(None,None)]*3+[(0.,1.)] and method=='highs'
        status,x=direct(h,c,A_ub,b_ub)
        return SimpleNamespace(success=status==_core.HighsModelStatus.kOptimal,x=x,message=str(status))
    original=braking.linprog;braking.linprog=adapter
    times=[];max_action_error=0.;wp.init()
    qgpu=wp.zeros((1,m.nq),dtype=wp.float32,device='cuda:0');vgpu=wp.zeros((1,m.nv),dtype=wp.float32,device='cuda:0')
    ngpu=wp.zeros((1,6),dtype=wp.float32,device='cuda:0');cgpu=wp.zeros((1,6),dtype=wp.float32,device='cuda:0')
    try:
        for repeat in range(4):
            for t in indices:
                qgpu.assign(z['pre_q'][t,1:2].astype(np.float32));vgpu.assign(z['pre_v'][t,1:2].astype(np.float32));ngpu.assign(z['nominal'][t,1:2].astype(np.float32))
                pv=z['pre_v'][t-1,1,va];pa=z['actions'][t-1,1];wp.synchronize()
                tic=perf_counter()
                q=qgpu.numpy()[0].astype(float);v=vgpu.numpy()[0].astype(float);nom=ngpu.numpy()[0].astype(float)
                a0=(v[va]-pv)/DT-G@pa;forecast_acceleration(q[qa],v[va],a0[None,:],limits)
                c,_,_,error=braking.solve_from_acceleration(m,q,v,a0,G,reserve,nom);assert error is None
                issued=nom+braking.basis(m,q,v)@c
                assert np.max(abs(issued)-braking.torque_box(m,v))<=1e-6 and np.max(abs(issued-nom))<=1+1e-6
                cgpu.assign(issued[None,:].astype(np.float32));wp.synchronize()
                elapsed=perf_counter()-tic
                max_action_error=max(max_action_error,float(abs(c-z['actions'][t,1]).max()))
                if repeat:times.append(elapsed)
    finally:braking.linprog=original
    assert max_action_error<=1e-8
    result=dict(role='direct_HiGHS_synchronous_safety_path_microbenchmark',warm_path=stats(times),max_action_difference=max_action_error,
        measured_path='Three GPU downloads, current a0 estimate, 5ms forecast, original barriers/LP assembly/assertions through direct binding, command mapping/current motor check, synchronized control upload.',
        exclusions='No physics or nominal-controller kernel, no real sensors, no OS deadline guarantee; archived history is already available in host memory. Single robot, sequential host execution.',
        control_deadline_ms=.5,observed_over_deadline=int(sum(t>.0005 for t in times)),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[trace]},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/audit_height_115_control_timing.py',
            ROOT/'wheelleg_warp/probe_height_115_live_braking.py',ROOT/'wheelleg_warp/probe_height_115_action_predict_loow.py')})
    out.mkdir();(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':run()
