"""同批次同恢复状态：恒a0、限幅趋势与五个零附加对照。"""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import warp as wp
from probe_height_115_recovery_feedback import split_graphs, predict_active
from audit_height_115_acceleration_trend import propagate
from probe_height_115_action_predict_loow import ROOT, DT, ACTIVE, forecast_acceleration, sha, START_LIMITS
from probe_height_115_continuous_common_1nm import GAMMA, LENGTH_SCALE_M, JOINT_SCALE_RAD
from probe_height_115_live_braking import solve_from_acceleration
from probe_height_115_live_common_local import align_start, common_basis
from probe_height_115_radial_authority import torque_box, pose
from probe_height_115_contact_action_pair import margins, contact_sets
from probe_height_115_passive import geometry
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario, bank_height_115, HEIGHT_115_GEOMETRIC_MIN


def run():
    folder=ROOT/'wheelleg_warp/results';output=folder/'height_115_paired_estimators_20260929';assert not output.exists()
    paths=[folder/n/'verification.json' for n in ('height_115_contact_response_2nm_20260929',
        'height_115_action_predict_1nm_single_graph_20260929','height_115_local_states_20260928','height_115_contact_hold_v2_20260929')]
    response,fit,archive,prior=[json.loads(p.read_text()) for p in paths]
    prior_trace=paths[3].parent/'trace.npz';assert sha(prior_trace)==prior['trace_sha256']
    with np.load(prior_trace) as data:old={k:data[k] for k in data.files}
    scenario=HeightTerrainScenario(**archive['events'][fit['selected_event_ids'][3]]['scenario'])
    env=NativeEnv(n=7,scenario=[scenario]*7,bank_factory=bank_height_115,height_conditioned=True,height_design='range115',residual_scale=0)
    env.reset();m=env.cpu;d=env.data;prepare,advance=split_graphs(env)
    qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    limits=np.array([m.jnt_range[m.joint(n).id] for n in ACTIVE]);gain=np.array(fit['folds'][3]['gain'])
    rs=fit['folds'][3]['reserves'];reserve=(rs['actual_A_length_m'],rs['eight_joint_margin_rad'])
    start=response['rows'][2]['start_step'];assert response['rows'][2]['world']==3
    for _ in range(start//40):wp.capture_launch(env.graph)
    for _ in range(start%40):wp.capture_launch(prepare.graph);wp.capture_launch(advance.graph)
    assert np.all(env.state.numpy()[:,0]==start)
    wp.launch(align_start,7,[wp.array(np.full(7,start,np.int32)),d.qpos,d.qvel,d.qacc_warmstart,d.sensordata,
        env.k['state'],env.state,env.command,env.contact_flags,env.stopped_q,env.stopped_v,env.stopped_w,env.active])
    names={1:'constant',2:'limited'};zero_arms=(0,3,4,5,6)
    failures={};timings={1:[],2:[]};rows=[];traces=[];prefix=dict(qpos=0.,qvel=0.,ctrl=0.)
    wheels={m.geom('wheel_collide_'+s).id for s in ('L','R')};fork_checks={};status='completed_comparison'
    for t in range(2040):
        if t==40:
            for label,arr in [('q',d.qpos),('v',d.qvel),('warm',d.qacc_warmstart),('sensor',d.sensordata),
                    ('controller',env.k['state']),('state',env.state),('command',env.command),('contacts',env.contact_flags),
                    ('stopped_q',env.stopped_q),('stopped_v',env.stopped_v),('stopped_w',env.stopped_w),('active',env.active)]:
                a=arr.numpy();a[:]=a[1].copy();arr.assign(a);back=arr.numpy()
                fork_checks[label]=bool(np.array_equal(back,np.broadcast_to(back[1],back.shape)))
            assert all(fork_checks.values())
        wp.capture_launch(prepare.graph)
        q=d.qpos.numpy().astype(float);v=d.qvel.numpy().astype(float);nom=d.ctrl.numpy().astype(float)
        if t==40:
            fork_checks['next_nominal']=bool(np.array_equal(nom,np.broadcast_to(nom[1],nom.shape)));assert fork_checks['next_nominal']
        if t<40:
            for key,e in (('qpos',abs(q-old['pre'][:,t,:m.nq]).max()),('qvel',abs(v-old['pre'][:,t,m.nq:m.nq+m.nv]).max()),
                          ('ctrl',abs(nom-old['pre'][:,t,m.nq+m.nv:m.nq+m.nv+6]).max())):prefix[key]=max(prefix[key],float(e))
        actions=old['schedule'][t].copy() if t<40 else np.zeros((7,3));decisions=[]
        if t>=47:
            for arm in (1,2):
                if arm in failures:continue
                tic=perf_counter();k=t-4;oq=traces[k][0][arm];ov=traces[k][1][arm];on=traces[k][2][arm]
                pv=traces[k-1][1][arm,va];pa=traces[k-1][3][arm];executed=np.array([traces[j][3][arm] for j in range(k,t)])
                if arm==1:qh,vh,a0=predict_active(oq[qa],ov[va],pv,pa,gain,executed)
                else:qh,vh,a0=propagate(oq[qa],ov[va],pv,traces[k-2][1][arm,va],pa,traces[k-2][3][arm],gain,executed,2,
                    older_v=traces[k-3][1][arm,va],older_action=traces[k-3][3][arm])
                pl,pj=forecast_acceleration(qh,vh,a0[None,:],limits)
                predicted=[float(pl.min()-HEIGHT_115_GEOMETRIC_MIN-reserve[0]-GAMMA*LENGTH_SCALE_M),
                    float(pj.min()-reserve[1]-GAMMA*JOINT_SCALE_RAD)]
                alarm=min(predicted)<0;error=None
                if alarm:
                    eq=oq.copy();ev=ov.copy();eq[qa]=qh;ev[va]=vh
                    c,bs,_,error=solve_from_acceleration(m,eq,ev,a0,gain,reserve,on)
                    if error:failures[arm]=dict(kind='model',step=t,time_ms=t*.5,reason=error,
                        barriers=[dict(name=b['name'],h=b['h'],rate=b['rate']) for b in bs],
                        true_A_margin_m=(geometry(m,q[arm])[1]-HEIGHT_115_GEOMETRIC_MIN).tolist())
                    else:actions[arm]=c
                elapsed=(perf_counter()-tic)*1000;timings[arm].append(elapsed)
                decisions.append(dict(arm=arm,observation_step=k,alarm=alarm,predicted=predicted,error=error,
                    estimated_q=qh.tolist(),estimated_v=vh.tolist(),estimated_a0=a0.tolist(),compute_ms=elapsed))
            if len(failures)==2:status='both_estimators_failed';break
        issued=nom.copy()
        for arm in range(7):
            issued[arm]+=common_basis(m,q[arm],v[arm])@actions[arm]
            excess=float((abs(issued[arm])-torque_box(m,v[arm])).max());peak=float(abs(issued[arm]-nom[arm]).max())
            if excess>1e-6 or (t>=40 and peak>1+1e-5):
                if t<40:raise AssertionError(('initial motor gate',arm,t,excess,peak))
                assert arm in names
                failures.setdefault(arm,dict(kind='execution',step=t,time_ms=t*.5,motor_excess=excess,increment=peak))
                actions[arm]=0.;issued[arm]=nom[arm]
        actual=issued.astype(np.float32);d.ctrl.assign(actual);wp.capture_launch(advance.graph)
        qn=d.qpos.numpy().astype(float);vn=d.qvel.numpy().astype(float);pairs=contact_sets(d,7);active=env.active.numpy();checks=[]
        for arm in range(7):
            L=geometry(m,qn[arm])[1];J=float(margins(m,qn[arm]).min());att=max(pose(qn[arm],scenario)['world_abs_deg'])
            nonwheel=any(not wheels.intersection(p) for p in pairs[arm]);safe=bool(L.min()>=HEIGHT_115_GEOMETRIC_MIN and J>=0 and att<=5 and not nonwheel and active[arm])
            checks.append(dict(arm=arm,safe=safe,score_active=bool(t<40 and arm==1 or t>=40 and arm not in failures),
                true_A_margin_m=(L-HEIGHT_115_GEOMETRIC_MIN).tolist(),joint_margin_rad=J,attitude_deg=att,
                nonwheel=nonwheel,active=bool(active[arm]),peak_extra_motor_Nm=float(abs(actual[arm]-nom[arm]).max())))
            if t>=40 and arm in names and not safe:failures.setdefault(arm,dict(kind='physical',step=t,time_ms=(t+1)*.5,check=checks[-1]))
        rows.append(dict(step=t,time_ms=(t+1)*.5,decisions=decisions,checks=checks))
        traces.append((q,v,nom,actions,actual,qn,vn))
        if t<40 and not checks[1]['safe']:status='initial_physical_failure';break
        if t>=40 and len(failures)==2:status='both_estimators_failed';break
    output.mkdir();np.savez_compressed(output/'trace.npz',**{key:np.stack([x[i] for x in traces]) for i,key in enumerate(
        ('pre_q','pre_v','nominal','actions','issued','post_q','post_v'))})
    summaries=[]
    for arm in (1,2):
        eligible=[r for r in rows[40:] if r['checks'][arm]['score_active']]
        summaries.append(dict(arm=arm,method=names[arm],failure=failures.get(arm),
            complete_1020ms=bool(len(rows)==2040 and arm not in failures),scored_recovery_steps=len(eligible),
            min_true_A_margin_m=min((min(r['checks'][arm]['true_A_margin_m']) for r in eligible),default=None),
            nonzero_correction_steps=int(sum(np.any(x[3][arm]) for x in traces[40:])),
            max_compute_ms=max(timings[arm],default=None)))
    result=dict(role='same_graph_same_recovery_state_estimator_comparison',status=status,summaries=summaries,
        completed_steps=len(rows),fork_checks=fork_checks,historical_prefix_errors=prefix,
        historical_numeric_pairing_would_pass=all(prefix[k]<=START_LIMITS[k] for k in prefix),
        historical_pairing_is_not_this_protocol_gate=True,zero_arms=zero_arms,rows=rows,
        stopped_method_continuation='After a method fails it is permanently unscored and receives zero extra action only to allow the other arm to finish. This is not a validated controller fallback.',
        limitations='Initial20ms remains offline oracle; ideal velocity and 2ms guard telemetry delay; no real-time/full-height claim. Same-run physical and fork gates replace cross-run prefix matching in this new protocol.',
        trace_sha256=sha(output/'trace.npz'),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[prior_trace]},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_recovery_feedback.py',
            ROOT/'wheelleg_warp/audit_height_115_acceleration_trend.py',ROOT/'wheelleg_warp/probe_height_115_live_braking.py',
            ROOT/'wheelleg_warp/probe_height_115_action_predict_loow.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False)+'\n');print(summaries,flush=True)


if __name__=='__main__':run()
