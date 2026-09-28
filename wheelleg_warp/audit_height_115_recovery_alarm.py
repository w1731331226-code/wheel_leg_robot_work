"""恢复轨迹的固定因果5ms警报；未来实际几何只作评分。"""
import json
from pathlib import Path
import numpy as np
from probe_height_115_action_predict_loow import ROOT, DT, STEPS, ACTIVE, forecast, sha
from probe_height_115_continuous_common_1nm import GAMMA, LENGTH_SCALE_M, JOINT_SCALE_RAD
from probe_height_115_live_braking import solve_current
from probe_height_115_passive import geometry
from probe_height_115_contact_action_pair import margins
from native.terrain import model, HeightTerrainScenario, HEIGHT_115_GEOMETRIC_MIN


def run():
    folder=ROOT/'wheelleg_warp/results';source=folder/'height_115_contact_release_1s_20260929'
    output=folder/'height_115_recovery_alarm_20260929';assert not output.exists()
    paths=[source/'verification.json',folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json',
        folder/'height_115_local_states_20260928/verification.json']
    captured,fit,archive=[json.loads(p.read_text()) for p in paths]
    assert sha(source/'trace.npz')==captured['trace_sha256']
    with np.load(source/'trace.npz') as data:z={k:data[k] for k in data.files}
    m=model(HeightTerrainScenario(**archive['events'][fit['selected_event_ids'][3]]['scenario']))
    qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    q=z['pre'][1,:,:m.nq];v=z['pre'][1,:,m.nq:m.nq+m.nv];ctrl=z['pre'][1,:,m.nq+m.nv:m.nq+m.nv+6]
    assert np.array_equal(ctrl[40:],z['applied'][1,40:])
    actual=geometry(m,z['post'][1])[1].min(axis=1);joint=margins(m,z['post'][1]).min(axis=1)
    unsafe=(actual<HEIGHT_115_GEOMETRIC_MIN)|(joint<0)
    r=fit['folds'][3]['reserves'];reserve=(r['actual_A_length_m'],r['eight_joint_margin_rad'])
    rows=[]
    # At index41 the preceding measured acceleration already excludes the removed action.
    for t in range(41,len(q)-STEPS+1):
        sample=dict(q0=q[t,qa],v0=v[t,va],vprev=v[t-1,va],coeff=np.zeros((1,3)),
                    active_joint_limits=np.array([m.jnt_range[m.joint(n).id] for n in ACTIVE]))
        pl,pj=forecast(sample,np.zeros((4,3)))
        lh=float(pl.min()-HEIGHT_115_GEOMETRIC_MIN-reserve[0]-GAMMA*LENGTH_SCALE_M)
        jh=float(pj.min()-reserve[1]-GAMMA*JOINT_SCALE_RAD)
        alarm=lh<0 or jh<0
        truth=bool(unsafe[t:t+STEPS].any())
        rows.append(dict(step=t,time_ms=t*.5,alarm=alarm,predicted_leg_margin_m=lh,predicted_joint_margin_rad=jh,
            actual_next5ms_failure=truth,false_safe=bool(not alarm and truth),
            alert_without_next5ms_failure=bool(alarm and not truth),
            optimistic_leg_position_error_m=float(np.max(pl[0]-actual[t:t+STEPS]))))
    failure=int(np.flatnonzero(unsafe)[0]);first=next(x for x in rows if x['alarm'] and x['step']<=failure)
    t=first['step'];action,bs,a0,error=solve_current(m,q[t],v[t],v[t-1,va],np.zeros(3),
        np.array(fit['folds'][3]['gain']),reserve,ctrl[t])
    schedules=[]
    hits=[x['step'] for x in rows if x['alarm'] and x['step']<=failure]
    for stride in (1,2,4,10,40):
        lead=[next(((failure+1-h)*.5 for h in hits if h%stride==phase),None) for phase in range(stride)]
        schedules.append(dict(period_ms=stride*.5,missed_phases=sum(x is None for x in lead),
            minimum_lead_ms=min((x for x in lead if x is not None),default=None)))
    alert_ends=[];on=False
    for row in rows:
        if row['alarm'] and not on:begin=row['time_ms'];on=True
        elif not row['alarm'] and on:alert_ends.append([begin,row['time_ms']-.5]);on=False
    if on:alert_ends.append([begin,rows[-1]['time_ms']])
    result=dict(role='fixed_causal_recovery_forecast_audit',first_alarm=first,
        first_actual_failure_ms=(failure+1)*.5,first_alarm_lead_ms=(failure+1-t)*.5,
        first_alarm_current_true_A_margin_m=(geometry(m,q[t])[1]-HEIGHT_115_GEOMETRIC_MIN).tolist(),
        first_alarm_FK_rate_m_s=[x['rate'] for x in bs[:2]],
        first_alarm_frozen_1Nm_model_action=None if action is None else action.tolist(),
        first_alarm_frozen_1Nm_model_error=error,
        first_alarm_barriers=[dict(name=b['name'],distance=b['h'],rate=b['rate']) for b in bs],
        scanned_predictions=len(rows),false_safe_count=sum(x['false_safe'] for x in rows),
        alert_without_next5ms_failure_count=sum(x['alert_without_next5ms_failure'] for x in rows),
        maximum_optimistic_leg_error_m=max(x['optimistic_leg_position_error_m'] for x in rows),
        frozen_leg_position_reserve_m=reserve[0],alarm_intervals_ms=alert_ends,schedule_phase_audit=schedules,rows=rows,
        limitations=['Public trajectory with fixed nominal command after20ms; no new holdout or online intervention.',
            'Alarm uses only current active q/dq and previous dq; actual A/passive joints are posterior scoring.',
            'Frozen position reserve and G were calibrated elsewhere and are not a continuous safety guarantee.',
            '1Nm constant-acceleration LP still has conservative far-joint constraints and uncalibrated acceleration error.'],
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[source/'trace.npz']},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_action_predict_loow.py',
            ROOT/'wheelleg_warp/probe_height_115_live_braking.py',ROOT/'wheelleg_warp/probe_height_115_braking_budget.py')})
    output.mkdir();(output/'verification.json').write_text(json.dumps(result,ensure_ascii=False)+'\n')
    print({k:v for k,v in result.items() if k not in ('rows','source_sha256','input_sha256','limitations','first_alarm_barriers')},flush=True)


if __name__=='__main__':run()
