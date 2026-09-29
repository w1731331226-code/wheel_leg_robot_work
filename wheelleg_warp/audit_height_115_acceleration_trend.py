"""固定两点加速度趋势的离线转移与噪声审计；不更新闭环或拟合参数。"""
import argparse
import json
from pathlib import Path
import numpy as np
from probe_height_115_action_predict_loow import ROOT, DT, ACTIVE, sha
from native.terrain import model, HeightTerrainScenario, HEIGHT_115_GEOMETRIC_MIN
from probe_height_115_passive import geometry
from probe_height_115_contact_action_pair import margins

LAG=4
METHODS=('constant','linear_trend','limited_trend')


def propagate(q,v,pv,ppv,pa,ppa,gain,actions,trend,older_v=None,older_action=None):
    a=(v-pv)/DT-gain@pa
    previous=(pv-ppv)/DT-gain@ppa
    jerk=(a-previous)/DT if trend else np.zeros(4)
    if trend==2:
        assert older_v is not None and older_action is not None
        older=(ppv-older_v)/DT-gain@older_action
        s1=a-previous;s2=previous-older
        jerk=np.where(s1*s2>0,np.sign(s1)*np.minimum(abs(s1),abs(s2)),0.)/DT
        if not (np.array_equal(pa,ppa) and np.array_equal(pa,older_action) and np.all(actions==pa)):
            jerk=np.zeros(4)
    qh=q.copy();vh=v.copy()
    for j,u in enumerate(actions):
        acceleration=a+(j+1)*DT*jerk+gain@u
        qh+=DT*vh+.5*DT*DT*acceleration;vh+=DT*acceleration
    return qh,vh,a+len(actions)*DT*jerk


def evaluate(q,v,u,gain,start,limited=False):
    methods=range(3 if limited else 2)
    errors=[[] for _ in methods];terminal=[]
    for t in range(start,len(q)):
        k=t-LAG
        reference=(v[t]-v[t-1])/DT-gain@u[t-1]
        for method in methods:
            extra=dict(older_v=v[k-3],older_action=u[k-3]) if method==2 else {}
            qh,vh,ah=propagate(q[k],v[k],v[k-1],v[k-2],u[k-1],u[k-2],gain,u[k:t],method,**extra)
            errors[method].append(np.stack((qh-q[t],vh-v[t],ah-reference)))
            if t==len(q)-1:terminal.append(dict(method=METHODS[method],
                estimated_latest_nominal_acceleration=ah.tolist(),reference=reference.tolist()))
    errors=np.array(errors)
    metrics=[]
    for method in methods:
        one=dict(method=METHODS[method],windows=errors.shape[1])
        for k,key in enumerate(('q_rad','v_rad_s','latest_a0_rad_s2')):
            e=abs(errors[method,:,k,:]);one[key]=dict(median=float(np.median(e)),p95=float(np.quantile(e,.95)),maximum=float(e.max()))
        metrics.append(one)
    return errors,metrics,terminal


def noise_check(limited=False):
    # Unit bounded velocity error, q and issued inputs held exact; no sensor specification implied.
    rows=[]
    for method in range(3 if limited else 2):
        measured=[]
        for signs in np.array(np.meshgrid(*[[-1.,1.]]*4)).T.reshape(-1,4):
            qh,vh,ah=propagate(np.zeros(4),np.full(4,signs[0]),np.full(4,signs[1]),np.full(4,signs[2]),
                np.zeros(3),np.zeros(3),np.zeros((4,3)),np.zeros((LAG,3)),method,
                older_v=np.full(4,signs[3]),older_action=np.zeros(3))
            measured.append([abs(qh).max(),abs(vh).max(),abs(ah).max()])
        bound=np.max(measured,axis=0)
        expected=np.array(([.01,9.,4000.],[.04,49.,36000.],[.025,29.,20000.])[method])
        assert np.allclose(bound,expected,atol=1e-9,rtol=0)
        global_bound=bound if method<2 else np.array([.04,49.,36000.])
        rows.append(dict(method=METHODS[method],zero_signal_corner_max_q_v_a=bound.tolist(),
            global_velocity_error_upper_gain_q_v_a=global_bound.tolist(),
            stress_bounds=[dict(velocity_error_bound_rad_s=e,q_bound_rad=float(global_bound[0]*e),
                v_bound_rad_s=float(global_bound[1]*e),a_bound_rad_s2=float(global_bound[2]*e)) for e in (1e-5,1e-4,1e-3)]))
    return rows


def run(limited=False):
    folder=ROOT/'wheelleg_warp/results'
    output=folder/('height_115_acceleration_limited_trend_20260929' if limited else 'height_115_acceleration_trend_20260929')
    assert not output.exists()
    fitpath=folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    arcpath=folder/'height_115_local_states_20260928/verification.json'
    fit=json.loads(fitpath.read_text());arc=json.loads(arcpath.read_text());inputs=[fitpath,arcpath];arrays={};results=[]
    names=['height_115_recovery_feedback_delay2_predict_20260929','height_115_switching_error_20260929']
    for name in names:
        source=folder/name;meta=json.loads((source/'verification.json').read_text());inputs.append(source/'verification.json')
        worlds=(3,) if name==names[0] else range(6)
        for w in worlds:
            m=model(HeightTerrainScenario(**arc['events'][fit['selected_event_ids'][w]]['scenario']))
            qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
            gain=np.array(fit['folds'][w]['gain']);path=source/('trace.npz' if name==names[0] else f'world_{w}.npz')
            expected=meta['trace_sha256'] if name==names[0] else meta['worlds'][w]['trace_sha256']
            assert sha(path)==expected;inputs.append(path)
            with np.load(path) as data:z={k:data[k] for k in data.files}
            arms=(1,) if name==names[0] else range(7)
            for arm in arms:
                if name==names[0]:
                    q=np.r_[z['pre_q'][:,arm,qa],z['post_q'][-1:,arm,qa]]
                    v=np.r_[z['pre_v'][:,arm,va],z['post_v'][-1:,arm,va]];u=z['actions'][:,arm];start=46
                else:
                    q=np.r_[z['pre'][arm,:,:m.nq][:,qa],z['post'][arm,-1:,qa].reshape(1,4)]
                    v=np.r_[z['pre'][arm,:,m.nq:m.nq+m.nv][:,va],z['post_velocity'][arm,-1:,va].reshape(1,4)]
                    u=z['schedule'][:,arm];start=6
                if limited:start+=1
                error,metrics,terminal=evaluate(q,v,u,gain,start,limited)
                key=f'{"recovery" if name==names[0] else "switch"}_w{w}_a{arm}'
                arrays[key]=error
                prefix_metrics=None
                if limited:
                    fullq=np.r_[z['pre_q'][:,arm],z['post_q'][-1:,arm]] if name==names[0] else np.r_[z['pre'][arm,:,:m.nq],z['post'][arm,-1:]]
                    bad=np.flatnonzero((geometry(m,fullq)[1].min(axis=1)<HEIGHT_115_GEOMETRIC_MIN)|(margins(m,fullq).min(axis=1)<0))
                    end=int(bad[0]) if len(bad) else len(fullq);mask=np.arange(start,len(q))<end
                    prefix_metrics=[dict(method=METHODS[i],windows=int(mask.sum()),
                        v_max_rad_s=float(abs(error[i,mask,1]).max()),a0_max_rad_s2=float(abs(error[i,mask,2]).max())) for i in range(3)]
                results.append(dict(case=key,metrics=metrics,prefix_metrics=prefix_metrics,terminal=terminal if name==names[0] else None))
    output.mkdir();np.savez_compressed(output/'errors.npz',**arrays)
    result=dict(role='fixed_causal_acceleration_trend_offline_audit',lag_ms=2.,limited=limited,
        previous_source_git_revision='8d60abf',results=results,noise=noise_check(limited),
        limiter='Same-sign adjacent acceleration increments: use smaller magnitude; otherwise zero trend. Also zero trend after any known safety-action change in the history/propagation interval.' if limited else None,
        acceptance='Do not promote if cross-scenario worst velocity or acceleration error grows; a single terminal improvement is insufficient.',
        limitations=['No fitting or new physics; all data are public development trajectories, with correlated windows.',
            'Fresh-frame a0 reference subtracts frozen G times issued action; it is not independently identified true external acceleration.',
            'Trend uses only aged q/dq and two preceding velocity frames, previous issued inputs and known executed actions.',
            'Velocity-noise bounds assume exact position, gain and command, not a measured hardware sensor specification.'],
        trace_sha256=sha(output/'errors.npz'),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        source_sha256=sha(Path(__file__)))
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(results[0],flush=True)
    for method in range(3 if limited else 2):print(METHODS[method],
        {k:max(r['metrics'][method][k]['maximum'] for r in results[1:]) for k in ('q_rad','v_rad_s','latest_a0_rad_s2')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limited',action='store_true')
    run(parser.parse_args().limited)
