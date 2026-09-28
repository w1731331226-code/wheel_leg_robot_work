"""独立重算切换轨迹误差，区分首次几何失效前的转换与失效后数据。"""
import json
from pathlib import Path
import numpy as np
from probe_height_115_action_predict_loow import ROOT, DT, sha
from probe_height_115_braking_budget import ACTIVE, barriers
from probe_height_115_contact_action_pair import margins
from probe_height_115_passive import geometry
from native.terrain import model, HeightTerrainScenario, HEIGHT_115_GEOMETRIC_MIN


def run():
    folder=ROOT/'wheelleg_warp/results'
    source=folder/'height_115_switching_error_20260929'
    target=source/'prefix_audit.json'
    assert not target.exists()
    report=json.loads((source/'verification.json').read_text())
    for group in ('input_sha256','source_sha256'):
        for path,digest in report[group].items():assert sha(ROOT/path)==digest,path
    fit=json.loads((folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json').read_text())
    bound=json.loads((folder/'height_115_acceleration_error_20260929/verification.json').read_text())
    archive=json.loads((folder/'height_115_local_states_20260928/verification.json').read_text())
    worlds=[]
    for world in range(6):
        trace=source/f'world_{world}.npz';old=report['worlds'][world]
        assert sha(trace)==old['trace_sha256']
        z=np.load(trace);event=archive['events'][fit['selected_event_ids'][world]]
        m=model(HeightTerrainScenario(**event['scenario']))
        qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
        gain=np.array(fit['folds'][world]['gain']);b=bound['folds'][world]
        rows=[]
        for arm in range(7):
            q=z['pre'][arm,:,:m.nq];v=z['pre'][arm,:,m.nq:m.nq+m.nv]
            post=z['post'][arm];vn=z['post_velocity'][arm];c=z['schedule'][:,arm]
            leg=geometry(m,q)[1].min(axis=-1);joint=margins(m,q).min(axis=-1)
            bad=np.flatnonzero((leg<HEIGHT_115_GEOMETRIC_MIN)|(joint<0))
            # Include the transition into failure, exclude states already past the first failure.
            end=int(bad[0]) if len(bad) else 40
            assert end>0
            for age in (0,1,4):
                total=[];radial=[];switch=[]
                for t in range(age,40):
                    k=t-age;prev=z['previous_velocity'][arm,va] if k==0 else v[k-1,va]
                    prev_c=np.zeros(3) if k==0 else c[k-1]
                    pred=(v[k,va]-prev)/DT+gain@(c[t]-prev_c)
                    total.append(pred-(vn[t,va]-v[t,va])/DT)
                    a,_=barriers(q[k,qa],v[k,va],pred,gain,(0.,0.))
                    current,_=barriers(q[t,qa],v[t,va],np.zeros(4),gain,(0.,0.))
                    after,_=barriers(post[t,qa],vn[t,va],np.zeros(4),gain,(0.,0.))
                    radial.append([a[j]['a0']-(after[j]['rate']-current[j]['rate'])/DT for j in range(2)])
                    switch.append(not np.array_equal(c[t],np.zeros(3) if t==0 else c[t-1]))
                e=np.array(total);excess=np.maximum(e-b['positive_error_bound_rad_s2'],-e-b['negative_error_bound_rad_s2'])
                miss=np.any(excess>1e-9,axis=1);sw=np.array(switch)
                original=next(r for r in old['rows'] if r['arm']==arm and r['observation_age_ms']==age*.5)
                assert int(miss.sum())==original['uncovered_steps']
                assert abs(abs(e).max()-original['max_abs_joint_error_rad_s2'])<1e-9
                prefix=np.arange(age,40)<end
                assert prefix.any()
                rows.append(dict(arm=arm,observation_age_ms=age*.5,
                    first_geometrically_invalid_pre_step=end if end<40 else None,
                    prefix_steps=int(prefix.sum()),uncovered_prefix_steps=int(miss[prefix].sum()),
                    switch_prefix_steps=int((sw&prefix).sum()),
                    uncovered_switch_prefix_steps=int((miss&sw&prefix).sum()),
                    max_abs_joint_error_prefix_rad_s2=float(abs(e[prefix]).max()),
                    max_optimistic_radial_error_prefix_m_s2=float(np.array(radial)[prefix].max())))
        worlds.append(dict(world=world,rows=rows))
    result=dict(role='independent_reconstruction_and_first_geometry_failure_prefix_audit',worlds=worlds,
        filter='Pre-state true A and all eight joint limits. Includes transition into first failure; excludes later states. This filter alone is not full task safety.',
        verification_sha256=sha(source/'verification.json'),source_sha256=sha(Path(__file__)))
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for age in (0,.5,2):
        rows=[r for w in worlds for r in w['rows'] if r['observation_age_ms']==age]
        print(age,'prefix misses',sum(r['uncovered_prefix_steps'] for r in rows),'/',sum(r['prefix_steps'] for r in rows),flush=True)
    for w in worlds:
        rows=[r for r in w['rows'] if r['observation_age_ms']==0]
        print(w['world'],sum(r['uncovered_prefix_steps'] for r in rows),sum(r['prefix_steps'] for r in rows),
              max(r['max_abs_joint_error_prefix_rad_s2'] for r in rows),
              max(r['max_optimistic_radial_error_prefix_m_s2'] for r in rows),flush=True)


if __name__=='__main__':
    run()
