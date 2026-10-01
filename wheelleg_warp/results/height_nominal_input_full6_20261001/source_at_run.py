"""Fixed causal nominal-command-history audit; no fitting, physics, or promotion."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import mujoco
from probe_height_115_action_predict_loow import DT,ACTIVE,sha
from probe_height_115_live_common_local import common_basis
from probe_height_115_passive import geometry
from probe_height_115_contact_action_pair import margins
from native.terrain import model,HeightTerrainScenario,HEIGHT_115_GEOMETRIC_MIN
from state_estimation import leg_kinematics


def propagate_nominal(q,v,previous_v,previous_action,gain,actions,basis,previous_nominal,nominals,nominal_gain=None):
    n=len(actions)
    arrays=(q,v,previous_v,previous_action,gain,actions,basis,previous_nominal,nominals)
    shapes=((4,),(4,),(4,),(3,),(4,3),(n,3),(6,3),(6,),(n,6))
    if any(np.shape(x)!=s or not np.isfinite(x).all() for x,s in zip(arrays,shapes)):
        raise ValueError('名义输入传播数组形状或有限性错误')
    if np.linalg.matrix_rank(basis)!=3:raise ValueError('共同输入映射不满秩')
    if nominal_gain is not None and (np.shape(nominal_gain)!=(4,6) or not np.isfinite(nominal_gain).all()):raise ValueError('六路名义输入响应无效')
    delta=nominals-previous_nominal
    virtual=delta@np.linalg.pinv(basis).T
    residual=delta-virtual@basis.T
    a0=(v-previous_v)/DT-gain@previous_action
    changes=virtual@gain.T if nominal_gain is None else delta@nominal_gain.T
    qh=q.copy();vh=v.copy()
    for extra,change in zip(actions,changes):
        acceleration=a0+gain@extra+change
        qh+=DT*vh+.5*DT*DT*acceleration;vh+=DT*acceleration
    last=a0+changes[-1] if n else a0
    support=np.sum(abs(actions+virtual)*np.max(abs(basis),axis=0),axis=1)
    return qh,vh,last,residual,support


def self_check():
    B=np.array([[.05,.5,0],[-.05,.5,0],[.05,.5,0],[-.05,.5,0],[0,0,1],[0,0,1]])
    G=np.arange(12).reshape(4,3)/7;a=np.array([1.,-2.,3.,-4.])
    q=np.array([.5,-.7,.52,-.72]);v=np.array([.01,-.02,.03,-.04]);pa=np.array([.2,-.3,.4]);prior=np.array([.1,-.2,.3])
    pv=v-DT*(a+G@(prior+pa));u=np.eye(3);nom=np.array([[.2,.1,-.1],[-.1,.2,.1],[.3,-.2,.4]])
    got=propagate_nominal(q,v,pv,pa,G,u,B,B@prior,nom@B.T)
    acc=a+ (nom+u)@G.T
    np.testing.assert_allclose(got[0],q+3*DT*v+DT**2*np.sum(np.array([2.5,1.5,.5])[:,None]*acc,axis=0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(got[1],v+DT*acc.sum(axis=0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(got[2],a+G@nom[-1],atol=1e-10,rtol=0)
    assert abs(got[3]).max()<1e-12
    held=propagate_nominal(q,v,pv,pa,G,u,B,B@prior,np.tile(B@prior,(3,1)))
    old=a+G@prior
    np.testing.assert_allclose(held[1],v+DT*np.sum(old+u@G.T,axis=0),atol=1e-12,rtol=0)
    assert np.array_equal(q,[.5,-.7,.52,-.72])
    H=G@np.linalg.pinv(B)+np.arange(24).reshape(4,6)/11@(np.eye(6)-B@np.linalg.pinv(B))
    commands=nom@B.T+np.arange(1,4)[:,None]*np.array([.02,.03,-.02,-.03,.04,-.04])
    prior_command=B@prior;pv=v-DT*(a+H@prior_command+G@pa)
    full=propagate_nominal(q,v,pv,pa,G,u,B,prior_command,commands,H)
    acc=a+commands@H.T+u@G.T
    np.testing.assert_allclose(full[1],v+DT*acc.sum(axis=0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(full[0],q+3*DT*v+DT**2*np.sum(np.array([2.5,1.5,.5])[:,None]*acc,axis=0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(full[2],a+H@commands[-1],atol=1e-10,rtol=0)


def virtual6_basis(m,q,v):
    E=np.zeros((6,6))
    for side,names in enumerate((('alphaL','betaL'),('alphaR','betaR'))):
        qa=[m.joint(n).qposadr[0] for n in names];va=[m.joint(n).dofadr[0] for n in names]
        E[2*side:2*side+2,2*side:2*side+2]=leg_kinematics(q[qa],v[va])[3]
    E[4,4]=E[5,5]=1.
    return E


def load_nominal_gains():
    folder=ROOT/'wheelleg_warp/results';path=folder/'height_115_local_lp_20260928/verification.json'
    arcpath=folder/'height_115_local_states_20260928/verification.json'
    meta,arc=[json.loads(p.read_text()) for p in (path,arcpath)]
    linear=path.parent/'linearization.npz';windows=arcpath.parent/'windows.npz'
    assert sha(linear)==meta['linearization_sha256'] and sha(windows)==meta['source_windows_sha256']==arc['windows_sha256']
    assert meta['finite_difference_eps_Nm']==.1
    gains={};records=[]
    with np.load(linear) as z,np.load(windows) as states:
        for event_id,row in zip((1,9,10),meta['rows']):
            assert row['finite_difference_clipped_motor_commands']==0 and row['linearization_contact_stable']
            event=arc['events'][event_id];m=model(HeightTerrainScenario(**event['scenario']))
            assert m.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST and m.opt.timestep==DT
            pre=states[f'event_{event_id}_pre'];i=int(np.argmin(abs(pre[:,0]-row['start_s'])))
            cols=arc['columns'];q=pre[i,cols['qpos_start']:cols['qvel_start']];v=pre[i,cols['qvel_start']:cols['warmstart_start']]
            # Lower joint-margin derivatives are dq/dtau divided by the recorded0.1rad scale.
            H=z[f'event_{event_id}_sensitivity'][0,np.array([2,3,6,7]),:]*.1/DT**2
            K=H@virtual6_basis(m,q,v);gains[row['world']]=K
            records.append(dict(world=row['world'],event=event_id,source_start_s=row['start_s'],motor_response_shape=list(H.shape),
                reconstructed_virtual6_response=K.tolist(),calibration='CPU six independent ±0.1Nm axes, first-step hinge finite differences'))
    held={w:np.mean([g for k,g in gains.items() if k!=w],axis=0) for w in gains}
    return held,records,[path,linear,arcpath,windows]


def run(output):
    assert not output.exists(),output;self_check()
    folder=ROOT/'wheelleg_warp/results'
    fitpath=folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    arcpath=folder/'height_115_local_states_20260928/verification.json'
    fit,arc=[json.loads(p.read_text()) for p in (fitpath,arcpath)]
    inputs=[fitpath,arcpath];results=[];saved={}
    nominal_gains,gain_records,more=load_nominal_gains();inputs.extend(more)
    for name in ('height_115_recovery_feedback_delay2_predict_20260929','height_115_switching_error_20260929'):
        source=folder/name;meta=json.loads((source/'verification.json').read_text());inputs.append(source/'verification.json')
        recovery=name.startswith('height_115_recovery')
        for world in ((3,) if recovery else range(6)):
            m=model(HeightTerrainScenario(**arc['events'][fit['selected_event_ids'][world]]['scenario']))
            qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE]);G=np.array(fit['folds'][world]['gain'])
            p=source/('trace.npz' if recovery else f'world_{world}.npz')
            assert sha(p)==(meta['trace_sha256'] if recovery else meta['worlds'][world]['trace_sha256']);inputs.append(p)
            with np.load(p) as z:raw={k:z[k] for k in z.files}
            for arm in ((1,) if recovery else range(7)):
                if recovery:
                    fullq=np.r_[raw['pre_q'][:,arm],raw['post_q'][-1:,arm]]
                    fullv=np.r_[raw['pre_v'][:,arm],raw['post_v'][-1:,arm]]
                    action=raw['actions'][:,arm];nom=raw['nominal'][:,arm]
                else:
                    fullq=np.r_[raw['pre'][arm,:,:m.nq],raw['post'][arm,-1:]]
                    fullv=np.r_[raw['pre'][arm,:,m.nq:m.nq+m.nv],raw['post_velocity'][arm,-1:]]
                    action=raw['schedule'][:,arm];nom=raw['pre'][arm,:,m.nq+m.nv:m.nq+m.nv+6]
                q=fullq[:,qa];v=fullv[:,va];geo=geometry(m,fullq)
                middle=np.array([(m.body_pos[m.body('leg'+s).id]+m.body_pos[m.body('leg'+s+'_D').id])/2 for s in ('L','R')])
                real_min=np.minimum(geo[1].min(axis=1),np.linalg.norm(geo[5]-middle,axis=-1).min(axis=1))
                bad=np.flatnonzero((real_min<HEIGHT_115_GEOMETRIC_MIN)|(margins(m,fullq).min(axis=1)<0))
                end=int(bad[0]) if len(bad) else len(fullq)
                for age in (1,4):
                    start=max(46 if recovery else 0,age+2);errors=[];diagnostics=[]
                    for t in range(start,min(end,len(q))):
                        k=t-age;B=common_basis(m,fullq[k],fullv[k])
                        qh,vh,ah,residual,support=propagate_nominal(q[k],v[k],v[k-1],action[k-1],G,
                            action[k:t],B,nom[k-1],nom[k:t])
                        a0=(v[k]-v[k-1])/DT-G@action[k-1];acc=a0+action[k:t]@G.T
                        oldq=q[k]+age*DT*v[k]+DT**2*np.sum((age-np.arange(age)-.5)[:,None]*acc,axis=0)
                        oldv=v[k]+DT*acc.sum(axis=0);reference=(v[t]-v[t-1])/DT-G@action[t-1]
                        errors.append(np.stack((np.stack((oldq-q[t],oldv-v[t],a0-reference)),np.stack((qh-q[t],vh-v[t],ah-reference)))))
                        diagnostics.append([float(abs(residual).max()),float(support.max())])
                    key=f'{"recovery" if recovery else "switch"}_w{world}_a{arm}_age{age}'
                    e=np.array(errors);d=np.array(diagnostics);assert len(e)>0
                    metrics=[dict(method=label,q_max_rad=float(abs(e[:,i,0]).max()),v_max_rad_s=float(abs(e[:,i,1]).max()),a0_max_rad_s2=float(abs(e[:,i,2]).max()))
                        for i,label in enumerate(('constant_nominal','known_nominal_history'))]
                    gate=metrics[1]['v_max_rad_s']<=metrics[0]['v_max_rad_s']+1e-9 and metrics[1]['a0_max_rad_s2']<=metrics[0]['a0_max_rad_s2']+1e-9
                    eligibility=(d[:,0]<=1e-5)&(d[:,1]<=1+1e-9)
                    results.append(dict(case=key,age_ms=age*.5,valid_prefix_end_step=end,windows=len(e),metrics=metrics,
                        worst_error_gate_pass=bool(gate),supported_windows=int(eligibility.sum()),
                        projection_residual_max_Nm=float(d[:,0].max()),combined_input_L1_support_max=float(d[:,1].max())))
                    saved[key+'_errors']=e;saved[key+'_projection']=d
                    if not recovery and world in nominal_gains:
                        full_errors=[];support_rows=[]
                        for t in range(start,min(end,len(q))):
                            k=t-age;B=common_basis(m,fullq[k],fullv[k])
                            H=np.linalg.solve(virtual6_basis(m,fullq[k],fullv[k]).T,nominal_gains[world].T).T
                            qh,vh,ah,_,_=propagate_nominal(q[k],v[k],v[k-1],action[k-1],G,action[k:t],B,nom[k-1],nom[k:t],H)
                            reference=(v[t]-v[t-1])/DT-G@action[t-1]
                            full_errors.append(np.stack((qh-q[t],vh-v[t],ah-reference)))
                            support_rows.append([float(abs(nom[k:t]-nom[k-1]).max()),float(np.max(np.sum(abs(action[k:t])*np.max(abs(B),axis=0),axis=1)))])
                        fe=np.array(full_errors);sd=np.array(support_rows)
                        full_metrics=dict(method='heldout_CPU_virtual6_nominal_response',q_max_rad=float(abs(fe[:,0]).max()),v_max_rad_s=float(abs(fe[:,1]).max()),a0_max_rad_s2=float(abs(fe[:,2]).max()))
                        results[-1]['full6_nominal']=dict(metrics=full_metrics,
                            worst_error_gate_pass=bool(full_metrics['v_max_rad_s']<=metrics[0]['v_max_rad_s']+1e-9 and full_metrics['a0_max_rad_s2']<=metrics[0]['a0_max_rad_s2']+1e-9),
                            supported_windows=int(((sd[:,0]<=.1+1e-9)&(sd[:,1]<=1+1e-9)).sum()),nominal_change_peak_Nm=float(sd[:,0].max()),
                            calibration_worlds=[w for w in nominal_gains if w!=world])
                        saved[key+'_full6_errors']=fe;saved[key+'_full6_support']=sd
    failed=[r['case'] for r in results if not r['worst_error_gate_pass']]
    summary=dict(cases=len(results),windows=sum(r['windows'] for r in results),worse_cases=failed,
        all_cases_nonworse=not failed,unsupported_windows=sum(r['windows']-r['supported_windows'] for r in results))
    full=[r for r in results if 'full6_nominal' in r]
    summary['full6_flat_transfer']=dict(cases=len(full),worse_cases=[r['case'] for r in full if not r['full6_nominal']['worst_error_gate_pass']],
        unsupported_windows=sum(r['windows']-r['full6_nominal']['supported_windows'] for r in full))
    paths=[Path(__file__)]+[ROOT/p for p in ('wheelleg_warp/probe_height_115_live_common_local.py',
        'wheelleg_warp/probe_height_115_passive.py','wheelleg_warp/probe_height_115_contact_action_pair.py',
        'wheelleg_warp/native/models.py','wheelleg_warp/native/terrain.py','wheelleg_ppo/tools/state_estimation.py',
        'wheelleg_ppo/tools/wheelleg_sim.py','wheelleg_ppo/tools/hardware_profile.py','wheelleg_ppo/xml/wheelleg.xml','wheelleg_warp/probe_height_115_local_lp.py')]
    output.mkdir(parents=True);np.savez_compressed(output/'errors.npz',**saved)
    result=dict(role='fixed_causal_nominal_command_history_projection_audit',summary=summary,results=results,full6_calibration=gain_records,
        protocol='Only aged q/v, earlier v/action/nominal and commands/actions already issued before target time; frozen aged B and G. No fresh target q/v or nominal enters the estimator.',
        acceptance='Every valid-prefix case must not worsen maximum velocity or a0-proxy error; numerical projection residual<=1e-5Nm and normalized input L1<=1 are separate required eligibility checks. No retrospective coefficient or threshold tuning.',
        limitations='Common3 projection cannot identify differential response. Full6 nominal transfer only uses existing CPU first-step ±0.1Nm data at three flat poses, leaving each tested world out; no terrain/recovery full6 calibration, no certified interaction or noise bound. a0 target subtracts frozen G, not independently measured external acceleration. Old-control archives and interval averages are not safety guarantees.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        errors_sha256=sha(output/'errors.npz'))
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED',summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
