"""Receive frozen984 model episodes and decide the original force-study gates offline."""
import json
import pickle
import numpy as np
import mujoco
import torch
from stable_baselines3 import PPO
from train_fixed_force_study import OUT,verify
from train_height_comparison import summary
from score_reference_learning import load_rows,yaw_gate
from complete_contact_witness import check_files
from run_phase_support_qualification import logs
from parking_withdrawal_probe import check_log
from fixed_reference_force import embed
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
import wheelleg_sim as sim


def chain_minima(q,m):
    # A/B mean the two closed-chain branches, each minimized over both legs.
    values=[[],[]];loop=0.
    def rotate(v,angle):
        c=np.cos(angle);s=np.sin(angle)
        return np.column_stack((c*v[0]+s*v[2],np.full(len(q),v[1]),-s*v[0]+c*v[2]))
    for side in ('L','R'):
        ids=[m.jnt_qposadr[m.joint(n).id] for n in ('alpha'+side,'beta'+side,'passA_'+side,'passC_'+side)]
        offsets=[m.body_pos[m.body(n).id].astype(np.float32).astype(float) for n in ('leg'+side,'kneeA_'+side,'wheel'+side,'leg'+side+'_D','kneeB_'+side)]
        offsets.append(m.site_pos[m.site('couplerB_'+side+'_end').id].astype(np.float32).astype(float))
        a=offsets[0]+rotate(offsets[1],q[:,ids[0]])+rotate(offsets[2],q[:,ids[0]]+q[:,ids[2]])
        b=offsets[3]+rotate(offsets[4],q[:,ids[1]])+rotate(offsets[5],q[:,ids[1]]+q[:,ids[3]])
        mid=(offsets[0]+offsets[3])/2
        values[0].append(float(np.linalg.norm(a-mid,axis=1).min()));values[1].append(float(np.linalg.norm(b-mid,axis=1).min()))
        loop=max(loop,float(np.linalg.norm(a-b,axis=1).max()))
    return [min(v) for v in values],loop


def physical_metrics(pre,post,m):
    names=('alphaL','betaL','passA_L','passC_L','alphaR','betaR','passA_R','passC_R')
    joints=[m.joint(n).id for n in names];qids=m.jnt_qposadr[joints]
    limits=m.jnt_range[joints].astype(np.float32).astype(float);q=post[:,qids]
    margin=float(np.minimum(q-limits[:,0],limits[:,1]-q).min())
    vids=[m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
    rpm=abs(pre[:,17+np.asarray(vids)])*60/(2*np.pi)
    rated=np.array([175.]*4+[490.]*2);no_load=np.array([280.]*4+[710.]*2);peak=np.array([40.]*4+[4.5]*2)
    bounds=peak*np.where(rpm>rated,np.maximum(0,(no_load-rpm)/(no_load-rated)),1)
    return margin,max(0.,float((abs(post[:,33:39])-bounds).max())),max(0.,float((abs(pre[:,33:39])-bounds).max()))


def retention(candidate,reference,bounds,legacy=False):
    assert [r['seed'] for r in candidate]==[r['seed'] for r in reference]
    lost=[];velocity=[];arrival=[];attitude=[]
    for x,y in zip(candidate,reference):
        if not y['success']:continue
        if not x['success']:lost.append(x['seed'])
        if legacy:
            if x['velocity_rmse'] is None or x['velocity_rmse']>bounds['velocity_multiplier']*y['velocity_rmse']+bounds['velocity_add_m_s']:velocity.append(x['seed'])
            if x['arrival_s'] is None or x['arrival_s']>bounds['arrival_multiplier']*y['arrival_s']+bounds['arrival_add_s']:arrival.append(x['seed'])
            if any(x['peak_deg'][j]>y['peak_deg'][j]+bounds['legacy_roll_pitch_add_deg'] for j in (0,1)):attitude.append(x['seed'])
    return dict(lost_success=lost,velocity_failures=velocity,arrival_failures=arrival,attitude_failures=attitude,
                passed=not(lost or velocity or arrival or attitude))


def self_check():
    b=dict(velocity_multiplier=1.05,velocity_add_m_s=.005,arrival_multiplier=1.05,arrival_add_s=.05,legacy_roll_pitch_add_deg=.1)
    r=dict(seed=1,success=True,velocity_rmse=.1,arrival_s=2.,peak_deg=[1.,1.,0.])
    assert retention([r],[r],b,True)['passed']
    assert not retention([{**r,'success':False}],[r],b)['passed']
    assert not retention([{**r,'arrival_s':None}],[r],b,True)['passed']
    assert not retention([{**r,'velocity_rmse':.111}],[r],b,True)['passed']
    assert not retention([{**r,'peak_deg':[1.10001,1.,0.]}],[r],b,True)['passed']
    assert retention([{**r,'success':False}],[{**r,'success':False}],b)['passed']
    try:retention([r],[{**r,'seed':2}],b)
    except AssertionError:pass
    else:raise AssertionError('Mismatched paired case admitted')
    assert yaw_gate([.20,.21,.22],.30) and not yaw_gate([.20,.21,.31],.30)
    assert not yaw_gate([None,.20,.20],.30) and not yaw_gate([.26]*3,.30)
    spec=mujoco.MjSpec.from_file(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'));sim.hw.configure_spec(spec);m=spec.compile()
    pre=np.zeros((1,39));post=pre.copy();post[0,33]=41.
    assert physical_metrics(pre,post,m)[1]==1.
    print('PASS338 lostcase/legacy missingmetrics/boundaries/pairedIDs/allseed yaw;0physics',flush=True)


def run():
    self_check();p=verify();out=OUT/'evaluation_review.json';assert not out.exists()
    a=json.loads((OUT/'evaluation_source.json').read_text());admission=json.loads((OUT/'model_evaluation_admission.json').read_text())
    c=json.loads((OUT/'models/completion.json').read_text());base=json.loads((OUT/'baseline_gate.json').read_text())
    rawbase=json.loads((OUT/'baseline_raw_review.json').read_text())
    assert c['verified'] and c['episodes']==984 and len(c['records'])==60 and c['new_learning_samples']==0
    assert a['proposal_sha256']==admission['proposal_sha256']==sha(OUT/'proposal.json')
    assert admission['training_completion_sha256']==sha(OUT/'training/completion.json') and admission['training_review_sha256']==sha(OUT/'training_review.json')
    assert rawbase['verified'] and rawbase['completion_sha256']==sha(OUT/'baseline/completion.json') and rawbase['baseline_gate_sha256']==sha(OUT/'baseline_gate.json')
    for mapping in (a['source_sha256'],admission['model_sha256'],base['input_sha256'],rawbase['raw_sha256']):
        assert all(sha(ROOT/f)==h for f,h in mapping.items())
    conditions=[f'{arm}_{seed}' for seed in p['seeds'] for arm in p['arms']]
    assert [(r['condition'],r['panel'],r['batch']) for r in c['records']]==[(label,j['panel'],j['batch']) for label in conditions for j in p['eval_jobs']]
    cpu=json.loads((OUT/'CPU_legacy_pair.json').read_text());assert cpu['passed'] and base['CPU_legacy_pair_sha256']==sha(OUT/'CPU_legacy_pair.json')
    refs={}
    baseline=json.loads((OUT/'baseline/completion.json').read_text())
    for condition in p['classical_conditions']:
        for panel in ('regular','controlled','legacy'):
            f=OUT/'baseline'/f'{condition["label"]}_{panel}_aggregate.json'
            cases=[case for j in p['eval_jobs'] if j['panel']==panel for case in j['cases']]
            refs[condition['label'],panel]=load_rows(f,cases)
            original=[None]*len(cases)
            for record in baseline['records']:
                if record['condition']!=condition['label'] or record['panel']!=panel:continue
                job=next(j for j in p['eval_jobs'] if j['panel']==panel and j['batch']==record['batch'])
                path=OUT/'baseline'/record['path'];assert sha(path)==record['sha256']
                for index,row in zip(record['indices'],load_rows(path,job['cases'])['runs']):
                    assert original[index] is None;original[index]=row
            assert refs[condition['label'],panel]['runs']==original
    cpuf=ROOT/'wheelleg_ppo/tools/results/fixes_2026-09-17/final/baseline.json'
    assert sha(cpuf)==cpu['CPU_baseline_sha256']
    cpurows=json.loads(cpuf.read_text())['runs'];cpurows=[{**r,'seed':r['name']} for r in cpurows]
    assert [r['seed'] for r in cpurows]==[r['seed'] for r in refs['guard_B0','legacy']['runs']]
    for x,y in zip(refs['guard_B0','legacy']['runs'],cpurows):assert all(x['scenario'][k]==v for k,v in y['scenario'].items())
    spec=mujoco.MjSpec.from_file(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'));sim.hw.configure_spec(spec);m=spec.compile()
    ids=np.array([m.jnt_qposadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
    vids=np.array([m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
    models={};raw_sha={};stats={};steps=0;actor_rows=0;torch.set_num_threads(1)
    for label in conditions:
        arm=label.split('_')[0];d=OUT/'training'/label;model=PPO.load(d/'step_200000.zip',device='cpu')
        assert model.num_timesteps==200000 and model._n_updates==400
        with (d/'step_200000.pkl').open('rb') as stream:norm=pickle.load(stream)
        st=dict(episodes=0,nominal_corrected=0,residual_reduced=0,infeasible=0,minimum_margin=1e30,maximum_model_error=0.,maximum_prediction_violation=-1e30,minimum_secondary_barrier=1e30,maximum_actor_replay_error=0.)
        for panel in ('regular','controlled','legacy'):
            ordered=[None]*sum(len(j['cases']) for j in p['eval_jobs'] if j['panel']==panel)
            for entry in c['records']:
                if entry['condition']!=label or entry['panel']!=panel:continue
                j=next(j for j in p['eval_jobs'] if j['panel']==panel and j['batch']==entry['batch']);assert j['indices']==entry['indices']
                f=OUT/'models'/entry['path'];assert sha(f)==entry['sha256'];result=load_rows(f,j['cases'])
                checked=check_files(f.parent,j['cases'],{r['seed']:r for r in result['runs']});checked.pop('original_label_differences')
                assert checked==entry['checked'] and logs(f.parent,j['cases'])==checked['physics_steps']
                assert checked['physics_steps']==entry['first_episode_world_steps'];steps+=checked['physics_steps']
                for index,row in zip(entry['indices'],result['runs']):
                    assert ordered[index] is None;ordered[index]=row;st['episodes']+=1
                    for field in ('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace','joint_guard_trace','actor_trace'):
                        path=f.parent/row[field]['path'];assert sha(path)==row[field]['sha256'];raw_sha[str(path.relative_to(ROOT))]=sha(path)
                    with np.load(f.parent/row['complete_trace']['path'],allow_pickle=False) as z:
                        trace=z['trace'];pre=z['pre'];post=z['post'];cols={str(k):i for i,k in enumerate(z['columns'])}
                    margin=1.4-abs(post[:,ids]).max(axis=1)
                    np.testing.assert_array_equal(margin,trace[:,cols['design_margin']]);assert margin.min()==row['min_active_design_margin_rad']
                    lengths,loop=chain_minima(post[:,:17],m)
                    np.testing.assert_allclose(lengths,[row['min_actual_A_leg_m'],row['min_actual_B_leg_m']],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(loop,row['max_loop_error_m'],rtol=0,atol=1e-12)
                    physical=physical_metrics(pre,post,m)
                    np.testing.assert_allclose(physical,[row['min_eight_joint_margin_rad'],row['max_actual_torque_excess_Nm'],row['max_command_torque_excess_Nm']],rtol=0,atol=1e-12)
                    with np.load(f.parent/row['joint_guard_trace']['path'],allow_pickle=False) as z:g=z['trace']
                    assert g.shape==(len(trace),62) and np.isfinite(g).all() and (g[:,0]==1).all()
                    np.testing.assert_array_equal(g[:,1],np.arange(1,len(g)+1));np.testing.assert_array_equal(g[:,50:54],pre[:,ids])
                    np.testing.assert_array_equal(g[:,54:58],pre[:,17+vids]);np.testing.assert_array_equal(g[:,20:26],pre[:,33:39])
                    np.testing.assert_allclose(g[1:,30:34],np.diff(g[:,54:58],axis=0)/.0005,rtol=0,atol=1e-10)
                    np.testing.assert_allclose(g[1:,38:42],g[1:,30:34]-g[:-1,34:38],rtol=0,atol=1e-10)
                    assert np.isin(g[:,28:30],[0,1]).all() and (g[:,27]>=0).all() and (g[:,27]<=1).all()
                    st['nominal_corrected']+=int((g[:,60]>1e-12).sum());st['residual_reduced']+=int((g[:,27]<1-1e-9).sum());st['infeasible']+=int((g[:,28:30]==0).any(axis=1).sum())
                    st['minimum_margin']=min(st['minimum_margin'],float(margin.min()));st['maximum_model_error']=max(st['maximum_model_error'],float(abs(g[:,38:42]).max()))
                    st['maximum_prediction_violation']=max(st['maximum_prediction_violation'],float(g[:,58].max()));st['minimum_secondary_barrier']=min(st['minimum_secondary_barrier'],float(g[:,59].min()))
                    with np.load(f.parent/row['parking_trace']['path'],allow_pickle=False) as z:parking=z['trace']
                    check_log(parking,'parking_request_withdrawal')
                    with np.load(f.parent/row['actor_trace']['path'],allow_pickle=False) as z:actor=z['trace']
                    assert actor.shape==(int(np.ceil(len(trace)/40)),968) and np.isfinite(actor).all()
                    np.testing.assert_array_equal(norm.normalize_obs(actor[:,:481].copy()),actor[:,481:962])
                    predicted=embed(model.predict(actor[:,481:962],deterministic=True)[0],arm,len(actor))
                    error=float(abs(predicted-actor[:,962:]).max());assert error<=1e-6
                    st['maximum_actor_replay_error']=max(st['maximum_actor_replay_error'],error);actor_rows+=len(actor)
                    np.testing.assert_array_equal(parking[:,5:11],actor[(np.arange(len(trace))//40),962:].astype(np.float64))
                print('RECEIVED338',label,panel,entry['batch'],'firststeps',steps,flush=True)
            assert all(r is not None for r in ordered)
            models[label,panel]=dict(runs=ordered,summary=summary(ordered),physical=sum(r['physical_safety_passed'] for r in ordered),design=sum(r['design_joint_passed'] for r in ordered))
        stats[label]=st
    assert steps==c['first_episode_world_steps'] and sum(r['actual_graph_world_steps'] for r in c['records'])==c['actual_graph_world_steps']<=a['graph_world_step_budget']
    assert len(c['FD_calls'])<=a['FD_budget'] and sum(s['episodes'] for s in stats.values())==984
    labels=[x['label'] for x in p['classical_conditions']];details={};counts={}
    for label in conditions:
        arm,seed=label.split('_');vlabel=f'V6_{seed}';detail={};counts[label]={}
        for panel in ('regular','controlled','legacy'):
            r=models[label,panel];counts[label][panel]={k:v for k,v in r.items() if k!='runs'}
            comparisons={b:retention(r['runs'],refs[b,panel]['runs'],p['nondegradation'],panel=='legacy') for b in labels}
            if panel=='legacy':comparisons['original_CPU']=retention(r['runs'],cpurows,p['nondegradation'],True)
            best=min(refs[b,panel]['summary']['mean_yaw_score_deg'] for b in labels)
            score=r['summary']['mean_yaw_score_deg'];vs=models[vlabel,panel]['summary']['mean_yaw_score_deg']
            gate=dict(all_physical_design=r['physical']==r['design']==len(r['runs']),preserve_every_classical_success=all(x['passed'] for x in comparisons.values()))
            if panel=='controlled':gate.update(success_atleast34=r['summary']['success_count']>=34,J_lower_than_all_classics=score is not None and score<best,J_lower_than_same_seedV6=score is not None and vs is not None and score<vs)
            if panel=='regular':gate.update(success_atleastbestclassic=r['summary']['success_count']>=max(refs[b,panel]['summary']['success_count'] for b in labels),J_lower_than_all_classics=score is not None and score<best)
            if panel=='legacy':gate.update(all28_success=r['summary']['success_count']==28,CPU_capability_pair_received=cpu['passed'])
            detail[panel]=dict(gates=gate,comparisons=comparisons,passed=all(gate.values()))
        details[label]=detail
    candidate=[models[f'D3_{s}','controlled']['summary']['mean_yaw_score_deg'] for s in p['seeds']]
    reference=min(refs[b,'controlled']['summary']['mean_yaw_score_deg'] for b in labels)
    v6=[models[f'V6_{s}','controlled']['summary']['mean_yaw_score_deg'] for s in p['seeds']]
    across=dict(vs_best_classic=yaw_gate(candidate,reference),vs_V6_mean=False if any(x is None for x in candidate+v6) else float(np.mean(candidate))<=min(.85*float(np.mean(v6)),float(np.mean(v6))-.05))
    passed=all(across.values()) and all(details[f'D3_{s}'][panel]['passed'] for s in p['seeds'] for panel in ('regular','controlled','legacy'))
    atomic_json(out,dict(round=338,verified=True,model_episodes=984,classical_episodes=492,unique_development_cases=164,counts=counts,details=details,across_seed_gates=across,
        candidate_qualification_passed=passed,candidate_benefit_branch_closed=not passed,formal5_admitted=False,guard_statistics=stats,first_episode_world_steps=steps,
        actual_graph_world_steps=c['actual_graph_world_steps'],evaluation_wall_s=c['wall_seconds'],actor_rows_replayed=actor_rows,raw_sha256=raw_sha,
        proposal_sha256=sha(OUT/'proposal.json'),completion_sha256=sha(OUT/'models/completion.json'),evaluation_source_sha256=sha(OUT/'evaluation_source.json'),
        model_admission_sha256=sha(OUT/'model_evaluation_admission.json'),baseline_raw_review_sha256=sha(OUT/'baseline_raw_review.json'),reviewer_sha256=sha(__file__),
        original_CPU_baseline_sha256=sha(cpuf),
        new_physics_steps=0,new_learning_samples=0,limits='Seen development cases;three paired seeds,not984independent trials. CPUvsGPU frozenactor replay <=1e-6 only,nottaskgate relaxation. Unknown modelerror/idealencoders/thinmargin exclude real safetyguarantees. Receipt is distinct from method qualification.',
        next='339 saved-failure mechanism diagnosis only;340 direction/cleanup and whole-framework review. Failed candidate never rescued by extra epochs/seeds/checkpoints.'))
    print('DONE338 full984 raw/model receipt;candidatequalification',passed,'formal5False',flush=True)


if __name__=='__main__':
    import sys
    self_check() if '--self-check' in sys.argv else run()
