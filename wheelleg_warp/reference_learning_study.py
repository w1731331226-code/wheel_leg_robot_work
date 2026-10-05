"""Main finite study lifecycle; explicit source/baseline/init admission first."""
from pathlib import Path
import json,hashlib,time,copy
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize
from reference_budget_train import OUT,proposal,make_env,agent,initial_function_check,contract as engineering_contract
from route_pilot import RouteLedger
from route_pilot_env import evaluate as classical_evaluate
from train_height_comparison import summary
from smoke_reward_training import weight_digest,check_agent
from review_yaw_sector import ROOT,sha,success
from dashboard.live_env import atomic_json
import wheelleg_sim as sim


def source_contract():
    p=json.loads((OUT/'admission_source.json').read_text());assert p['proposal_sha256']==sha(OUT/'proposal.json') and all(sha(ROOT/n)==v for n,v in p['source_sha256'].items());return p


def freeze_admission():
    e=engineering_contract();review=json.loads((OUT/'engineering_review.json').read_text());assert review['verified'] and not (OUT/'admission_source.json').exists()
    sources={**e['source_sha256'],'wheelleg_warp/reference_learning_study.py':sha(__file__)}
    atomic_json(OUT/'admission_source.json',dict(proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,engineering_review_sha256=sha(OUT/'engineering_review.json'),
        role='Source freeze for100world init and408 fixedclassical evaluations; main queue still blocked without trainer_contract.'))
    print('FROZEN pre-main admission source',flush=True)


def initialization():
    source_contract();p=proposal();torch.set_num_threads(1);records=[]
    assert not (OUT/'full_initialization.json').exists()
    for seed in p['seeds']:
        peers=[]
        for arm in p['arms']:
            raw,route,env=make_env(p,arm,seed,100,p['curriculum_milestones'])
            try:
                model=agent(p,arm,seed,env);assert initial_function_check(model);assert model.num_timesteps==model._n_updates==0
                world=hashlib.sha256()
                for a in [raw.venv.q0.numpy(),raw.venv.param.numpy(),raw.venv.k['gains'].numpy(),raw.venv.k['reference'].numpy()]:world.update(a.tobytes())
                row=dict(seed=seed,arm=arm,weight_sha256=weight_digest(model),world_sha256=world.hexdigest(),worlds=100,observations=39,dimensions=p['arms'][arm]['dimensions'],feature_function_equal=True,
                    initial_RMS_count=float(env.obs_rms.count),initial_RMS_mean=env.obs_rms.mean.tolist(),initial_RMS_var=env.obs_rms.var.tolist(),optimizer_empty=not model.policy.optimizer.state_dict()['state'],explicit_owner=hasattr(raw.venv,'_learning_budget_buffers'))
                assert row['optimizer_empty'] and row['explicit_owner'] and raw.venv.data.qpos.device.is_cuda
                if arm.startswith('L2'):peers.append(row)
                records.append(row);print('INITIALIZED',seed,arm,flush=True)
            finally:env.close()
        assert len({r['weight_sha256'] for r in peers})==len({r['world_sha256'] for r in peers})==1
        for key in ['initial_RMS_count','initial_RMS_mean','initial_RMS_var']:assert all(r[key]==peers[0][key] for r in peers)
    atomic_json(OUT/'full_initialization.json',dict(verified=True,records=records,policy_steps=0,physics_rollout_steps=0,source_contract_sha256=sha(OUT/'admission_source.json')))


def classical():
    source_contract();p=proposal();directory=OUT/'classical';directory.mkdir(exist_ok=False);records=[]
    for name in p['classical_reference_labels']:
        for panel in ['regular','controlled']:
            r=classical_evaluate(p[panel],'M3-route',classical=p['references'][name]);path=directory/f'{name}_{panel}.json';atomic_json(path,r)
            records.append(dict(label=name+'_'+panel,path='classical/'+path.name,sha256=sha(path),episodes=len(p[panel]),summary=r['summary'],physical=r['physical'],design=r['design']))
            print('CLASSICAL',name,panel,r['summary'],flush=True)
    assert sum(r['episodes'] for r in records)==408
    attainable=all(r['summary']['complete'] and r['summary']['mean_yaw_score_deg']>=.05 for r in records)
    atomic_json(OUT/'classical_admission.json',dict(verified=True,episodes=408,records=records,Jpsi_gate_numerically_attainable=attainable,
        scope='Fixed published development baselines; ceiling in success not treated as impossible because primary gate asks yaw reduction with success preservation. Numeric attainability does not show RL feasibility.',source_contract_sha256=sha(OUT/'admission_source.json')))


def admit():
    source=source_contract();init=json.loads((OUT/'full_initialization.json').read_text());c=json.loads((OUT/'classical_admission.json').read_text())
    assert init['verified'] and len(init['records'])==12 and c['verified'] and c['Jpsi_gate_numerically_attainable']
    p=proposal()
    for rec in c['records']:
        assert sha(OUT/rec['path'])==rec['sha256'];r=json.loads((OUT/rec['path']).read_text());panel=rec['label'].rsplit('_',1)[1]
        for row,case in zip(r['runs'],p[panel]):assert row['seed']==case['seed'] and row['scenario']==case['scenario'] and row['success']==success(row)
    assert not (OUT/'trainer_contract.json').exists()
    atomic_json(OUT/'trainer_contract.json',dict(proposal_sha256=sha(OUT/'proposal.json'),source_sha256=source['source_sha256'],engineering_review_sha256=sha(OUT/'engineering_review.json'),
        initialization_sha256=sha(OUT/'full_initialization.json'),classical_sha256=sha(OUT/'classical_admission.json'),
        primary_candidate='L2-room',selection='Only final200k, no arm/seed/checkpoint replacement',total_policy_budget=2400000,
        lifecycle='Continuous200k perrun; previous engineeringweights never loaded. Main interruptedrun preserves budget, no silent exact-trajectory resume. Stop on source/runtime errors.'))
    print('ADMITTED finite2.4M CUDA study;queue not automatically started',flush=True)


def verify_trainer():
    source_contract();c=json.loads((OUT/'trainer_contract.json').read_text());assert c['proposal_sha256']==sha(OUT/'proposal.json') and all(sha(ROOT/n)==v for n,v in c['source_sha256'].items());return c


def evaluation(p,arm,model,prefix,panel,zero=False):
    cfg=copy.deepcopy(p);spec=dict(cfg['arms'][arm]);cfg['training_banks']={'0':{str(k):p[panel] for k in [1,2,3]}}
    if zero:
        # Zero-leg allocation forroom finalmodels is implemented separately after main,
        # not silently folded into a performance-scoring arm here.
        raise ValueError('Use separately frozen registered zero-leg evaluator')
    raw,route,created=make_env(cfg,arm,0,len(p[panel]),[10**12,2*10**12]);env=VecNormalize.load(str(prefix)+'.pkl',created.venv);env.training=False;env.norm_reward=False
    rms=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count);before=(model.num_timesteps,model._n_updates,weight_digest(model));rows=[None]*len(p[panel])
    try:
        obs=env.reset();native=raw.venv;ids=native.ids.numpy();deadline=int(np.ceil((native.param.numpy()[:,3].max()+2)/.02))+2
        for _ in range(deadline):
            obs,_,done,infos=env.step(model.predict(obs,deterministic=True)[0]);stopped=native.stopped_q.numpy() if done.any() else None
            for w in np.flatnonzero(done):
                if rows[w] is None:
                    length=float(np.mean([sim.fk_joints(float(stopped[w,ids[2*s]]),float(stopped[w,ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    rows[w]={**p[panel][w],**{k:v for k,v in infos[w].items() if k!='terminal_observation'},'final_mean_fk_leg_m':length}
            if all(r is not None for r in rows):break
        assert all(r is not None for r in rows);np.testing.assert_array_equal(rms[0],env.obs_rms.mean);np.testing.assert_array_equal(rms[1],env.obs_rms.var);assert rms[2]==env.obs_rms.count
        assert before==(model.num_timesteps,model._n_updates,weight_digest(model))
        return dict(summary=summary(rows),physical=sum(r['physical_safety_passed'] for r in rows),design=sum(r['design_joint_passed'] for r in rows),runs=rows)
    finally:env.close()


def run_one(arm,seed):
    verify_trainer();p=proposal();torch.set_num_threads(1);directory=OUT/'runs'/arm/str(seed);directory.mkdir(parents=True,exist_ok=False)
    raw,route,env=make_env(p,arm,seed,100,p['curriculum_milestones']);model=agent(p,arm,seed,env);initial=weight_digest(model)
    match=next(r for r in json.loads((OUT/'full_initialization.json').read_text())['records'] if r['seed']==seed and r['arm']==arm);assert match['weight_sha256']==initial and initial_function_check(model)
    atomic_json(directory/'initialization.json',dict(arm=arm,seed=seed,weight_sha256=initial,admission_record=match,engineering_weights_reused=False))
    cb=RouteLedger(raw,route,directory,20000,p['arms'][arm]['dimensions']==2)
    try:
        start=time.perf_counter();model.learn(total_timesteps=200000,callback=cb);cb.save();wall=time.perf_counter()-start
        assert model.num_timesteps==200000 and model._n_updates==400 and weight_digest(model)!=initial;check_agent(model,8000)
        assert [r['policy_steps'] for r in cb.saved]==list(range(20000,200001,20000))
        atomic_json(directory/'episodes.json',dict(episodes=cb.rows,curriculum_transitions=raw.transition_log))
        for panel in ['regular','controlled']:atomic_json(directory/(panel+'_final.json'),evaluation(p,arm,model,directory/'step_200000',panel))
        atomic_json(directory/'verification.json',dict(verified=True,policy_steps=200000,ppo_epochs=400,adam_updates=8000,checkpoints=[r['policy_steps'] for r in cb.saved],completed_training_episodes=len(cb.rows),continuous_learn_wall_s=wall,initial_weight_sha256=initial,checkpoint_selection='final200k only',main_source_contract_sha256=sha(OUT/'trainer_contract.json')))
        atomic_json(directory/'progress.json',dict(status='complete',sampled_steps=200000,trained_steps=200000));print('COMPLETED MAIN',arm,seed,flush=True)
    except BaseException as e:
        atomic_json(directory/'interruption.json',dict(error=str(e),sampled_steps=model.num_timesteps,confirmed_trained_steps=cb.trained,nonresumable_as_exact_trajectory=True));raise
    finally:env.close()


def queue():
    verify_trainer();completed=[]
    for seed in proposal()['seeds']:
        for arm in proposal()['arms']:
            atomic_json(OUT/'main_progress.json',dict(status='running',completed_runs=completed,pending_arm=arm,pending_seed=seed));run_one(arm,seed);completed.append(dict(arm=arm,seed=seed))
    atomic_json(OUT/'main_progress.json',dict(status='complete',completed_runs=completed,trained_policy_steps=2400000))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze_admission','initialization','classical','admit','queue']);globals()[p.parse_args().command]()
