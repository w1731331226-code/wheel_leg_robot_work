"""Single source/engineering-qualified six-run1.2M queue and final1254 evaluation."""
import json
import time
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
import nominal_mean_training as training
import nominal_mean_evaluation as evaluation
from train_height_comparison import equal

OUT,ROOT,sha,write = training.OUT,training.ROOT,training.sha,training.atomic_json


def jobs(p):
    return [dict(seed=s,arm=a) for s in p['seeds'] for a in p['arms']]


def eval_jobs(p):
    result=[]
    for panel,key in [('regular','regular'),('controlled','controlled'),('legacy','legacy'),('auxiliary','auxiliary_cases')]:
        for batch,group in enumerate(training.light.phase.broad.batches(p[key])):
            result.append(dict(panel=panel,batch=batch,indices=[i for i,_ in group],cases=[c for _,c in group]))
    assert sum(len(j['cases']) for j in result)==209
    return result


def verify():
    p,e=training.verify_engineering();c=json.loads((OUT/'trainer_contract.json').read_text())
    assert c['proposal_sha256']==sha(OUT/'proposal.json') and c['engineering_review_sha256']==sha(OUT/'round203_engineering_review.json')
    assert c['evaluation_admission_sha256']==sha(OUT/'evaluation_admission.json') and c['main_source_sha256']==sha(__file__)
    assert all(sha(ROOT/n)==s for n,s in c['source_sha256'].items())
    assert all(sha(ROOT/n)==s for n,s in c['input_sha256'].items())
    assert c['jobs']==jobs(p) and c['eval_jobs']==eval_jobs(p)
    assert c['policy_steps']==1200000 and c['evaluation_budget']==1254
    return p


def freeze():
    assert not (OUT/'trainer_contract.json').exists()
    p,e=training.verify_engineering();v=json.loads((OUT/'round203_engineering_review.json').read_text())
    a=json.loads((OUT/'evaluation_admission.json').read_text());s=json.loads((OUT/'policy_source_contract.json').read_text())
    assert v['verified'] and a['verified'] and a['evaluation_source_admitted']
    sources=dict(s['source_sha256'])
    for name in ['nominal_mean_training.py','nominal_mean_evaluation.py','nominal_mean_main.py','test_nominal_mean_evaluation.py']:
        sources['wheelleg_warp/'+name]=sha(ROOT/'wheelleg_warp'/name)
    inputs={str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ['proposal.json','engineering_contract.json','engineering_verification.json','round203_engineering_review.json','evaluation_admission.json','round204_evaluation_unit.json']}
    inputs.update(a['reference_input_sha256'])
    write(OUT/'trainer_contract.json',dict(proposal_sha256=sha(OUT/'proposal.json'),engineering_review_sha256=sha(OUT/'round203_engineering_review.json'),
        evaluation_admission_sha256=sha(OUT/'evaluation_admission.json'),main_source_sha256=sha(__file__),source_sha256=sources,input_sha256=inputs,
        jobs=jobs(p),eval_jobs=eval_jobs(p),policy_steps=1200000,evaluation_budget=1254,reference_rows=418,
        selection='Final200000 each/all6 models;engineering notwarmstart;no intermediate scientific selection',
        lifecycle='One continuous learn/run. Save beforefinalevaluation, no silentresume/replay; allconsumed state kept. Main1.2M independent ofseparate24kengineering.'))
    verify();print('FROZEN6fresh runs1.2M+1254finalevaluations;0 mainsteps/evals consumed',flush=True)


def run_one(p,job,completed):
    import torch
    torch.set_num_threads(1)
    seed,arm=job['seed'],job['arm'];d=OUT/'runs'/arm/str(seed);d.mkdir(parents=True,exist_ok=False)
    raw,route,cache,norm,env=training.make_env(p,seed,100,p['curriculum_milestones']);model=training.new_agent(p,arm,seed,env)
    initial=training.weight_digest(model);native=raw.venv
    import hashlib
    world=hashlib.sha256()
    for a in (native.q0,native.param,native.k['gains'],native.k['reference']):world.update(a.numpy().tobytes())
    for peer in p['arms']:
        file=OUT/'runs'/peer/str(seed)/'initialization.json'
        if file.exists():
            old=json.loads(file.read_text());assert old['initial_weight_sha256']==initial and old['initial_world_sha256']==world.hexdigest()
    write(d/'initialization.json',dict(seed=seed,arm=arm,initial_weight_sha256=initial,initial_world_sha256=world.hexdigest(),engineering_warmstart=False,eta=model.policy.eta))
    cb=training.MatchedLedger(raw,route,cache,norm,d,20000)
    try:
        start=time.perf_counter();model.learn(total_timesteps=200000,callback=cb);cb.save();wall=time.perf_counter()-start
        assert model.num_timesteps==200000 and model._n_updates==400 and training.weight_digest(model)!=initial
        training.check_agent(model,8000);assert [r['policy_steps'] for r in cb.saved]==list(range(20000,200001,20000))
        assert cb.rows and np.any(raw.stages==3) and all(t['actual_episode_end'] for t in raw.transition_log)
        np.testing.assert_allclose(norm.obs_rms.count,200100.0001,atol=1e-6,rtol=0)
        final=d/'step_200000';loaded=PPO.load(final.with_suffix('.zip'),device='cuda')
        equal(model.policy.state_dict(),loaded.policy.state_dict());equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict());training.check_agent(loaded,8000)
        write(d/'episodes.json',dict(episodes=cb.rows,curriculum_transitions=raw.transition_log))
        write(d/'training_verification.json',dict(verified=True,policy_steps=200000,epochs=400,Adam=8000,initial_weight_sha256=initial,
            checkpoints=[r['policy_steps'] for r in cb.saved],continuous_training_checkpoint_wall_s=wall,trained_without_physics_restart=True,
            model_optimizer_reload_exact=True,normalization_dimensions=39,stored_observations=78,engineering_warmstart=False))
    except BaseException as error:
        model.save(d/'interrupted_training');norm.save(d/'interrupted_training.pkl')
        write(d/'training_interruption.json',dict(error=repr(error),sampled_steps=model.num_timesteps,confirmed_trained_steps=cb.trained,checkpoints=cb.saved,silently_resumable=False));raise
    finally:env.close()
    ledger=[];reserved=finished=0
    try:
        for current in eval_jobs(p):
            verify();directory=d/'evaluation'/current['panel']/f'batch_{current["batch"]}';directory.mkdir(parents=True,exist_ok=False)
            reserved+=len(current['cases'])
            write(OUT/'main_progress.json',dict(status='evaluating',completed_runs=completed,current_job=job,
                current_eval=current['panel'],current_reserved=reserved,current_completed=finished,completed_training_policy_steps=(len(completed)+1)*200000))
            result,checked=evaluation.evaluate(current['cases'],current['panel'],model,final.with_suffix('.pkl'),directory)
            finished+=len(result['runs']);ledger.append(dict(panel=current['panel'],indices=current['indices'],path=str((directory/'result.json').relative_to(OUT)),sha256=sha(directory/'result.json'),**checked))
            write(d/'evaluation_ledger.json',dict(completed=finished,records=ledger));print('EVALUATED',arm,seed,current['panel'],finished,'success',result['summary']['success_count'],flush=True)
        assert finished==reserved==209
        write(d/'verification.json',dict(verified=True,training=sha(d/'training_verification.json'),evaluation_records=ledger,final_only=True,policy_steps=200000,evaluations=209,model_RMS_immutable=True))
    except BaseException as error:
        write(d/'evaluation_interruption.json',dict(error=repr(error),reserved_evaluations=reserved,completed_evaluations=finished,records=ledger,silently_resumable=False));raise


def queue():
    p=verify();completed=[]
    try:
        for job in jobs(p):
            verify();write(OUT/'main_progress.json',dict(status='training',completed_runs=completed,current_job=job,completed_training_policy_steps=len(completed)*200000))
            run_one(p,job,completed);completed.append(job)
            write(OUT/'main_progress.json',dict(status='running',completed_runs=completed,completed_training_policy_steps=len(completed)*200000))
        write(OUT/'main_completion.json',dict(verified=True,completed_runs=completed,training_policy_steps=1200000,evaluations=1254,trainer_contract_sha256=sha(OUT/'trainer_contract.json'),formal_expansion=False))
        write(OUT/'main_progress.json',dict(status='complete',completed_runs=completed,completed_training_policy_steps=1200000,evaluations=1254))
    except BaseException as error:
        write(OUT/'main_interruption.json',dict(error=repr(error),completed_runs=completed,current_job=job,silently_resumable=False));raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','queue']);globals()[parser.parse_args().command]()
