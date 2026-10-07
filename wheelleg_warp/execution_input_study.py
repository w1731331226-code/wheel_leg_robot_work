"""One admitted six-run scientific information contrast;not formal5."""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
import execution_history_engineering as engine
import execution_history_evaluation as evaluation
from execution_history_env import ExecutionHistory
from route_state import RouteState
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json as write

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/execution_input_history_v1'


def verify():
    p=json.loads((OUT/'study_proposal.json').read_text());c=json.loads((OUT/'study_contract.json').read_text())
    assert c['proposal_sha256']==sha(OUT/'study_proposal.json') and c['admission_sha256']==sha(OUT/'evaluation_admission.json')
    assert all(sha(ROOT/n)==v for n,v in c['source_sha256'].items())
    assert all(sha(ROOT/n)==v for n,v in c['reference_input_sha256'].items())
    assert c['policy_samples']==p['total_policy_samples']==1200000 and c['evaluations']==p['evaluation_budget']==1312
    return p,c


def freeze():
    assert not (OUT/'study_contract.json').exists()
    p=json.loads((OUT/'study_proposal.json').read_text());a=json.loads((OUT/'evaluation_admission.json').read_text())
    assert a['verified'] and a['evaluation_source_admitted'] and a['proposal_sha256']==sha(OUT/'study_proposal.json')
    sources={str(Path(m.__file__).resolve().relative_to(ROOT)):sha(m.__file__) for m in list(sys.modules.values())
             if getattr(m,'__file__',None) and Path(m.__file__).is_file() and Path(m.__file__).resolve().is_relative_to(ROOT)
             and not Path(m.__file__).resolve().is_relative_to(ROOT/'.venv') and str(m.__file__).endswith('.py')}
    from ppo_env import XML
    sources[str(Path(XML).relative_to(ROOT))]=sha(XML)
    write(OUT/'study_contract.json',dict(proposal_sha256=sha(OUT/'study_proposal.json'),admission_sha256=sha(OUT/'evaluation_admission.json'),
        source_sha256=sources,reference_input_sha256=p['reference_input_sha256'],policy_samples=1200000,evaluations=1312,
        jobs=p['jobs'],eval_jobs=p['eval_jobs'],classical=p['new_classical'],worker='wheelleg-execution-input-study-v1.service',
        lifecycle='Onecontinuous200kfreshlearn/run,final-only164evaluation;noengineeringwarmstart,implicitrestart,seedselection orformal5.',
        interruption='Keepreserved/completedcounts,allcheckpoints/Adam/RMS/episodes,history/command/parking/worldbuffers andrawprefixes. No exacttrajectoryresume.',
        study_admitted=True,formal5_admitted=False,entire_goal_complete=False))
    verify();print('FROZEN2341.2M/1312scientific;formal5notadmitted',flush=True)


def make_env(p,job):
    factory=engine.base.raw_env
    engine.base.raw_env=lambda rows,mode:engine.collector.instrument(lambda a,b:engine.parking.instrument(factory,a,b),rows,mode)
    try:curriculum=engine.base.CurriculumEnv(p,'virtual6',job['seed'],100,milestones=p['curriculum_milestones'])
    finally:engine.base.raw_env=factory
    raw=curriculum.venv;route=RouteState(curriculum);history=ExecutionHistory(route,raw,job['arm'])
    norm=VecNormalize(VecCheckNan(history,raise_exception=True),**p['normalization'])
    return curriculum,raw,route,history,norm


def train(p,job):
    directory=OUT/'study_runs'/job['arm']/str(job['seed']);directory.mkdir(parents=True,exist_ok=False)
    curriculum,raw,route,history,norm=make_env(p,job);kwargs=dict(p['ppo']);policy=dict(kwargs.pop('policy_kwargs'))
    model=PPO('MlpPolicy',norm,seed=job['seed'],device='cuda',policy_kwargs=policy,**kwargs)
    with torch.no_grad():
        model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
        model.policy.log_std.copy_(torch.tensor(p['initial_log_std'],device='cuda'))
    initial=engine.weight_digest(model);world=hashlib.sha256()
    for b in (raw.q0,raw.param,raw.k['gains'],raw.k['reference']):world.update(b.numpy().tobytes())
    for arm in p['arms']:
        f=OUT/'study_runs'/arm/str(job['seed'])/'initialization.json'
        if f.exists():
            peer=json.loads(f.read_text());assert peer['initial_weights']==initial and peer['initial_world']==world.hexdigest()
    write(directory/'initialization.json',dict(**job,initial_weights=initial,initial_world=world.hexdigest(),engineering_warmstart=False))
    cb=engine.HistoryLedger(curriculum,route,history,norm,directory);cb.interval=20000
    try:
        start=time.perf_counter();model.learn(total_timesteps=200000,callback=cb);cb.save();wall=time.perf_counter()-start
        assert model.num_timesteps==200000 and model._n_updates==400;engine.check_agent(model,8000)
        assert engine.weight_digest(model)!=initial and [r['policy_steps'] for r in cb.saved]==list(range(20000,200001,20000))
        assert cb.rows and np.any(curriculum.stages==3)
        np.testing.assert_allclose(norm.obs_rms.count,200100.0001,atol=1e-6,rtol=0)
        final=directory/'step_200000';loaded=PPO.load(final.with_suffix('.zip'),device='cuda')
        engine.base.equal(model.policy.state_dict(),loaded.policy.state_dict());engine.base.equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict());engine.check_agent(loaded,8000)
        with final.with_suffix('.pkl').open('rb') as f:restored=pickle.load(f)
        for rms in ('obs_rms','ret_rms'):
            for field in ('mean','var','count'):engine.base.equal(getattr(getattr(norm,rms),field),getattr(getattr(restored,rms),field))
        write(directory/'episodes.json',dict(episodes=cb.rows,transitions=curriculum.transition_log))
        write(directory/'training_verification.json',dict(verified=True,policy_samples=200000,epochs=400,Adam=8000,
            continuous_learn_checkpoint_wall_s=wall,model_optimizer_481RMS_reload_exact=True,engineering_warmstart=False,parking=raw._light_parking_stats))
        return loaded,final.with_suffix('.pkl'),directory
    except BaseException as e:
        model.save(directory/'interrupted_model.zip');norm.save(str(directory/'interrupted_rms.pkl'))
        engine.collector.preserve(raw,directory/'interrupted_command.npz')
        np.savez_compressed(directory/'interrupted_history_world.npz',frames=history.frames,inputs=history.inputs,
            elapsed=history.elapsed,valid=history.valid,q=raw.data.qpos.numpy(),v=raw.data.qvel.numpy(),state=raw.state.numpy(),controller=raw.k['state'].numpy())
        write(directory/'training_interruption.json',dict(error=repr(e),sampled=model.num_timesteps,trained=cb.trained,checkpoints=cb.saved,implicit_resume=False));raise
    finally:norm.close()


def run():
    torch.set_num_threads(1);p,c=verify();assert not (OUT/'study_progress.json').exists()
    completed=[];records=[];reserved=finished=0;current=None;start=time.perf_counter()
    conditions=[dict(arm=a,seed=None) for a in p['new_classical']]+p['jobs']
    try:
        for current in conditions:
            verify();model=normalization=None
            if current['seed'] is not None:
                write(OUT/'study_progress.json',dict(status='training',current=current,completed=completed,completed_evaluations=finished))
                model,normalization,directory=train(p,current)
            else:directory=OUT/'study_reference'/current['arm'];directory.mkdir(parents=True,exist_ok=False)
            for job in p['eval_jobs']:
                verify();d=directory/'evaluation'/job['panel']/f'batch_{job["batch"]}';d.mkdir(parents=True,exist_ok=False)
                reserved+=len(job['cases'])
                write(OUT/'study_progress.json',dict(status='evaluating',current=current,panel=job['panel'],batch=job['batch'],
                    reserved_evaluations=reserved,completed_evaluations=finished,completed=completed))
                result,checked=evaluation.evaluate(job['cases'],current['arm'],model,normalization,d,p['classical'].get(current['arm']))
                finished+=len(job['cases']);records.append(dict(**current,panel=job['panel'],batch=job['batch'],indices=job['indices'],
                    path=str((d/'result.json').relative_to(OUT)),sha256=sha(d/'result.json'),**checked))
                write(OUT/'study_evaluation_ledger.json',dict(reserved=reserved,completed=finished,records=records))
                print('STUDY EVALUATED',current,job['panel'],finished,flush=True)
            completed.append(current)
        assert finished==reserved==1312 and completed==conditions
        verify();write(OUT/'study_completion.json',dict(verified=True,completed=completed,policy_samples=1200000,evaluations=1312,
            records=records,contract_sha256=sha(OUT/'study_contract.json'),queue_wall_s=time.perf_counter()-start,formal5_admitted=False))
        write(OUT/'study_progress.json',dict(status='complete',completed_evaluations=1312,policy_samples=1200000))
        print('COMPLETE scientificstudy;fullgate review stillrequired',flush=True)
    except BaseException as e:
        write(OUT/'study_interruption.json',dict(error=repr(e),current=current,reserved_evaluations=reserved,
            completed_evaluations=finished,completed=completed,records=records,implicit_resume=False));raise


if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
