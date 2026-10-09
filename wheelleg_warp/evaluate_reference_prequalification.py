"""Fixed final-checkpoint1476-episode queue; no training or checkpoint selection."""
import json
import time
import mujoco
import warp as wp
from stable_baselines3 import PPO
from train_reference_prequalification import OUT,verify as training_sources
from run_joint_reference_zero_pair import evaluate
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def conditions(p):
    fixed=[dict(label=arm,arm=arm,seed=None) for arm in p['classical_conditions']]
    learned=[dict(label=f'{arm}_{seed}',arm=arm,seed=seed) for seed in p['seeds'] for arm in p['arms']]
    assert len(fixed)==3 and len(learned)==6
    assert sum(len(j['cases']) for j in p['eval_jobs'])*len(fixed+learned)==1476
    return fixed+learned


def self_check():
    p=json.loads((OUT/'proposal.json').read_text());items=conditions(p)
    assert [c['label'] for c in items[:3]]==['old_B0','old_B1_route','joint_yaw']
    assert {c['label'] for c in items[3:]}=={f'{a}_{s}' for a in ('M_ref3','U_ref6') for s in (32031,32032,32033)}
    assert {panel:sum(len(j['cases']) for j in p['eval_jobs'] if j['panel']==panel) for panel in ('regular','controlled','legacy')}=={'regular':96,'controlled':40,'legacy':28}
    print('PASS323 evaluator schedule90batches/1476episodes/fixedfinal6models;0physics')


def admit():
    p=training_sources();done=json.loads((OUT/'training/completion.json').read_text())
    assert done['verified'] and done['policy_samples']==1200000 and len(done['reports'])==6
    assert not (OUT/'evaluation_admission.json').exists()
    sources=json.loads((OUT/'main_source_admission.json').read_text())['source_sha256']
    for f in ('wheelleg_warp/evaluate_reference_prequalification.py','wheelleg_warp/joint_reference_classical.py'):
        sources[f]=sha(ROOT/f)
    yaw=OUT.parent/'nonzero_reference_mechanism_v1/proposal.json';sources[str(yaw.relative_to(ROOT))]=sha(yaw)
    models={}
    for condition in conditions(p):
        if condition['seed'] is None:continue
        directory=OUT/'training'/condition['label'];record=json.loads((directory/'verification.json').read_text())
        assert record['reload_verified'] and record['policy_samples']==200000
        models[condition['label']]={}
        for name in ('step_200000.zip','step_200000.pkl','verification.json'):
            f=directory/name;models[condition['label']][name]=sha(f);sources[str(f.relative_to(ROOT))]=sha(f)
    atomic_json(OUT/'evaluation_admission.json',dict(verified=True,proposal_sha256=sha(OUT/'proposal.json'),training_completion_sha256=sha(OUT/'training/completion.json'),
        source_sha256=sources,models=models,yaw_config=json.loads(yaw.read_text())['yaw_config'],conditions=conditions(p),
        episodes=1476,world_step_budget=50000000,constructor_FD_budget=1000,model_updates=0,formal5_admitted=False))


def run():
    p=training_sources();a=json.loads((OUT/'evaluation_admission.json').read_text());assert a['verified'] and a['proposal_sha256']==sha(OUT/'proposal.json')
    assert a['training_completion_sha256']==sha(OUT/'training/completion.json')
    directory=OUT/'scientific_evaluation';directory.mkdir(exist_ok=False)
    records=[];calls=0;steps=0;worlds=0;fd_calls=[];current=None;start=time.perf_counter()
    launch=wp.capture_launch;fd=mujoco.mjd_transitionFD
    def capture(*args,**kwargs):
        nonlocal calls,steps
        calls+=1;steps+=40*worlds;assert steps<=a['world_step_budget']
        return launch(*args,**kwargs)
    def counted(*args,**kwargs):
        fd_calls.append(float(args[2]));assert len(fd_calls)<=a['constructor_FD_budget']
        return fd(*args,**kwargs)
    wp.capture_launch=capture;mujoco.mjd_transitionFD=counted
    atomic_json(directory/'started.json',dict(admission_sha256=sha(OUT/'evaluation_admission.json')))
    try:
        for condition in a['conditions']:
            model=None;normalization=None
            if condition['seed'] is not None:
                model=PPO.load(OUT/'training'/condition['label']/'step_200000.zip',device='cuda')
                normalization=OUT/'training'/condition['label']/'step_200000.pkl'
            for job in p['eval_jobs']:
                assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
                current=dict(condition=condition['label'],panel=job['panel'],batch=job['batch'])
                dest=directory/condition['label']/job['panel']/str(job['batch']);dest.mkdir(parents=True,exist_ok=False)
                worlds=len(job['cases']);before=steps
                result,checked=evaluate(job['cases'],condition['arm'],dest,a['world_step_budget']-steps,a['yaw_config'],model=model,normalization=normalization)
                assert steps-before==result['actual_graph_world_steps']
                entry=dict(**current,path=str((dest/'result.json').relative_to(directory)),sha256=sha(dest/'result.json'),indices=job['indices'],
                    episodes=worlds,actual_graph_world_steps=result['actual_graph_world_steps'],first_episode_world_steps=result['first_episode_world_steps'],checked=checked)
                records.append(entry)
                atomic_json(directory/'progress.json',dict(records=records,episodes=sum(r['episodes'] for r in records),actual_graph_world_steps=steps))
        assert sum(r['episodes'] for r in records)==1476 and len(records)==90
        atomic_json(directory/'completion.json',dict(verified=True,records=records,episodes=1476,actual_graph_world_steps=steps,actual_capture_calls=calls,
            first_episode_world_steps=sum(r['first_episode_world_steps'] for r in records),constructor_FD_calls=fd_calls,wall_seconds=time.perf_counter()-start,model_updates=0,formal5_admitted=False))
    except BaseException as error:
        atomic_json(directory/'failure.json',dict(error=repr(error),current=current,records=records,actual_graph_world_steps=steps,FD_calls=fd_calls,implicit_retry=False));raise
    finally:wp.capture_launch=launch;mujoco.mjd_transitionFD=fd


if __name__=='__main__':
    import sys
    {'check':self_check,'admit':admit,'run':run}[sys.argv[1]]()
