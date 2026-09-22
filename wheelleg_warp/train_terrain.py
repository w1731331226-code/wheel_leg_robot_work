"""Terrain-v1: warm-started 1024-world curriculum with bounded selection."""
import argparse,json,os,shutil,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import torch
from stable_baselines3.common.vec_env import VecCheckNan,VecNormalize
from benchmark_parallel import TimedPPO
from dashboard.live_env import atomic_json as write
from native.live import LiveNativeEnv
from native.terrain import bank as terrain_bank,sample_terrain
from ppo_env import sample_scenario
from terrain_eval import evaluate_terrain,summarize_terrain as summarize
from training_contract import (TASK_CONTRACT_VERSION,yaw_total,promotion_allowed,checkpoint_hashes,
    verify_checkpoint,source_hashes,verify_protocol)


def score(terrain,legacy):
    return (terrain['total']+legacy['total']-terrain['success_count']-legacy['success_count'],
            legacy['total']-legacy['success_count'],yaw_total(terrain,legacy))


def stratified(start,counts,split):
    rows=[];remaining=dict(counts);seed=start
    while any(remaining.values()):
        row=sample_terrain(seed,3,split);seed+=1
        if remaining.get(row.terrain,0):rows.append(row);remaining[row.terrain]-=1
    return rows


def initialize(output,source):
    selection=json.loads((source/'selection.json').read_text());best=selection['best'];origin=Path(best['path'])
    source_protocol=json.loads((source/'protocol.json').read_text());assert json.loads((source/'status.json').read_text())['status']=='completed'
    continuation=source_protocol.get('name')=='terrain-v1'
    if continuation:
        assert best['terrain']['success_count']>=24 and best['legacy']['success_count']>=13
    else:assert best['summary']['success_count']==32
    output.mkdir(parents=True,exist_ok=False);(output/'bootstrap').mkdir()
    for ext in ('.zip','.pkl'):shutil.copy2(str(origin)+ext,output/'bootstrap'/('policy'+ext))
    namespace=230000 if continuation else 130000
    development=stratified(namespace,dict(legacy=8,ramp=5,cross_slope=5,rough=5,step=5,mixed=4),'development')
    from native.terrain import TerrainScenario
    legacy=[TerrainScenario(**asdict(sample_scenario('test_iid',namespace+10000+i,3)),terrain='legacy',terrain_seed=namespace+10000+i) for i in range(16)]
    ood=stratified(namespace+20000,dict(ramp=13,cross_slope=13,rough=13,step=13,mixed=12),'ood')
    checkpoint=output/'bootstrap/policy'
    inspected=TimedPPO.load(str(checkpoint)+'.zip',device='cpu')
    protocol=dict(name='terrain-v2' if continuation else 'terrain-v1',created=time.time(),environments=1024,n_steps=50,steps_per_round=1024000,
        max_rounds=4 if continuation else 8,patience=2 if continuation else 3,continuation_from=source_protocol.get('name'),
        inherited_steps=inspected.num_timesteps,bootstrap_summary=best['summary'],curriculum={1:'full mix continuation'} if continuation else {1:'legacy+ramp+cross_slope',2:'add rough+step',3:'full mix'},
        terrain_ranges=dict(ramp_deg=[1,3],cross_slope_deg=[-3,3],roughness_mm=[2,6],step_mm=[5,20],legacy_replay_fraction=.3),
        development_cases=[asdict(s) for s in development],legacy_regression_cases=[asdict(s) for s in legacy],
        final_ood_cases=[asdict(s) for s in ood],
        selection='complete candidates only; preserve every successful paired legacy case; then combined failures, legacy failures and Jpsi',
        final_evaluation='only after stopping: frozen 64 OOD terrain cases plus frozen 16 legacy regression cases; never used to continue training',
        known_limits=['terrain geometry is boxes: ramp/plateau/ramp, split-level cross slope, tiled roughness and full-width step',
                      'terrain-v2 leaves a clear central obstacle corridor in mixed terrain; terrain-v1 mixed OOD is invalid because rough tiles could cover low bumps',
                      'training slopes limited to 3 degrees because the frozen success gate uses world-frame 5-degree attitude',
                      'OOD slopes extend to 5 degrees; steeper terrain requires terrain-relative attitude metrics before training'],
        task_contract_version=TASK_CONTRACT_VERSION,checkpoint=str(checkpoint),
        source_sha256=source_hashes(__file__),**checkpoint_hashes(checkpoint))
    write(output/'protocol.json',protocol);write(output/'status.json',dict(status='initial_evaluation',round=0,new_steps=0))
    torch.set_num_threads(1);model=TimedPPO.load(str(output/'bootstrap/policy.zip'),device='cpu')
    tr=evaluate_terrain(model,output/'bootstrap/policy.pkl',development);lr=evaluate_terrain(model,output/'bootstrap/policy.pkl',legacy)
    initial=dict(round=0,path=str(checkpoint),policy_steps=model.num_timesteps,**checkpoint_hashes(checkpoint),terrain=summarize(tr,[s.terrain_seed for s in development]),
                 legacy=summarize(lr,[s.terrain_seed for s in legacy]),terrain_runs=tr,legacy_runs=lr)
    initial['summary']=initial['terrain'];protocol['bootstrap_summary']=initial['terrain'];write(output/'protocol.json',protocol)
    verify_protocol(output,__file__,protocol)
    write(output/'initial.json',initial);write(output/'selection.json',dict(best=initial,anchor=initial,stagnant_rounds=0,rounds=[]))
    write(output/'status.json',dict(status='ready',round=0,new_steps=0,initial_score=score(initial['terrain'],initial['legacy']) if initial['terrain']['complete'] and initial['legacy']['complete'] else None))


def protocol(output):
    return verify_protocol(output,__file__)


def cases(items):
    from native.terrain import TerrainScenario
    return [TerrainScenario(**x) for x in items]


def run_round(output,index):
    p=protocol(output);selection=json.loads((output/'selection.json').read_text());previous=Path(selection['best']['path'])
    verify_checkpoint(previous,selection['best'])
    directory=output/f'round_{index:03d}';directory.mkdir(exist_ok=False);stage=min(index,3)
    model=TimedPPO.load(str(previous)+'.zip',device='cpu');model.timings=[];start_step=model.num_timesteps;torch.set_num_threads(1);started=time.perf_counter()
    if p.get('continuation_from'):stage=3
    train_namespace=260000 if p.get('continuation_from') else 160000
    raw=LiveNativeEnv(directory/'live',start_steps=(index-1)*p['steps_per_round'],n=1024,stage=stage,
                      seed=train_namespace+index*1024,phase='terrain_training',bank_factory=terrain_bank)
    env=VecNormalize.load(str(previous)+'.pkl',VecCheckNan(raw,raise_exception=True));env.training=True;env.norm_reward=False
    model.set_env(env);model.set_random_seed(730000+index);records=[];train_seconds=0.
    write(directory/'run_config.json',dict(round=index,stage=stage,resume_from=str(previous),start_policy_steps=start_step,
          terrain_seed=train_namespace+index*1024,exploration_seed=730000+index,exact_trajectory_resume=False))
    try:
        for extra in (p['steps_per_round']//2,p['steps_per_round']):
            verify_protocol(output,__file__,p);verify_checkpoint(previous,selection['best'])
            write(output/'status.json',dict(status='training',round=index,new_steps=(index-1)*p['steps_per_round']+extra//2,stage=stage))
            start=time.perf_counter();model.learn(total_timesteps=start_step+extra-model.num_timesteps,reset_num_timesteps=False);train_seconds+=time.perf_counter()-start
            assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
            prefix=directory/f'step_{model.num_timesteps}';model.save(prefix);env.save(str(prefix)+'.pkl')
            artifacts=checkpoint_hashes(prefix)
            write(output/'status.json',dict(status='evaluating',round=index,new_steps=(index-1)*p['steps_per_round']+extra,stage=stage))
            dev=cases(p['development_cases']);legacy=cases(p['legacy_regression_cases'])
            tr=evaluate_terrain(model,str(prefix)+'.pkl',dev);lr=evaluate_terrain(model,str(prefix)+'.pkl',legacy)
            record=dict(round=index,path=str(prefix),**artifacts,policy_steps=model.num_timesteps,round_steps=extra,train_seconds=train_seconds,
                terrain=summarize(tr,[s.terrain_seed for s in dev]),legacy=summarize(lr,[s.terrain_seed for s in legacy]),terrain_runs=tr,legacy_runs=lr)
            record.update(summary=record['terrain'],total_seconds=time.perf_counter()-started,updates=len(model.timings))
            verify_protocol(output,__file__,p);verify_checkpoint(prefix,record)
            records.append(record);write(directory/(prefix.name+'.json'),record)
        for record in records:record['promotion_eligible']=promotion_allowed(record,selection['best'])
        chosen=min([selection['best']]+[r for r in records if r['promotion_eligible']],key=lambda x:score(x['terrain'],x['legacy']))
        verify_protocol(output,__file__,p);verify_checkpoint(previous,selection['best'])
        write(directory/'completed.json',dict(passed=True,round=index,stage=stage,records=records,selected=chosen,
              last=records[-1],train_seconds=train_seconds,total_seconds=time.perf_counter()-started))
    except Exception as exc:write(directory/'failed.json',dict(error=repr(exc),policy_steps=model.num_timesteps));raise
    finally:env.close()


def orchestrate(output):
    p=protocol(output);selection=json.loads((output/'selection.json').read_text());assert not selection['rounds']
    for index in range(1,p['max_rounds']+1):
        write(output/'status.json',dict(status='initializing',round=index,new_steps=(index-1)*p['steps_per_round']))
        with (output/f'round_{index:03d}.log').open('x') as log:
            child=subprocess.run([sys.executable,'-u',str(Path(__file__)),'round','--output',str(output),'--round',str(index)],stdout=log,stderr=subprocess.STDOUT)
        if child.returncode:raise RuntimeError(f'round {index} failed')
        completed=json.loads((output/f'round_{index:03d}/completed.json').read_text());row=completed['selected']
        selection['rounds'].append({**row,'round':index,'selected_source_round':row['round'],'stage':completed['stage'],
            'round_steps':completed['last']['round_steps'],'last_trained_policy_steps':completed['last']['policy_steps'],
            'train_seconds':completed['train_seconds'],'total_seconds':completed['total_seconds'],'updates':completed['last']['updates']})
        old=score(selection['best']['terrain'],selection['best']['legacy']);new=score(row['terrain'],row['legacy'])
        if new<old and promotion_allowed(row,selection['best']):selection['best']=row
        anchor=score(selection['anchor']['terrain'],selection['anchor']['legacy'])
        if new<anchor and promotion_allowed(row,selection['anchor']):selection['anchor']=row;selection['stagnant_rounds']=0
        else:selection['stagnant_rounds']+=1
        write(output/'selection.json',selection)
        if selection['stagnant_rounds']>=p['patience']:break
    verify_protocol(output,__file__,p)
    best=selection['best'];verify_checkpoint(best['path'],best);model=TimedPPO.load(best['path']+'.zip',device='cpu')
    write(output/'status.json',dict(status='final_evaluation',round=index,new_steps=index*p['steps_per_round']))
    ood=cases(p['final_ood_cases']);legacy=cases(p['legacy_regression_cases'])
    tr=evaluate_terrain(model,best['path']+'.pkl',ood);lr=evaluate_terrain(model,best['path']+'.pkl',legacy)
    terrain_summary=summarize(tr,[s.terrain_seed for s in ood])
    write(output/'final_evaluation.json',dict(checkpoint=best,summary=terrain_summary,terrain=terrain_summary,
          legacy=summarize(lr,[s.terrain_seed for s in legacy]),terrain_runs=tr,legacy_runs=lr,used_for_selection=False))
    verify_protocol(output,__file__,p);verify_checkpoint(best['path'],best)
    write(output/'status.json',dict(status='completed',round=index,new_steps=index*p['steps_per_round'],best=best,
          stop_reason='plateau' if selection['stagnant_rounds']>=p['patience'] else 'round_budget'))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['init','round','orchestrate']);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--source',type=Path);a.add_argument('--round',type=int);args=a.parse_args();out=args.output.resolve()
    if args.command=='init':initialize(out,args.source.resolve())
    elif args.command=='round':run_round(out,args.round)
    else:
        try:orchestrate(out)
        except Exception as exc:
            old=json.loads((out/'status.json').read_text());write(out/'status.json',{**old,'status':'failed','error':repr(exc)});raise
