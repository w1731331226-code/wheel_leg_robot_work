"""Formal 1024-world terrain-v3 training on corrected MuJoCo Warp geometry."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import torch
from stable_baselines3.common.vec_env import VecCheckNan,VecNormalize
from benchmark_parallel import TimedPPO
from dashboard.live_env import atomic_json as write
from native.live import LiveNativeEnv
from native.terrain import TerrainScenario,bank_v3
from terrain_eval import evaluate_terrain,summarize_terrain as summarize
from training_contract import (TASK_CONTRACT_VERSION,yaw_total,promotion_allowed,checkpoint_hashes,
    verify_checkpoint,source_hashes,verify_protocol)


def cases(items):return [TerrainScenario(**x) for x in items]


def score(row):
    terrain=row['terrain'];legacy=row['legacy'];mixed=[x for x in row['terrain_runs'] if x['scenario']['terrain']=='mixed']
    return (terrain['total']+legacy['total']-terrain['success_count']-legacy['success_count'],legacy['total']-legacy['success_count'],sum(not x['success'] for x in mixed),yaw_total(terrain,legacy))


def verify(output):
    return verify_protocol(output,__file__)


def initialize(output,source,candidate):
    output.mkdir(parents=True,exist_ok=False);(output/'bootstrap').mkdir();checkpoint=output/'bootstrap/policy'
    for ext in ('.zip','.pkl'):shutil.copy2(str(source)+ext,str(checkpoint)+ext)
    protocol=json.loads((candidate/'protocol.json').read_text());model=TimedPPO.load(str(checkpoint)+'.zip',device='cpu')
    protocol.update(name='terrain-v3',status='initial_evaluation',inherited_steps=model.num_timesteps,continuation_from='terrain-v2',
        task_contract_version=TASK_CONTRACT_VERSION,checkpoint=str(checkpoint),source_sha256=source_hashes(__file__),
        selection='complete candidates only; preserve every successful paired legacy case; then combined failures, legacy failures, mixed failures and Jpsi',
        **checkpoint_hashes(checkpoint))
    write(output/'protocol.json',protocol);write(output/'status.json',dict(status='initial_evaluation',round=0,new_steps=0))
    # Reuse frozen scenario definitions, never historical scores from another checkpoint or task contract.
    development=cases(protocol['development_cases']);legacy_cases=cases(protocol['legacy_regression_cases'])
    terrain_rows=evaluate_terrain(model,str(checkpoint)+'.pkl',development);legacy_rows=evaluate_terrain(model,str(checkpoint)+'.pkl',legacy_cases)
    initial=dict(round=0,path=str(checkpoint),**checkpoint_hashes(checkpoint),policy_steps=model.num_timesteps,
        terrain=summarize(terrain_rows,[x.terrain_seed for x in development]),legacy=summarize(legacy_rows,[x.terrain_seed for x in legacy_cases]),
        terrain_runs=terrain_rows,legacy_runs=legacy_rows)
    initial['summary']=initial['terrain'];protocol.update(status='ready',bootstrap_summary=initial['terrain'])
    write(output/'protocol.json',protocol);verify_protocol(output,__file__,protocol)
    write(output/'initial.json',initial);write(output/'selection.json',dict(best=initial,anchor=initial,stagnant_rounds=0,rounds=[]));write(output/'status.json',dict(status='ready',round=0,new_steps=0,initial_score=score(initial) if initial['terrain']['complete'] and initial['legacy']['complete'] else None))


def run_round(output,index):
    protocol=verify(output);selection=json.loads((output/'selection.json').read_text());previous=Path(selection['best']['path']);directory=output/f'round_{index:03d}';directory.mkdir(exist_ok=False)
    verify_checkpoint(previous,selection['best'])
    stage=min((index+1)//2,3);model=TimedPPO.load(str(previous)+'.zip',device='cpu');model.timings=[];start_step=model.num_timesteps;started=time.perf_counter();torch.set_num_threads(1)
    raw=LiveNativeEnv(directory/'live',start_steps=(index-1)*protocol['steps_per_round'],n=1024,stage=stage,seed=400000+index*1024,phase='terrain_v3_training',bank_factory=bank_v3)
    env=VecNormalize.load(str(previous)+'.pkl',VecCheckNan(raw,raise_exception=True));env.training=True;env.norm_reward=False;model.set_env(env);model.set_random_seed(930000+index);records=[];train_seconds=0.
    write(directory/'run_config.json',dict(round=index,stage=stage,resume_from=str(previous),start_policy_steps=start_step,terrain_seed=400000+index*1024,exploration_seed=930000+index,exact_trajectory_resume=False))
    try:
        for additional in (protocol['steps_per_round']//2,protocol['steps_per_round']):
            verify_protocol(output,__file__,protocol);verify_checkpoint(previous,selection['best'])
            write(output/'status.json',dict(status='training',round=index,new_steps=(index-1)*protocol['steps_per_round']+additional//2,stage=stage))
            start=time.perf_counter();model.learn(total_timesteps=start_step+additional-model.num_timesteps,reset_num_timesteps=False);train_seconds+=time.perf_counter()-start
            assert all(torch.isfinite(value).all() for value in model.policy.state_dict().values())
            checkpoint=directory/f'step_{model.num_timesteps}';model.save(checkpoint);env.save(str(checkpoint)+'.pkl')
            artifacts=checkpoint_hashes(checkpoint)
            write(output/'status.json',dict(status='evaluating',round=index,new_steps=(index-1)*protocol['steps_per_round']+additional,stage=stage))
            development=cases(protocol['development_cases']);legacy_cases=cases(protocol['legacy_regression_cases']);terrain_rows=evaluate_terrain(model,str(checkpoint)+'.pkl',development);legacy_rows=evaluate_terrain(model,str(checkpoint)+'.pkl',legacy_cases)
            record=dict(round=index,path=str(checkpoint),**artifacts,policy_steps=model.num_timesteps,round_steps=additional,train_seconds=train_seconds,terrain=summarize(terrain_rows,[x.terrain_seed for x in development]),legacy=summarize(legacy_rows,[x.terrain_seed for x in legacy_cases]),terrain_runs=terrain_rows,legacy_runs=legacy_rows,total_seconds=time.perf_counter()-started,updates=len(model.timings));record['summary']=record['terrain']
            verify_protocol(output,__file__,protocol);verify_checkpoint(checkpoint,record)
            records.append(record);write(directory/(checkpoint.name+'.json'),record)
        for record in records:record['promotion_eligible']=promotion_allowed(record,selection['best'])
        chosen=min([selection['best']]+[r for r in records if r['promotion_eligible']],key=score)
        verify_protocol(output,__file__,protocol);verify_checkpoint(previous,selection['best'])
        write(directory/'completed.json',dict(passed=True,round=index,stage=stage,records=records,selected=chosen,last=records[-1],train_seconds=train_seconds,total_seconds=time.perf_counter()-started))
    except Exception as exc:write(directory/'failed.json',dict(error=repr(exc),policy_steps=model.num_timesteps));raise
    finally:env.close()


def orchestrate(output):
    protocol=verify(output);selection=json.loads((output/'selection.json').read_text());assert not selection['rounds']
    for index in range(1,protocol['max_rounds']+1):
        write(output/'status.json',dict(status='initializing',round=index,new_steps=(index-1)*protocol['steps_per_round']))
        with (output/f'round_{index:03d}.log').open('x') as log:child=subprocess.run([sys.executable,'-u',str(Path(__file__)),'round','--output',str(output),'--round',str(index)],stdout=log,stderr=subprocess.STDOUT)
        if child.returncode:raise RuntimeError(f'round {index} failed')
        completed=json.loads((output/f'round_{index:03d}/completed.json').read_text());row=completed['selected']
        selection['rounds'].append({**row,'round':index,'selected_source_round':row['round'],'stage':completed['stage'],
            'round_steps':completed['last']['round_steps'],'last_trained_policy_steps':completed['last']['policy_steps'],
            'train_seconds':completed['train_seconds'],'total_seconds':completed['total_seconds'],'updates':completed['last']['updates']})
        if score(row)<score(selection['best']) and promotion_allowed(row,selection['best']):selection['best']=row
        if score(row)<score(selection['anchor']) and promotion_allowed(row,selection['anchor']):selection['anchor']=row;selection['stagnant_rounds']=0
        else:selection['stagnant_rounds']+=1
        write(output/'selection.json',selection)
        if selection['stagnant_rounds']>=protocol['patience']:break
    verify_protocol(output,__file__,protocol)
    best=selection['best'];verify_checkpoint(best['path'],best);model=TimedPPO.load(best['path']+'.zip',device='cpu');write(output/'status.json',dict(status='final_evaluation',round=index,new_steps=index*protocol['steps_per_round']))
    holdout=cases(protocol['final_holdout_cases']);legacy_cases=cases(protocol['legacy_regression_cases']);terrain_rows=evaluate_terrain(model,best['path']+'.pkl',holdout);legacy_rows=evaluate_terrain(model,best['path']+'.pkl',legacy_cases)
    write(output/'final_evaluation.json',dict(checkpoint=best,terrain=summarize(terrain_rows,[x.terrain_seed for x in holdout]),legacy=summarize(legacy_rows,[x.terrain_seed for x in legacy_cases]),terrain_runs=terrain_rows,legacy_runs=legacy_rows,used_for_selection=False))
    verify_protocol(output,__file__,protocol);verify_checkpoint(best['path'],best)
    write(output/'status.json',dict(status='completed',round=index,new_steps=index*protocol['steps_per_round'],best=best,stop_reason='plateau' if selection['stagnant_rounds']>=protocol['patience'] else 'round_budget'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('init','round','orchestrate'));parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source',type=Path);parser.add_argument('--candidate',type=Path);parser.add_argument('--round',type=int);args=parser.parse_args();output=args.output.resolve()
    if args.command=='init':initialize(output,args.source.resolve(),args.candidate.resolve())
    elif args.command=='round':run_round(output,args.round)
    else:
        try:orchestrate(output)
        except Exception as exc:
            old=json.loads((output/'status.json').read_text());write(output/'status.json',{**old,'status':'failed','error':repr(exc)});raise
