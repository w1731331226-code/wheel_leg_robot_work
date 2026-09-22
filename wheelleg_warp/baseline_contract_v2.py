"""Repeat fixed checkpoints on public development cases under terrain task contract v2."""
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path
import argparse,hashlib,json,random,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import torch
from stable_baselines3 import PPO
from native.terrain import TerrainScenario
from terrain_eval import evaluate_terrain,validate_terrain_rows
from training_contract import TASK_CONTRACT_VERSION,source_hashes,checkpoint_hashes
from dashboard.live_env import atomic_json as write

SOURCE=ROOT/'wheelleg_warp/results'
ORDER=['terrain_v3','original_gpu']*3


def prepare(output):
    v3=json.loads((SOURCE/'terrain_v3_1024_20260922/protocol.json').read_text())
    difficult=json.loads((SOURCE/'failure_feedback_20260922/control/protocol.json').read_text())
    panels=[]
    for scenario in v3['development_cases']:
        panels.append(dict(group='v3/'+scenario['terrain'],scenario=asdict(TerrainScenario(**scenario))))
    for scenario in v3['legacy_regression_cases']:
        panels.append(dict(group='v3/original_regression',scenario=asdict(TerrainScenario(**scenario))))
    for label,scenario in zip(difficult['labels'],difficult['development_cases']):
        panels.append(dict(group='difficult/'+label,scenario=asdict(TerrainScenario(**scenario))))
    assert len(panels)==160 and len({r['scenario']['terrain_seed'] for r in panels})==160
    assert not any(720000<=r['scenario']['terrain_seed']<721000 for r in panels)
    fixed=list(range(len(panels)));shuffled=fixed.copy();random.Random(20260923).shuffle(shuffled)
    world_orders=[fixed,fixed,fixed,fixed,shuffled,shuffled]
    checkpoints={
        'terrain_v3':json.loads((SOURCE/'terrain_v3_1024_20260922/selection.json').read_text())['best']['path'],
        'original_gpu':json.loads((SOURCE/'formal_native_1024_20260921/selection.json').read_text())['best']['path'],
    }
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(task_contract_version=TASK_CONTRACT_VERSION,panels=panels,order=ORDER,world_orders=world_orders,
        checkpoints={name:dict(path=path,**checkpoint_hashes(path)) for name,path in checkpoints.items()},
        source_sha256=source_hashes(__file__),backend='MuJoCo Warp GPU',training=False,
        protocol_kind='repeated public development baseline, not model selection or held-out acceptance',
        source_panels=['terrain_v3 development 48 and original regression 16','failure_feedback development 96'],
        source_panel_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (SOURCE/'terrain_v3_1024_20260922/protocol.json',SOURCE/'failure_feedback_20260922/control/protocol.json')},
        historical_holdout_opened=False,reserved_unopened_holdout_namespace=720000)
    write(output/'protocol.json',protocol)
    write(output/'status.json',dict(status='ready',completed_runs=0,training=False))


def verify(protocol):
    if protocol['task_contract_version']!=TASK_CONTRACT_VERSION or protocol['order']!=ORDER:raise ValueError('Wrong task contract or run order')
    if len(protocol['world_orders'])!=len(ORDER) or any(sorted(order)!=list(range(len(protocol['panels']))) for order in protocol['world_orders']):
        raise ValueError('Invalid world permutations')
    for path,digest in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed: '+path)
    for path,digest in protocol['source_panel_sha256'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Public source panel changed: '+path)
    for item in protocol['checkpoints'].values():
        if checkpoint_hashes(item['path'])!={k:item[k] for k in ('checkpoint_sha256','normalization_sha256')}:
            raise ValueError('Policy or normalization changed: '+item['path'])


def summary(rows,panels):
    validate_terrain_rows(rows);groups=defaultdict(list)
    for row,case in zip(rows,panels):
        if row['seed']!=case['scenario']['terrain_seed'] or row['scenario']!=case['scenario']:
            raise ValueError('Development cases changed or reordered')
        groups[case['group']].append(row)
    return {name:dict(total=len(group),success=sum(r['success'] for r in group),
        complete=sum(r['reason']=='completed' for r in group),evidence=sum(r['terrain_evidence_passed'] for r in group),
        reasons=dict(Counter(r['reason'] for r in group))) for name,group in sorted(groups.items())}


def finish(output,protocol):
    saved=[json.loads((output/f'run_{i:02d}_{name}.json').read_text()) for i,name in enumerate(ORDER,1)]
    for item,name in zip(saved,ORDER):
        if item['policy']!=name:raise ValueError('Saved policy does not match frozen order')
        summary(item['runs'],protocol['panels'])
    by_policy=defaultdict(list)
    for item in saved:by_policy[item['policy']].append(item)
    repetitions={}
    for name,records in by_policy.items():
        flips=[]
        for i,case in enumerate(protocol['panels']):
            values=[bool(r['runs'][i]['success']) for r in records]
            if len(set(values))>1:
                flips.append(dict(seed=case['scenario']['terrain_seed'],group=case['group'],outcomes=values,
                    reasons=[r['runs'][i]['reason'] for r in records],peaks_deg=[r['runs'][i]['peak_deg'] for r in records]))
        repetitions[name]=dict(runs=len(records),success_counts=[sum(r['success'] for r in item['runs']) for item in records],
            complete_counts=[sum(r['reason']=='completed' for r in item['runs']) for item in records],flips=flips,
            group_success_counts={group:[item['summary'][group]['success'] for item in records] for group in records[0]['summary']})
    paired=[]
    for index in (0,2,4):
        a=saved[index]['runs'];b=saved[index+1]['runs']
        gained=[];lost=[]
        for terrain,original,case in zip(a,b,protocol['panels']):
            key=dict(seed=case['scenario']['terrain_seed'],group=case['group'])
            if terrain['success'] and not original['success']:gained.append(key)
            if original['success'] and not terrain['success']:lost.append(key)
        paired.append(dict(terrain_run=index+1,original_run=index+2,gained=gained,lost=lost,
            world_order='fixed' if index<4 else 'shuffled'))
    result=dict(task_contract_version=TASK_CONTRACT_VERSION,training=False,weights_unchanged=True,
        heldout_evaluated=False,panels_total=len(protocol['panels']),run_order=ORDER,
        repetitions=repetitions,paired=paired,formal_training_ready=False)
    write(output/'verification.json',result)
    write(output/'status.json',dict(status='completed',completed_runs=len(saved),training=False))
    print(json.dumps({k:result[k] for k in ('panels_total','repetitions')},ensure_ascii=False),flush=True)


def run(output):
    protocol=json.loads((output/'protocol.json').read_text());verify(protocol)
    cases=[TerrainScenario(**item['scenario']) for item in protocol['panels']]
    torch.set_num_threads(1)
    for index,name in enumerate(ORDER,1):
        path=output/f'run_{index:02d}_{name}.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['policy']!=name:raise ValueError('Existing run mismatches protocol')
            if saved['world_order']!=protocol['world_orders'][index-1]:raise ValueError('Existing run has wrong world layout')
            summary(saved['runs'],protocol['panels'])
            continue
        verify(protocol)
        write(output/'status.json',dict(status='evaluating',run=index,policy=name,completed_runs=index-1,training=False))
        checkpoint=protocol['checkpoints'][name]['path'];model=PPO.load(checkpoint+'.zip',device='cpu')
        if model.policy.observation_space.shape!=(32,) or model.policy.action_space.shape!=(3,):
            raise ValueError('M3 observation/action space mismatch')
        initial={key:value.clone() for key,value in model.policy.state_dict().items()}
        world_order=protocol['world_orders'][index-1]
        observed=evaluate_terrain(model,checkpoint+'.pkl',[cases[i] for i in world_order])
        rows=[None]*len(cases)
        for original_index,row in zip(world_order,observed):rows[original_index]=row
        if any(not torch.equal(value,model.policy.state_dict()[key]) for key,value in initial.items()):
            raise AssertionError('Evaluation changed policy weights')
        write(path,dict(policy=name,checkpoint=checkpoint,world_order=world_order,summary=summary(rows,protocol['panels']),runs=rows))
        print('run',index,name,'success',sum(row['success'] for row in rows),'complete',sum(row['reason']=='completed' for row in rows),flush=True)
    verify(protocol);finish(output,protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args.output)
    else:run(args.output)
