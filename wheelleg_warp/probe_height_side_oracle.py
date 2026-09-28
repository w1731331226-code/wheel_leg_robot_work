"""验证侧别条件M3接触前预瞄与接触触发的受限上界。"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import numpy as np
from ppo_env import sample_scenario
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario,bank_height_v3

HEIGHTS=(.16,.20,.25,.30,.35,.38)


def new_cases():
    cells={(side,direction,band):None for side in ('L','R') for direction in (-1,1) for band in ('low','high')}
    for seed in range(1138000,1140000):
        base=sample_scenario('train',seed,3)
        if (base.height_l==0)==(base.height_r==0) or max(base.height_l,base.height_r)<.016:continue
        side='L' if base.height_l else 'R'
        key=(side,1 if base.speed>0 else -1,'low' if abs(base.speed)<.75 else 'high')
        if cells[key] is None:cells[key]=(seed,base)
        if all(item is not None for item in cells.values()):break
    assert all(item is not None for item in cells.values())
    cases=[]
    for side in ('L','R'):
        for direction in (-1,1):
            for band in ('low','high'):
                seed,base=cells[side,direction,band]
                for h in HEIGHTS:
                    cases.append(HeightTerrainScenario(**asdict(base),terrain='legacy',terrain_seed=seed,
                        relative_attitude=True,stand_height_m=h))
    assert len(cases)==48
    return cases


def rollout(cases,reference=None,window=None):
    n=len(cases);env=NativeEnv(n=n,scenario=cases,bank_factory=bank_height_v3,height_conditioned=True,
                               residual_scale=0 if reference is None else 1)
    env.reset();contacts=[None]*n;results=[None]*n;filtered=[None]*n
    directions=np.asarray([np.sign(s.speed)*(1 if s.height_l else -1) for s in cases],np.float32)
    for step in range(700):
        time=step*.02;actions=np.zeros((n,3),np.float32)
        if reference is not None:
            begin,end=window
            active=np.asarray([(t+begin<=time<t+end) for t in reference])
            actions[:,2]=active*directions
        env.step_async(actions)
        state=env.state.numpy();controller=env.k['state'].numpy()
        for i,s in enumerate(cases):
            mask=1 if s.height_l else 2
            if contacts[i] is None and int(state[i,14])&mask:
                contacts[i]=round(float(state[i,0]*.0005),3)
            if reference is not None and filtered[i] is None and time+.02>=reference[i]:
                filtered[i]=float(controller[i,18])
        _,_,done,infos=env.step_wait()
        for i in np.flatnonzero(done):
            if results[i] is None:
                info=infos[i]
                results[i]=dict(success=info['success'],reason=info['reason'],peak_deg=info['peak_deg'],
                    height_rmse_m=info['height_rmse_m'],velocity_rmse=info['velocity_rmse'],
                    terrain_evidence_passed=info['terrain_evidence_passed'])
        if all(row is not None for row in results):break
    assert all(row is not None for row in results)
    if reference is None:assert all(t is not None for t in contacts)
    if reference is not None and window[0]<0:assert all(v is None or abs(v)>=.99 for v in filtered)
    return [dict(scenario=asdict(s),contact_bin_s=t,filtered_at_reference_contact=f,result=r)
            for s,t,f,r in zip(cases,contacts,filtered,results)]


def summarize(rows):
    return dict(success=sum(row['result']['success'] for row in rows),total=len(rows),
        by_height={str(h):sum(row['result']['success'] for row in rows if row['scenario']['stand_height_m']==h)
                   for h in HEIGHTS},
        by_side={side:sum(row['result']['success'] for row in rows
                          if (row['scenario']['height_l']>0)==(side=='L')) for side in ('L','R')})


def run(output,reverse=False):
    old_path=Path(__file__).resolve().parents[1]/'wheelleg_warp/results/height_scope_recheck_20260928/boundary_run1/verification.json'
    old=[HeightTerrainScenario(**row['scenario']) for row in json.loads(old_path.read_text())['rows']]
    panels=dict(old_public=old,new_public=new_cases())
    if reverse:panels={name:list(reversed(cases)) for name,cases in panels.items()}
    records={};summaries={}
    for name,cases in panels.items():
        baseline=rollout(cases);reference=[row['contact_bin_s'] for row in baseline]
        preview=rollout(cases,reference,(-.1,.05))
        reactive=rollout(cases,reference,(0.,.15))
        records[name]=dict(baseline=baseline,preview=preview,reactive=reactive)
        summaries[name]={kind:summarize(rows) for kind,rows in records[name].items()}
        print('PANEL',name,summaries[name],flush=True)
    root=Path(__file__).resolve().parents[1]
    names=('wheelleg_warp/probe_height_side_oracle.py','wheelleg_warp/native/environment.py',
           'wheelleg_warp/native/controller.py','wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py',
           'wheelleg_ppo/tools/ppo_env.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    (output/'verification.json').write_text(json.dumps(dict(protocol='side-aware privileged preview and contact trigger, fixed ±1 M3 channel3',
        selection='old public 20mm boundary and first high one-sided 16-20mm seed per side/direction/speed band from train 1138000:1140000',
        source_sha256=hashes,old_panel_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        summary=summaries,panels=records,reversed_world_order=reverse,training=False,holdout_opened=False),ensure_ascii=False,indent=2)+'\n')
    print('PASS',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reverse',action='store_true');args=parser.parse_args()
    run(args.output,args.reverse)
