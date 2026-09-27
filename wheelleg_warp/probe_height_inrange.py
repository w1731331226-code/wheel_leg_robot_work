"""按现有训练范围分层配对六档腿长，检查多高度名义闭环。"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import argparse
import numpy as np
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario,TerrainScenario,V3_TERRAINS,bank_height_v3,sample_terrain_v3


HEIGHTS=(.16,.20,.25,.30,.35,.38)


def panel():
    selected={(terrain,direction):[] for terrain in V3_TERRAINS for direction in (-1,1)}
    for seed in range(1132000,1133000):
        scenario=sample_terrain_v3(seed,stage=3,split='train')
        key=(scenario.terrain,1 if scenario.speed>0 else -1)
        quota=2 if scenario.terrain=='legacy' else 1
        if len(selected[key])<quota:selected[key].append((seed,scenario))
        if all(len(items)==(2 if key[0]=='legacy' else 1) for key,items in selected.items()):break
    assert sum(map(len,selected.values()))==20
    return [item for terrain in V3_TERRAINS for direction in (-1,1) for item in selected[terrain,direction]]


def run(output,boundary=False):
    if boundary:
        bases=[(-1,TerrainScenario(speed=speed,height_l=.02 if side=='L' else 0.,height_r=.02 if side=='R' else 0.))
               for side in ('L','R') for speed in (.5,-.5,1.,-1.)]
    else:bases=panel()
    scenarios=[];seeds=[]
    for seed,base in bases:
        for height in HEIGHTS:
            scenarios.append(HeightTerrainScenario(**asdict(base),stand_height_m=height));seeds.append(seed)
    assert all(max(s.height_l,s.height_r,s.step_height_m)<=.02 for s in scenarios)
    n=len(scenarios);env=NativeEnv(n=n,scenario=scenarios,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=0)
    obs=env.reset();assert np.allclose(obs[:,11],[s.stand_height_m-.3 for s in scenarios],atol=1e-6)
    assert np.allclose(obs[:,22:24],np.asarray([s.stand_height_m for s in scenarios])[:,None],atol=1e-6)
    results=[None]*n;zero=np.zeros((n,3),np.float32)
    for _ in range(700):
        _,_,done,infos=env.step(zero)
        for i in np.flatnonzero(done):
            if results[i] is None:
                info=infos[i]
                results[i]=dict(reason=info['reason'],success=info['success'],duration_s=info['duration_s'],
                    peak_deg=info['peak_deg'],relative_peak_deg=info['relative_peak_deg'],
                    height_rmse_m=info['height_rmse_m'],velocity_rmse=info['velocity_rmse'],
                    terrain_evidence_passed=info['terrain_evidence_passed'])
        if all(row is not None for row in results):break
    assert all(row is not None for row in results)
    rows=[dict(seed=seed,scenario=asdict(s),result=result) for seed,s,result in zip(seeds,scenarios,results)]
    counts={str(h):dict(success=sum(r['result']['success'] for r in rows if r['scenario']['stand_height_m']==h),
                         total=sum(r['scenario']['stand_height_m']==h for r in rows)) for h in HEIGHTS}
    root=Path(__file__).resolve().parents[1]
    names=('wheelleg_warp/probe_height_inrange.py','wheelleg_warp/native/environment.py','wheelleg_warp/native/controller.py',
           'wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py','wheelleg_ppo/tools/ppo_env.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    selection=('single-side 20 mm boundary, either side, speeds ±0.5/±1.0 m/s, nominal other parameters' if boundary else
               'first two legacy and first one of each other terrain per speed sign from train seeds 1132000:1133000')
    (output/'verification.json').write_text(json.dumps(dict(protocol='public training-range boundary engineering panel, no PPO' if boundary else
        'public training-range stratified engineering panel, no PPO',selection=selection,
        heights_m=HEIGHTS,residual_scale=0,source_sha256=hashes,counts_by_height=counts,rows=rows),ensure_ascii=False,indent=2)+'\n')
    print('PASS',sum(r['result']['success'] for r in rows),'/',n,counts,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--boundary',action='store_true');args=parser.parse_args()
    run(args.output,args.boundary)
