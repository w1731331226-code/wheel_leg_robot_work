"""同一height-v1检查点的默认关闭余量分配诊断。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from benchmark_parallel import TimedPPO
from native.terrain import HeightTerrainScenario
from terrain_eval import evaluate_terrain


def run(output,mode):
    output.mkdir(parents=True,exist_ok=False)
    prefix=ROOT/'wheelleg_warp/results/height_v1_capability_pilot_20260928/policy_102400'
    paths={name:ROOT/f'wheelleg_warp/results/height_scope_recheck_20260928/{name}/verification.json'
           for name in ('boundary_run1','stratified_run1')}
    model=TimedPPO.load(str(prefix)+'.zip',device='cpu')
    option=dict(height_conditioned=True,grouped_residual=mode=='grouped',project_clipped_base=mode=='clipped')
    rows={};summary={}
    for name,path in paths.items():
        cases=[HeightTerrainScenario(**row['scenario']) for row in json.loads(path.read_text())['rows']]
        rows[name]=evaluate_terrain(model,str(prefix)+'.pkl',cases,env_kwargs=option)
        summary[name]=dict(success=sum(row['success'] for row in rows[name]),total=len(rows[name]))
    root=ROOT
    names=('wheelleg_warp/probe_height_grouped.py','wheelleg_warp/native/environment.py',
           'wheelleg_warp/native/controller.py','wheelleg_warp/native/terrain.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    result=dict(role='diagnostic_not_policy_promotion',policy_steps=102400,mode=mode,
        summary=summary,rows=rows,source_sha256=hashes,
        checkpoint_sha256={suffix:hashlib.sha256(Path(str(prefix)+suffix).read_bytes()).hexdigest()
                           for suffix in ('.zip','.pkl')},
        panel_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS',summary,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('grouped','clipped'),required=True);args=parser.parse_args()
    run(args.output,args.mode)
