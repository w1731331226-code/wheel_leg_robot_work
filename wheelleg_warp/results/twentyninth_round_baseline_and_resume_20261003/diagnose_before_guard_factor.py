"""Original CPU28 vs current CPU and GPU static/current tables; no gain tuning."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from train_height_comparison import protocol,evaluate,summary,write
import train_height_comparison as runner
from pretrain_yaw import evaluate_case
from rm_controller import nominal_design,nominal_design_115
from native.environment import NativeEnv
from native.terrain import bank_height_115,HeightTerrainScenario
from training_contract import digest,source_hashes
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'twentyeighth_round_comparison_entry_20261003/protocol_v3'

def run(mode):
    p=protocol(PRIOR);dest=HERE/mode;dest.mkdir()
    write(dest/'registration.json',dict(mode=mode,case_names=[r['seed'] for r in p['regression']],
        no_yaw_gain_or_gate_changes=True,source_sha256=source_hashes(__file__)))
    if mode=='cpu_current':
        tasks=[(p['b1_candidates'][0],dict(seed=r['name'],scenario=r['scenario'])) for r in p['original_regression']]
        with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(evaluate_case,tasks))
        result=dict(total=28,success=sum(r['success'] for r in rows),runs=rows)
    else:
        original=runner.raw_env
        h,t,_,_=nominal_design_115();oldh,oldt,_,_=nominal_design()
        for height,table in zip(oldh,oldt):
            index=int(np.flatnonzero(h==height)[0])
            for a,b in zip(t[index],table):np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)
        def factory(cases,residual_mode):
            env=(NativeEnv(n=len(cases),scenario=[HeightTerrainScenario(**r['scenario']) for r in cases],residual_mode=residual_mode,bank_factory=bank_height_115,height_conditioned=True,height_design='range115',height_safety='physical_v1',observation_contract='request_state_v1',nominal_correction=True,design_joint_gate=True) if mode=='gpu_bare_static' else original(cases,residual_mode))
            if mode in ('gpu_static5','gpu_bare_static'):
                env.k['gains'].assign(np.stack([v[0] for v in t]));env.k['feed'].assign(np.stack([v[1] for v in t]));env.k['angles'].assign(np.array([v[2] for v in t]))
            return env
        runner.raw_env=factory
        rows=evaluate(p['regression'],'diff3',candidate=p['b1_candidates'][0])
        result=dict(summary=summary(rows),runs=rows,success=sum(r['success'] for r in rows),
            current_body_and_v6_flags_same=True,only_design_table_changed=mode=='gpu_static5',static4_shared_nodes_equal_static5=True)
    write(dest/'verification.json',result);print('DIAG',mode,result['success'],'/28',flush=True)
    print('FAIL',[(r['seed'],r['reason'],r['peak_deg'][2]) for r in rows if not r['success']],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu_current','gpu_static5','gpu_current','gpu_bare_static']);run(p.parse_args().mode)
