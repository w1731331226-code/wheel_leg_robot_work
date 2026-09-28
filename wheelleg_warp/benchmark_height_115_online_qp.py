"""现有Python CPU有限差分＋HiGHS局部安全求解的实际耗时。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from scipy.optimize import linprog
from native.terrain import HeightTerrainScenario,model
from probe_height_115_local_lp import EPS,GAMMA,rollout
from probe_height_115_passive import JOINTS


def timing(call,repeats):
    for _ in range(3):call()
    duration=[]
    for _ in range(repeats):
        start=time.perf_counter();call();duration.append((time.perf_counter()-start)*1000)
    return dict(repeats=repeats,median_ms=float(np.median(duration)),p95_ms=float(np.percentile(duration,95)))


def run(output):
    assert not output.exists()
    folder=ROOT/'wheelleg_warp/results/height_115_local_lp_20260928'
    lp_path=folder/'verification.json';saved=json.loads(lp_path.read_text());linear=np.load(folder/'linearization.npz')
    state_folder=ROOT/'wheelleg_warp/results/height_115_local_states_20260928'
    state_path=state_folder/'verification.json';state=json.loads(state_path.read_text());windows=np.load(state_folder/'windows.npz')
    assert hashlib.sha256((folder/'linearization.npz').read_bytes()).hexdigest()==saved['linearization_sha256']
    assert hashlib.sha256((state_folder/'windows.npz').read_bytes()).hexdigest()==state['windows_sha256']
    for record in (saved,state):
        assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
                   for name,digest in record['source_sha256'].items())
    event_id=1;row=saved['rows'][0];event=state['events'][event_id]
    assert event['world']==row['world'] and event['kind']==row['event']
    m=model(HeightTerrainScenario(**event['scenario']))
    pre=windows[f'event_{event_id}_pre'];start=int(np.argmin(abs(pre[:,0]-row['start_s'])))
    ids=np.array([m.joint(name).qposadr[0] for name in JOINTS])
    ranges=np.asarray([m.jnt_range[m.joint(name).id] for name in JOINTS])
    g=linear[f'event_{event_id}_baseline'];s=linear[f'event_{event_id}_sensitivity']
    def solver(steps):
        values=g[:steps].reshape(-1);matrix=s[:steps].reshape(-1,6)
        safe=np.column_stack((-matrix,np.zeros(len(values))))
        positive=np.column_stack((np.eye(6),-np.ones(6)))
        negative=np.column_stack((-np.eye(6),-np.ones(6)))
        limits=row['trust_bounds_Nm']
        def one():
            result=linprog(np.r_[np.zeros(6),1.],A_ub=np.vstack((safe,positive,negative)),
                b_ub=np.r_[values-GAMMA,np.zeros(12)],bounds=[tuple(item) for item in limits]+[(0.,EPS)],method='highs')
            assert result.success
        return one
    zero=np.zeros(6)
    def thirteen_rollouts():
        rollout(m,pre,start,state['columns'],zero,ids,ranges)
        for aid in range(6):
            for sign in (-1,1):
                delta=np.zeros(6);delta[aid]=sign*EPS
                rollout(m,pre,start,state['columns'],delta,ids,ranges)
    result=dict(role='cpu_python_current_implementation_cost_not_hardware_deadline_proof',
                physical_step_budget_ms=.5,finite_difference_13x40=timing(thirteen_rollouts,5),
                highs_24_constraints=timing(solver(1),100),highs_960_constraints=timing(solver(40),100),
                python_version=sys.version.split()[0],training=False,final_holdout_opened=False)
    sources=('wheelleg_warp/benchmark_height_115_online_qp.py','wheelleg_warp/probe_height_115_local_lp.py',
             'wheelleg_warp/probe_height_115_passive.py','wheelleg_warp/native/terrain.py',
             'wheelleg_warp/native/models.py','wheelleg_ppo/xml/wheelleg.xml')
    result['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    result['linearization_sha256']=saved['linearization_sha256']
    result['source_lp_verification_sha256']=hashlib.sha256(lp_path.read_bytes()).hexdigest()
    result['source_state_verification_sha256']=hashlib.sha256(state_path.read_bytes()).hexdigest()
    result['state_windows_sha256']=state['windows_sha256']
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print({key:value for key,value in result.items() if key in ('finite_difference_13x40','highs_24_constraints','highs_960_constraints')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
