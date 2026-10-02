"""Failed-grid mechanism check; same public cases, zero Actor, reference on/off."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario
from probe_height_115_action_predict_loow import sha
OUT=Path(__file__).resolve().parent


def run():
    assert not (OUT/'contrast.json').exists()
    grid=json.loads((OUT/'grid/verification.json').read_text())
    indices=[i for i,e in enumerate(grid['episodes']) if not e['success']]
    scenes=[HeightTerrainScenario(**grid['registration']['scenarios'][i]) for i in indices]
    results=[]
    for enabled in (False,True):
        env=NativeEnv.height115_candidate(n=len(scenes),scenario=scenes,residual_scale=0,
            nominal_correction=not enabled,braking_reference=enabled)
        try:
            env.reset();rows=[None]*len(scenes);first=[None]*len(scenes);previous=np.zeros(len(scenes))
            for _ in range(700):
                _,_,done,infos=env.step(np.zeros((len(scenes),3),np.float32))
                state=env.state.numpy()
                for w in range(len(scenes)):
                    if rows[w] is not None:continue
                    end=infos[w]['physical_steps']*.0005 if done[w] else state[w,0]*.0005
                    margin=infos[w]['min_active_design_margin_rad'] if done[w] else state[w,38]
                    arrival=infos[w]['arrival_s'] if done[w] else (state[w,1] if state[w,1]>=0 else None)
                    if margin<0 and first[w] is None:first[w]=dict(interval_s=[float(previous[w]),float(end)],arrival_s=None if arrival is None else float(arrival))
                    previous[w]=end
                    if done[w]:rows[w]={k:v for k,v in infos[w].items() if k!='terminal_observation'}
                if all(r is not None for r in rows):break
            assert all(r is not None for r in rows)
            results.append(dict(reference_enabled=enabled,episodes=rows,first_design_failure=first))
            print('CONTRAST',enabled,[(indices[w],e['success'],e['stop_distance_m'],e['tail_speed_m_s'],e['min_active_design_margin_rad'],first[w]) for w,e in enumerate(rows)],flush=True)
        finally:env.close()
    result=dict(grid_indices=indices,results=results,learning=False,full_admission=False,
        scope='Previously failed public grid cases, descriptive paired runs;20ms first-failure intervals, no gain/weight search or bitwise replay claim.',
        input_sha256=sha(OUT/'grid/verification.json'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/braking_reference.py')})
    (OUT/'contrast.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':run()
