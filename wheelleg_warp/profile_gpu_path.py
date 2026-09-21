"""隔离进程测量实际非零残差闭环，不修改训练代码。"""
import cProfile
import json
from pathlib import Path
import pstats
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import numpy as np
from ppo_env import WheelLegEnv,Scenario
from fast_physics import FastWarpEnv

rows=[]
for cls in (WheelLegEnv,FastWarpEnv):
    env=cls(scenario=Scenario(height_l=.012,mu_l=.7,mu_r=.9),max_seconds=1.02)
    env.reset(seed=1609);env.step(np.zeros(3))
    profiler=cProfile.Profile();start=time.perf_counter();cpu=time.process_time()
    profiler.enable()
    for i in range(50):env.step(np.array([.1,-.1,.1]))
    profiler.disable()
    elapsed=time.perf_counter()-start;cpu=time.process_time()-cpu
    stats=pstats.Stats(profiler)
    selected=[]
    for (file,line,name),(cc,nc,own,total,callers) in stats.stats.items():
        if name in ('control','step','_observation','leg_kinematics','synchronize_stream','capture_launch') or 'cuda_stream_synchronize' in name:
            selected.append(dict(file=file,line=line,name=name,calls=nc,self_seconds=own,cumulative_seconds=total))
    rows.append(dict(backend=cls.__name__,policy_steps=50,physics_steps=2000,wall_seconds=elapsed,
                     process_cpu_seconds=cpu,functions=sorted(selected,key=lambda x:-x['cumulative_seconds'])))
    env.close()
out=ROOT/'wheelleg_warp/results/gpu_saturation_20260921';out.mkdir(exist_ok=True)
(out/'profile.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,ensure_ascii=False),flush=True)
