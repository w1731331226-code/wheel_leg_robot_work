"""1024-world startup and real snapshot check, without any learning."""
from pathlib import Path
from dataclasses import replace
import json,struct,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.live import LiveNativeEnv
from native.terrain import bank,sample_terrain_v4
from dashboard.live_env import atomic_json as write

if (HERE/'scale.json').exists():raise FileExistsError('Refusing to overwrite scale evidence')
cases=[sample_terrain_v4(820000+i,'train') for i in range(1024)]
cases[0]=replace(cases[0],delay_ms=20.);cases[1]=replace(cases[1],delay_ms=15.)
started=time.perf_counter();env=LiveNativeEnv(HERE/'scale_live',n=1024,scenario=cases,bank_factory=bank,phase='contract_engineering_no_training')
try:
    env.reset();assert env.history.shape==(1024,41,32) and env.model.stat.meaninertia.shape==(1024,)
    for _ in range(3):
        obs,reward,done,_=env.step(np.zeros((1024,3),np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward).all() and not done.any()
    assert np.isfinite(env.state.numpy()).all()
    env.save_overview();packet=(HERE/'scale_live/overview.bin').read_bytes();length=struct.unpack('<I',packet[:4])[0]
    overview=json.loads(packet[4:4+length]);assert overview['environments']==1024
    metadata=json.loads((HERE/'scale_live/latest.json').read_text());assert metadata['task_contract_version']==2
    result=dict(passed=True,environments=1024,physics_steps_per_world=120,training_steps=0,
        terrain_types=sorted({s.terrain for s in cases}),history_slots=41,per_world_inertia=True,overview_worlds=1024,
        seconds=time.perf_counter()-started,full_terrain_traversal_claimed=False)
    write(HERE/'scale.json',result);print(json.dumps(result),flush=True)
finally:env.close()
