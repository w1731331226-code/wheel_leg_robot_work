"""Small terrain-v1 engineering check: generator, topology and real closed loop."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import numpy as np
from native.terrain import TERRAINS,TerrainScenario,model,sample_terrain
from native.terrain_env import TerrainEnv

first=sample_terrain(12345,3);assert first==sample_terrain(12345,3)
sample=[sample_terrain(20000+i,3) for i in range(1024)]
assert set(x.terrain for x in sample)==set(TERRAINS)
legacy=sum(x.terrain=='legacy' for x in sample)/len(sample);assert .25<legacy<.35
try:TerrainScenario(terrain='ramp',grade_deg=6)
except ValueError:pass
else:raise AssertionError('超范围坡度未拒绝')
cases=[]
for name in TERRAINS:
    cases.append(next(sample_terrain(seed,3) for seed in range(30000,40000) if sample_terrain(seed,3).terrain==name))
models=[model(x) for x in cases];assert len({(m.nq,m.nv,m.ngeom,m.nu) for m in models})==1
env=TerrainEnv(len(cases),scenario=cases);obs=env.reset();finished=[]
static=env.cpu.geom('terrain_00').id
np.testing.assert_allclose(env.data.geom_xpos.numpy()[:,static],env.model.geom_pos.numpy()[:,static])
for _ in range(450):
    obs,reward,done,infos=env.step(np.zeros((len(cases),3),np.float32))
    assert np.isfinite(obs).all() and np.isfinite(reward).all()
    for i in np.flatnonzero(done):
        finished.append(infos[i]['reason'])
        assert infos[i]['terrain_passed']==(cases[i].terrain=='legacy' or infos[i]['touched_terrain_contact_mask']==12)
assert len(finished)>=len(cases) and set(finished)=={'completed'}
np.testing.assert_allclose(env.data.geom_xpos.numpy()[:,static],env.model.geom_pos.numpy()[:,static])
try:env.step_async(np.full((len(cases),3),2,dtype=np.float32))
except ValueError:pass
else:raise AssertionError('非法动作未拒绝')
env.close();print('PASS',dict(types=list(TERRAINS),legacy_fraction=legacy,episodes=len(finished)))
