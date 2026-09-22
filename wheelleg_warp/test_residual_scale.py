"""Runtime attenuation must match scaling the bounded policy actions externally."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import numpy as np
from native.environment import NativeEnv
from native.terrain_env import TerrainEnv
from native.terrain import TerrainScenario

for bad in (-.1,1.1,float('nan'),float('inf')):
    try:NativeEnv(1,residual_scale=bad)
    except ValueError:pass
    else:raise AssertionError('invalid scale accepted')
s=TerrainScenario(speed=.7,height_l=0.,height_r=0.,terrain='step',step_height_m=.02)
a=TerrainEnv(1,scenario=s);b=TerrainEnv(1,scenario=s,residual_scale=.5)
try:
    a.reset();b.reset();rng=np.random.default_rng(20260922)
    for _ in range(100):
        action=rng.uniform(-1,1,(1,3)).astype(np.float32)
        x=a.step(action*.5);y=b.step(action)
        np.testing.assert_array_equal(a.targets.numpy(),b.targets.numpy())
        np.testing.assert_allclose(x[0],y[0],atol=1e-6,rtol=0.)
        np.testing.assert_allclose(x[1],y[1],atol=1e-6,rtol=0.)
    try:b.step(np.full((1,3),2,dtype=np.float32))
    except ValueError:pass
    else:raise AssertionError('scale must not conceal invalid policy actions')
finally:a.close();b.close()
print('PASS: runtime scale equals external action scaling; original limits validated first')
