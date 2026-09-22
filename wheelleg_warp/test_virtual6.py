"""virtual6 includes diff3 exactly and permits nonzero common-mode virtual forces."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import numpy as np
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv

s=TerrainScenario(speed=.7,height_l=0.,height_r=0.,terrain='step',step_height_m=.021)
a=TerrainEnv(1,scenario=s);b=TerrainEnv(1,scenario=s,residual_mode='virtual6')
try:
    a.reset();b.reset();rng=np.random.default_rng(714)
    for _ in range(80):
        action=rng.uniform(-.5,.5,(1,3)).astype(np.float32)
        six=np.stack([action[:,0],-action[:,0],action[:,1],-action[:,1],action[:,2],-action[:,2]],axis=1)
        x=a.step(action);y=b.step(six)
        np.testing.assert_allclose(a.diag.numpy(),b.diag.numpy(),atol=1e-6,rtol=0.)
        np.testing.assert_allclose(x[0],y[0],atol=1e-6,rtol=0.)
    # Common wheel torque is outside the diff3 subspace, but inside virtual6.
    for _ in range(4):b.step(np.array([[0,0,0,0,.4,.4]],np.float32))
    diag=b.diag.numpy()[0]
    assert diag[10]>0 and diag[11]>0 and np.isfinite(diag).all(),diag
    b.reset();assert np.all(b.k['state'].numpy()[0,16:]==0)
    try:b.step(np.zeros((1,3),np.float32))
    except ValueError:pass
    else:raise AssertionError('wrong action dimension accepted')
finally:a.close();b.close()
print('PASS: differential transfer, common-mode execution, reset and action dimensions')
