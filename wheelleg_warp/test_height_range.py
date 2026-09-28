"""多高度协议的最小物理检查：六档×四种地形，零残差。"""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import numpy as np
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario,TerrainScenario,bank_height_v3,bank_v3,sample_height_terrain_v3


def main():
    heights=(.16,.20,.25,.30,.35,.38)
    assert [sample_height_terrain_v3(1130000+i).stand_height_m for i in range(3)]==[.16,.38,.3]
    for invalid in (.114,.381,float('nan')):
        try:HeightTerrainScenario(stand_height_m=invalid)
        except ValueError:pass
        else:raise AssertionError(f'非法高度被接受: {invalid}')
    terrains=(('legacy',{}),('step',{'step_height_m':.02}),('cross_slope',{'grade_deg':3.}),('rough',{'roughness_m':.004}))
    scenarios=[HeightTerrainScenario(speed=.5,stand_height_m=h,terrain=terrain,terrain_seed=1234,relative_attitude=True,**extra)
               for terrain,extra in terrains for h in heights]
    env=NativeEnv(n=len(scenarios),scenario=scenarios,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=0)
    obs=env.reset()
    targets=np.tile(heights,len(terrains))
    assert np.allclose(obs[:,11],targets-.3,atol=1e-6)
    assert np.allclose(obs[:,22:24],targets[:,None],atol=1e-6)
    completed=[None]*len(scenarios)
    for _ in range(500):
        _,_,done,infos=env.step(np.zeros((len(scenarios),3),dtype=np.float32))
        for i in np.flatnonzero(done):
            if completed[i] is None:completed[i]=infos[i]
        if all(row is not None for row in completed):break
    for scenario,row in zip(scenarios,completed):
        assert row is not None and row['reason']=='completed' and row['success'],(scenario.terrain,scenario.stand_height_m,row)
        assert row['height_rmse_m']<=.02,(scenario.terrain,scenario.stand_height_m,row['height_rmse_m'])
    legacy=NativeEnv(n=1,scenario=TerrainScenario(speed=.5),bank_factory=bank_v3,residual_scale=0)
    old=legacy.reset()
    assert np.allclose(old[0,[11,22,23]],[0.,.3,.3],atol=1e-6)
    legacy.step(np.zeros((1,3),dtype=np.float32))
    print('PASS',[(s.terrain,s.stand_height_m,round(row['height_rmse_m'],6)) for s,row in zip(scenarios,completed)])
    hard=[HeightTerrainScenario(speed=v,height_l=.027,stand_height_m=h) for v in (.5,-.5) for h in heights]
    probe=NativeEnv(n=len(hard),scenario=hard,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=0)
    probe.reset();results=[None]*len(hard)
    for _ in range(700):
        _,_,done,infos=probe.step(np.zeros((len(hard),3),dtype=np.float32))
        for i in np.flatnonzero(done):
            if results[i] is None:results[i]=infos[i]
        if all(row is not None for row in results):break
    print('ASYMMETRIC_DIAGNOSTIC',[(s.speed,s.stand_height_m,None if row is None else row['success'],
         None if row is None else round(max(row['peak_deg']),2)) for s,row in zip(hard,results)])


if __name__=='__main__':main()
