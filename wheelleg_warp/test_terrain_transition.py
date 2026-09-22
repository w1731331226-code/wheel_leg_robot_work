"""Geometry and GPU regression for smooth slope entry versus abrupt obstacles."""
from pathlib import Path
import sys
from dataclasses import replace
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
from native.terrain import TerrainScenario,model
from native.terrain_env import TerrainEnv

def top(m,geom,x,y):
    matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,m.geom_quat[geom]);normal=matrix.reshape(3,3)[:,2]
    p=m.geom_pos[geom]
    return p[2]+(m.geom_size[geom,2]-normal[0]*(x-p[0])-normal[1]*(y-p[1]))/normal[2]

def check():
    cases=[]
    for kind in ('cross_slope','split_level'):
        for direction in (-1,1):
            for sign in (-1,1):
                s=TerrainScenario(terrain=kind,speed=direction*.7,grade_deg=sign*4.,height_l=0.,height_r=0.,step_height_m=.021,
                    relative_attitude=True,transition_run_m=.4,lateral_margin_m=.35)
                fixed=model(s);old=model(replace(s,transition_run_m=0.,lateral_margin_m=0.))
                for side,y in enumerate((-.15,.15)):
                    central=fixed.geom('terrain_00' if kind=='cross_slope' else f'terrain_{side:02d}').id
                    h=top(fixed,central,direction*s.center,y)
                    np.testing.assert_allclose(h,top(old,central,direction*s.center,y),atol=1e-9)
                    for end,sigma in enumerate((-1,1)):
                        g=fixed.geom(f'terrain_{12+side*2+end:02d}').id
                        np.testing.assert_allclose(top(fixed,g,direction*(s.center+sigma*.65),y),h,atol=1e-9)
                        np.testing.assert_allclose(top(fixed,g,direction*(s.center+sigma*1.05),y),0.,atol=1e-9)
                cases.append(s)
    # Validation rejects wrong terrain/type and non-finite geometry parameters.
    for args in (dict(transition_run_m=float('nan')),dict(lateral_margin_m=-.1),dict(terrain='step',transition_run_m=.4)):
        try:TerrainScenario(**args)
        except ValueError:pass
        else:raise AssertionError(args)
    env=TerrainEnv(len(cases),scenario=cases);env.reset();pending=set(range(len(cases)))
    try:
        np.testing.assert_allclose(env.param.numpy()[:,2],np.array([s.center+end+.15 for s,end in zip(cases,env.required_terrain_end)]),atol=1e-12,rtol=0.)
        assert all(end>=1.05 for end in env.required_terrain_end)
        while pending:
            _,_,done,infos=env.step(np.zeros((len(cases),3),np.float32))
            for i in list(pending):
                if done[i]:
                    assert infos[i]['reason']=='completed' and infos[i]['terrain_evidence_passed'],(i,infos[i])
                    assert infos[i]['terrain_required_end_m']>=1.05
                    pending.remove(i)
    finally:env.close()
    print('PASS: unchanged central heights, zero-height ramp endpoints, both directions/sides, exit beyond ramps')

if __name__=='__main__':check()
