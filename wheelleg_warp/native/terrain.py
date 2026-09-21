"""Uniform-topology terrain bank for the independent terrain-v1 experiment."""
from dataclasses import asdict,dataclass
import math
import numpy as np
import mujoco
from ppo_env import Scenario,sample_scenario
from native.models import build_spec,compile_spec,batch

TERRAINS=('legacy','ramp','cross_slope','rough','step','mixed')
TERRAIN_GEOMS=16


@dataclass(frozen=True)
class TerrainScenario(Scenario):
    terrain:str='legacy'
    grade_deg:float=0.
    roughness_m:float=0.
    step_height_m:float=0.
    terrain_seed:int=0

    def __post_init__(self):
        # Reuse the frozen base validation without passing terrain's string fields into it.
        Scenario(**{name:getattr(self,name) for name in Scenario.__dataclass_fields__})
        if self.terrain not in TERRAINS or not -5<=self.grade_deg<=5 or not 0<=self.roughness_m<=.012 or not 0<=self.step_height_m<=.03:
            raise ValueError('地形参数超出terrain-v1冻结范围')


def sample_terrain(seed,stage=3,split='train'):
    if stage not in (1,2,3) or split not in ('train','development','ood'):
        raise ValueError('无效地形课程或划分')
    base=sample_scenario('train',seed,stage)
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),73091,stage,{'train':1,'development':2,'ood':3}[split]]))
    if split=='ood':
        terrain=str(rng.choice(('ramp','cross_slope','rough','step','mixed')))
    else:
        choices={1:(('legacy','ramp','cross_slope'),(.4,.3,.3)),
                 2:(('legacy','ramp','cross_slope','rough','step'),(.3,.2,.15,.2,.15)),
                 3:(TERRAINS,(.3,.15,.15,.15,.15,.1))}[stage]
        terrain=str(rng.choice(choices[0],p=choices[1]))
    values=asdict(base)
    if terrain not in ('legacy','mixed'):
        values.update(height_l=0.,height_r=0.)
    grade=0.;rough=0.;step=0.
    if terrain in ('ramp','cross_slope'):
        limit=(3.,5.) if split=='ood' else (1.,3.)
        grade=float(rng.choice((-1,1))*rng.uniform(*limit)) if terrain=='cross_slope' else float(rng.uniform(*limit))
    if terrain in ('rough','mixed'):
        rough=float(rng.uniform(.006,.010) if split=='ood' else rng.uniform(.002,.006 if stage==3 else .004))
    if terrain=='step':
        step=float(rng.uniform(.020,.030) if split=='ood' else rng.uniform(.005,.020 if stage==3 else .012))
    return TerrainScenario(**values,terrain=terrain,grade_deg=grade,roughness_m=rough,step_height_m=step,terrain_seed=int(seed))


def _quat(axis,angle):
    out=np.zeros(4);mujoco.mju_axisAngle2Quat(out,np.asarray(axis,dtype=float),angle);return out


def model(s):
    spec=build_spec(s);tiles=[]
    for i in range(TERRAIN_GEOMS):
        tiles.append(spec.worldbody.add_geom(name=f'terrain_{i:02d}',type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0,0,-10],size=[.05,.16,.001],friction=[(s.mu_l+s.mu_r)/2,.02,.001]))
    direction=1 if s.speed>0 else -1;thickness=.012
    def box(index,u,z,size,quat=(1,0,0,0)):
        g=tiles[index];g.pos=[direction*u,0,z];g.size=size;g.quat=quat
    if s.terrain=='ramp':
        angle=math.radians(abs(s.grade_deg));run=.5;plateau=.3;height=run*math.sin(angle)
        center=s.center;z=height/2-math.cos(angle)*thickness/2
        box(0,center-plateau/2-run/2,z,[run/2,.16,thickness/2],_quat((0,1,0),-direction*angle))
        box(1,center,height-thickness/2,[plateau/2,.16,thickness/2])
        box(2,center+plateau/2+run/2,z,[run/2,.16,thickness/2],_quat((0,1,0),direction*angle))
    elif s.terrain=='cross_slope':
        angle=math.radians(s.grade_deg);half_width=.16
        z=abs(math.sin(angle))*half_width-math.cos(angle)*thickness/2
        box(0,s.center,z,[.65,half_width,thickness/2],_quat((1,0,0),angle))
    elif s.terrain in ('rough','mixed'):
        rng=np.random.default_rng(np.random.SeedSequence([s.terrain_seed,889]))
        for i in range(12):
            height=float(rng.uniform(.001,s.roughness_m));u=s.center-.66+i*.12
            box(i,u,height/2,[.061,.16,height/2])
    elif s.terrain=='step':
        box(0,s.center,s.step_height_m/2,[.25,.16,s.step_height_m/2])
    return compile_spec(spec,s)


def bank(n,stage=3,seed=730000,scenario=None):
    scenarios=list(scenario) if isinstance(scenario,(list,tuple)) else [scenario or sample_terrain(seed+i,stage) for i in range(n)]
    if len(scenarios)!=n:raise ValueError('固定地形场景数量与环境数不一致')
    return batch([model(s) for s in scenarios],scenarios)
