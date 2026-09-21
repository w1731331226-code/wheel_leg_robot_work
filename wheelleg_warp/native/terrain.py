"""Uniform-topology terrain bank for the independent terrain-v1 experiment."""
from dataclasses import asdict,dataclass
import math
import numpy as np
import mujoco
from ppo_env import Scenario,sample_scenario
from native.models import build_spec,compile_spec,batch

TERRAINS=('legacy','ramp','cross_slope','rough','step','mixed')
V3_TERRAINS=TERRAINS+('rolling_slope','multi_step','split_level')
TERRAIN_GEOMS=16


@dataclass(frozen=True)
class TerrainScenario(Scenario):
    terrain:str='legacy'
    grade_deg:float=0.
    roughness_m:float=0.
    step_height_m:float=0.
    terrain_seed:int=0
    relative_attitude:bool=False

    def __post_init__(self):
        # Reuse the frozen base validation without passing terrain's string fields into it.
        Scenario(**{name:getattr(self,name) for name in Scenario.__dataclass_fields__})
        if self.terrain not in V3_TERRAINS or not -5<=self.grade_deg<=5 or not 0<=self.roughness_m<=.012 or not 0<=self.step_height_m<=.03:
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


def sample_terrain_v3(seed,stage=3,split='train'):
    if stage not in (1,2,3) or split not in ('train','development','ood'):raise ValueError('无效terrain-v3课程或划分')
    base=sample_scenario('train',seed,3);rng=np.random.default_rng(np.random.SeedSequence([int(seed),93091,{'train':1,'development':2,'ood':3}[split]]))
    if split=='train':
        choices={1:(TERRAINS,(.3,.1,.1,.1,.1,.3)),2:(V3_TERRAINS,(.3,.08,.08,.08,.08,.2,.06,.06,.06)),3:(V3_TERRAINS,(.3,.08,.08,.08,.08,.15,.08,.08,.07))}[stage]
        terrain=str(rng.choice(choices[0],p=choices[1]))
    else:terrain=str(rng.choice(V3_TERRAINS))
    values=asdict(base)
    if terrain not in ('legacy','mixed'):values.update(height_l=0.,height_r=0.)
    grade=0.;rough=0.;step=0.;limits=(3.,5.) if split=='ood' else (1.,3.)
    if terrain in ('ramp','rolling_slope'):grade=float(rng.uniform(*limits))
    elif terrain in ('cross_slope','split_level'):grade=float(rng.choice((-1,1))*rng.uniform(*limits))
    if terrain in ('rough','mixed'):rough=float(rng.uniform(.006,.010) if split=='ood' else rng.uniform(.002,.006))
    if terrain=='step':step=float(rng.uniform(.020,.030) if split=='ood' else rng.uniform(.005,.020))
    elif terrain=='multi_step':step=float(rng.uniform(.008,.012) if split=='ood' else rng.uniform(.004,.008))
    elif terrain=='split_level':step=float(.3*math.tan(math.radians(abs(grade))))
    return TerrainScenario(**values,terrain=terrain,grade_deg=grade,roughness_m=rough,step_height_m=step,terrain_seed=int(seed),relative_attitude=True)


def _quat(axis,angle):
    out=np.zeros(4);mujoco.mju_axisAngle2Quat(out,np.asarray(axis,dtype=float),angle);return out


def model(s):
    spec=build_spec(s);tiles=[]
    for i in range(TERRAIN_GEOMS):
        tiles.append(spec.worldbody.add_geom(name=f'terrain_{i:02d}',type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0,0,-10],size=[.05,.16,.001],friction=[(s.mu_l+s.mu_r)/2,.02,.001]))
    direction=1 if s.speed>0 else -1;thickness=.012
    def box(index,u,z,size,quat=(1,0,0,0),y=0.):
        g=tiles[index];g.pos=[direction*u,y,z];g.size=size;g.quat=quat
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
        offsets=np.arange(-.66,.67,.12)
        if s.terrain=='mixed':offsets=np.r_[np.arange(-.96,-.35,.12),np.arange(.36,.97,.12)]
        for i,offset in enumerate(offsets):
            height=float(rng.uniform(.001,s.roughness_m));u=s.center+float(offset)
            box(i,u,height/2,[.061,.16,height/2])
    elif s.terrain=='step':
        box(0,s.center,s.step_height_m/2,[.25,.16,s.step_height_m/2])
    elif s.terrain=='rolling_slope':
        angle=math.radians(abs(s.grade_deg));run=.32;height=0.
        for i,sign in enumerate((1,-1,1,-1)):
            delta=sign*run*math.sin(angle);z=(height+height+delta)/2-math.cos(angle)*thickness/2
            box(i,s.center-.64+run*(i+.5),z,[run/2,.16,thickness/2],_quat((0,1,0),-direction*sign*angle));height+=delta
    elif s.terrain=='multi_step':
        for i,level in enumerate((1,2,3,2,1)):
            height=level*s.step_height_m;box(i,s.center-.55+.22*(i+.5),height/2,[.11,.16,height/2])
    elif s.terrain=='split_level':
        high_right=s.grade_deg>0
        for i,(y,high) in enumerate(((-.09,not high_right),(.09,high_right))):
            height=s.step_height_m if high else .001;box(i,s.center,height/2,[.65,.09,height/2],y=y)
    return compile_spec(spec,s)


def bank(n,stage=3,seed=730000,scenario=None):
    scenarios=list(scenario) if isinstance(scenario,(list,tuple)) else [scenario or sample_terrain(seed+i,stage) for i in range(n)]
    if len(scenarios)!=n:raise ValueError('固定地形场景数量与环境数不一致')
    return batch([model(s) for s in scenarios],scenarios)


def bank_v3(n,stage=3,seed=930000,scenario=None):
    scenarios=list(scenario) if isinstance(scenario,(list,tuple)) else [scenario or sample_terrain_v3(seed+i,stage) for i in range(n)]
    if len(scenarios)!=n:raise ValueError('固定terrain-v3场景数量与环境数不一致')
    return batch([model(s) for s in scenarios],scenarios)
