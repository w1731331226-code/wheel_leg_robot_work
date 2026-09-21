"""GPU场景库：统一拓扑，缺失障碍放到任务永远到不了的地下。"""
from dataclasses import asdict,fields
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from ppo_env import XML,Scenario,sample_scenario
import wheelleg_sim as sim


def build_spec(s):
    spec=mujoco.MjSpec.from_file(XML);sim.hw.configure_spec(spec)
    if abs(s.mass-sim.hw.DESIGN_MASS)>1e-12:spec.geom('chassis_lid').mass+=s.mass-sim.hw.DESIGN_MASS
    direction=np.sign(s.speed)
    for side,sign,height,mu in (('L',-1,s.height_l,s.mu_l),('R',1,s.height_r,s.mu_r)):
        wheel=spec.geom('wheel_collide_'+side);ground=spec.geom('floor')
        weight=wheel.solmix/(wheel.solmix+ground.solmix)
        wheel.solref=weight*wheel.solref+(1-weight)*ground.solref
        wheel.solimp=weight*wheel.solimp+(1-weight)*ground.solimp
        wheel.priority=1;wheel.friction=(mu,.02,.001)
        spec.worldbody.add_geom(name='bump_'+side,type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[direction*(s.center+sign*s.offset/2),sign*sim.hw.TRACK_WIDTH/2,height/2 if height else -10.],
            size=[.25,.035,max(height/2,.0001)],friction=[.8,.02,.001])
    return spec


def compile_spec(spec,s):
    m=spec.compile();m.opt.iterations=s.solver_iterations
    for side,sign in (('L',1),('R',-1)):
        m.actuator_gainprm[m.actuator('motor_wheel'+side).id,0]*=1+sign*s.drive_difference
    return m


def model(s):
    return compile_spec(build_spec(s),s)


def batch(cpu,scenarios):
    """Batch identical model topology; callers only provide the per-world models."""
    n=len(cpu);batched=[f.name for f in fields(mjw.Model) if getattr(f.type,'shape',())[0:1]==('*',)]
    template=mjw.put_model(cpu[0],batch_sizes={k:n for k in batched})
    values={k:[] for k in batched}
    for m in cpu:
        single=mjw.put_model(m)
        for k in batched:values[k].append(getattr(single,k).numpy()[0])
    for k,items in values.items():getattr(template,k).assign(np.stack(items))
    seed_data=mujoco.MjData(cpu[0]);mujoco.mj_resetDataKeyframe(cpu[0],seed_data,cpu[0].keyframe('stand').id)
    mujoco.mj_forward(cpu[0],seed_data)
    data=mjw.put_data(cpu[0],seed_data,nworld=n,nconmax=64,njmax=256)
    # MuJoCo Warp initializes world-body geoms once from cpu[0] and deliberately
    # skips them in later kinematics, so seed every world's static poses here.
    body=cpu[0].geom_bodyid;static=(cpu[0].body_weldid[body]==0)&(cpu[0].body_mocapid[cpu[0].body_rootid[body]]==-1)
    xpos=data.geom_xpos.numpy();xmat=data.geom_xmat.numpy()
    for world,m in enumerate(cpu):
        xpos[world,static]=m.geom_pos[static]
        for geom in np.flatnonzero(static):
            matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,m.geom_quat[geom]);xmat[world,geom]=matrix.reshape(3,3)
    data.geom_xpos.assign(xpos);data.geom_xmat.assign(xmat)
    return cpu[0],template,data,scenarios


def bank(n,stage=3,seed=730000,scenario=None):
    scenarios=[scenario or sample_scenario('train',seed+i,stage) for i in range(n)]
    return batch([model(s) for s in scenarios],scenarios)
