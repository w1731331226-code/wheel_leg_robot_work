"""GPU驻留地面环境：控制、奖励、观测、终止；每40物理步才与Python通信。"""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'wheelleg_ppo/tools')]
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw
from mujoco_warp._src.types import vec5
from native.controller import control,control_physical,constants,D,fk,polar_jac,allowed
import wheelleg_sim as sim
from native.models import bank
from native.terrain import HEIGHT_115_MIN,HEIGHT_115_GEOMETRIC_MIN
from training_contract import TASK_CONTRACT_VERSION
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym

PUBLIC_ACTUATOR_GAIN_UPPER=(1.,1.,1.,1.,1.05,1.05)


@wp.kernel
def begin(reward:wp.array[D]):
    reward[wp.tid()]=D(0)


@wp.kernel
def command_step(state:wp.array2d[D],param:wp.array2d[D],command:wp.array[D],active:wp.array[int],
                 q:wp.array2d[float],vel:wp.array2d[float],warm:wp.array2d[float],
                 held_q:wp.array2d[float],held_v:wp.array2d[float],held_w:wp.array2d[float],contact_flags:wp.array2d[int]):
    w=wp.tid();v=param[w,0]*wp.clamp((state[w,0]*D(.0005)-D(1)),D(0),D(1))
    if state[w,1]>=D(0):v=D(0)
    command[w]=v;contact_flags[w,0]=0;contact_flags[w,1]=0
    if active[w]:
        for j in range(q.shape[1]):held_q[w,j]=q[w,j]
        for j in range(held_v.shape[1]):held_v[w,j]=vel[w,j];held_w[w,j]=warm[w,j]


@wp.kernel
def reduce_contacts(ncon:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],ids:wp.array[int],flags:wp.array2d[int]):
    j=wp.tid()
    if j>=ncon[0]:return
    w=world[j]
    if w<0 or w>=flags.shape[0]:return
    a=geom[j][0];b=geom[j][1]
    if a!=ids[11] and a!=ids[12] and b!=ids[11] and b!=ids[12]:wp.atomic_or(flags,w,0,1)
    if a==ids[13] or b==ids[13]:wp.atomic_or(flags,w,1,1)
    if a==ids[14] or b==ids[14]:wp.atomic_or(flags,w,1,2)
    terrain=(a>=ids[15] and a<=ids[16]) or (b>=ids[15] and b<=ids[16])
    if terrain and (a==ids[11] or b==ids[11]):wp.atomic_or(flags,w,1,4)
    if terrain and (a==ids[12] or b==ids[12]):wp.atomic_or(flags,w,1,8)


@wp.func
def terrain_attitude(param:wp.array2d[D],qpos:wp.array2d[float],w:int):
    roll=D(0);pitch=D(0)
    if int(param[w,7])==0:return wp.vec2d(D(0),D(0))
    kind=int(param[w,8]);angle=param[w,9];relative=param[w,1]*D(qpos[w,0])-param[w,10]
    if kind==1 and relative>=D(-.65) and relative<D(-.15):pitch=-param[w,1]*angle
    elif kind==1 and relative>D(.15) and relative<=D(.65):pitch=param[w,1]*angle
    elif kind==2 and wp.abs(relative)<=D(.65):roll=angle
    elif kind==3 and relative>=D(-.64) and relative<=D(.64):
        segment=int((relative+D(.64))/D(.32))
        if segment>3:segment=3
        pitch=-param[w,1]*angle
        if segment==1 or segment==3:pitch=param[w,1]*angle
    elif kind==4 and wp.abs(relative)<=D(.65):roll=angle
    return wp.vec2d(roll,pitch)


@wp.func
def attitude_passed(state:wp.array2d[D],param:wp.array2d[D],w:int):
    passed=wp.max(state[w,8],wp.max(state[w,9],state[w,10]))<=D(.08726646259971647)
    if int(param[w,7]):passed=wp.max(state[w,21],wp.max(state[w,22],state[w,23]))<=D(.08726646259971647) and wp.max(state[w,8],state[w,9])<=D(.17453292519943295)
    return passed


@wp.func
def rotate_y(point:wp.vec3d,angle:D):
    c=wp.cos(angle);s=wp.sin(angle)
    return wp.vec3d(c*point[0]+s*point[2],point[1],-s*point[0]+c*point[2])


@wp.func
def wheel_center(qpos:wp.array2d[float],ids:wp.array[int],offsets:wp.array3d[wp.vec3d],w:int,side:int):
    # Use the actual passive-joint chain, not ideal closed-link FK or stale geom_xpos.
    alpha=D(qpos[w,ids[2*side]]);passive=D(qpos[w,ids[17+side]])
    local=offsets[w,side,0]+rotate_y(offsets[w,side,1],alpha)+rotate_y(offsets[w,side,2],alpha+passive)
    rotation=wp.quatd(D(qpos[w,4]),D(qpos[w,5]),D(qpos[w,6]),D(qpos[w,3]))
    return wp.vec3d(D(qpos[w,0]),D(qpos[w,1]),D(qpos[w,2]))+wp.quat_rotate(rotation,local)


@wp.func
def physical_passed(state:wp.array2d[D],limit:D,w:int):
    # Every scored substep must have actual geometry and actuator evidence.
    return (state[w,37]==state[w,0] and state[w,37]>D(0) and
            wp.min(state[w,31],state[w,32])>=limit and state[w,33]>=D(0) and
            wp.max(state[w,35],state[w,36])<=D(1.e-6))


@wp.kernel
def collect_physical(q:wp.array2d[float],pre_v:wp.array2d[float],force:wp.array2d[float],ctrl:wp.array2d[float],
                     active:wp.array[int],ids:wp.array[int],joints:wp.array[int],limits:wp.array3d[D],
                     offsets:wp.array3d[wp.vec3d],state:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    for side in range(2):
        k=4*side
        alpha=D(q[w,joints[k]]);beta=D(q[w,joints[k+1]])
        pa=D(q[w,joints[k+2]]);pc=D(q[w,joints[k+3]])
        a=offsets[w,side,0]+rotate_y(offsets[w,side,1],alpha)+rotate_y(offsets[w,side,2],alpha+pa)
        b=offsets[w,side,3]+rotate_y(offsets[w,side,4],beta)+rotate_y(offsets[w,side,5],beta+pc)
        middle=(offsets[w,side,0]+offsets[w,side,3])/D(2)
        state[w,31]=wp.min(state[w,31],wp.length(a-middle))
        state[w,32]=wp.min(state[w,32],wp.length(b-middle))
        state[w,34]=wp.max(state[w,34],wp.length(a-b))
    for j in range(8):
        value=D(q[w,joints[j]])
        state[w,33]=wp.min(state[w,33],wp.min(value-limits[w,j,0],limits[w,j,1]-value))
    for j in range(6):
        bound=allowed(D(pre_v[w,ids[4+j]]),j<4)
        f=D(force[w,j]);u=D(ctrl[w,j])
        if not wp.isfinite(f) or not wp.isfinite(u):
            state[w,35]=D(1.e30);state[w,36]=D(1.e30)
        else:
            state[w,35]=wp.max(state[w,35],wp.abs(f)-bound)
            state[w,36]=wp.max(state[w,36],wp.abs(u)-bound)
    state[w,37]=state[w,37]+D(1)


@wp.kernel
def after(qpos:wp.array2d[float],qvel:wp.array2d[float],sensors:wp.array2d[float],
          warm:wp.array2d[float],clock:wp.array[float],contact_flags:wp.array2d[int],
          ids:wp.array[int],param:wp.array2d[D],cmd:wp.array[D],state:wp.array2d[D],controller_state:wp.array2d[D],
          diag:wp.array2d[D],last_residual:wp.array2d[D],active:wp.array[int],done:wp.array[int],
          reward:wp.array[D],obs:wp.array2d[float],history:wp.array3d[float],
          stopped_q:wp.array2d[float],stopped_v:wp.array2d[float],stopped_w:wp.array2d[float],wheel_offsets:wp.array3d[wp.vec3d]):
    w=wp.tid()
    if active[w]==0:
        for j in range(qpos.shape[1]):qpos[w,j]=stopped_q[w,j]
        for j in range(qvel.shape[1]):qvel[w,j]=stopped_v[w,j];warm[w,j]=stopped_w[w,j]
        clock[w]=float(state[w,0]*D(.0005))
        return
    invalid=bool(False)
    for j in range(qpos.shape[1]):
        if not wp.isfinite(qpos[w,j]):invalid=True
    for j in range(qvel.shape[1]):
        if not wp.isfinite(qvel[w,j]):invalid=True
    if diag[w,14]>D(1):invalid=True
    if invalid:
        active[w]=0;done[w]=1;reward[w]=reward[w]-D(10);state[w,20]=state[w,20]-D(10)
        for j in range(qpos.shape[1]):qpos[w,j]=stopped_q[w,j]
        for j in range(qvel.shape[1]):qvel[w,j]=stopped_v[w,j];warm[w,j]=stopped_w[w,j]
        clock[w]=float(state[w,0]*D(.0005))
        return
    state[w,0]=state[w,0]+D(1);t=state[w,0]*D(.0005);clock[w]=float(t)
    qw=D(qpos[w,3]);qx=D(qpos[w,4]);qy=D(qpos[w,5]);qz=D(qpos[w,6])
    roll=wp.atan2(D(2)*(qw*qx+qy*qz),D(1)-D(2)*(qx*qx+qy*qy))
    pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    reference=terrain_attitude(param,qpos,w);roll_error=roll-reference[0];pitch_error=pitch-reference[1]
    vx=wp.cos(yaw)*D(qvel[w,0])+wp.sin(yaw)*D(qvel[w,1]);err=vx-cmd[w]
    state[w,8]=wp.max(state[w,8],wp.abs(roll));state[w,9]=wp.max(state[w,9],wp.abs(pitch));state[w,10]=wp.max(state[w,10],wp.abs(yaw))
    state[w,21]=wp.max(state[w,21],wp.abs(roll_error));state[w,22]=wp.max(state[w,22],wp.abs(pitch_error));state[w,23]=wp.max(state[w,23],wp.abs(yaw))
    state[w,11]=state[w,11]+roll*roll*D(.0005);state[w,12]=state[w,12]+pitch*pitch*D(.0005);state[w,13]=state[w,13]+yaw*yaw*D(.0005)
    if cmd[w]!=D(0):state[w,4]=state[w,4]+err*err*D(.0005);state[w,5]=state[w,5]+D(.0005)
    touch=int(state[w,14])|contact_flags[w,1];nonwheel=contact_flags[w,0]!=0
    state[w,14]=D(touch)
    left_wheel=wheel_center(qpos,ids,wheel_offsets,w,0);right_wheel=wheel_center(qpos,ids,wheel_offsets,w,1)
    state[w,24]=param[w,1]*left_wheel[0]-param[w,10];state[w,25]=param[w,1]*right_wheel[0]-param[w,10]
    exited=int(param[w,6])==0 or wp.min(state[w,24],state[w,25])>=param[w,12]
    state[w,26]=D(0)
    if exited:state[w,26]=D(1)
    penalty=D(0)
    for j in range(6):
        scale=D(4.5)
        if j<4:scale=D(40)
        residual=diag[w,6+j]/scale;smooth=(residual-last_residual[w,j])/D(.0005)
        penalty=penalty+D(.05)*residual*residual+D(.0001)*smooth*smooth
        last_residual[w,j]=residual
    value=D(.0005)*(wp.exp(-(err/D(.25))*(err/D(.25)))-(roll_error*roll_error+pitch_error*pitch_error+yaw*yaw)/(D(.08726646)*D(.08726646))-penalty)
    reward[w]=reward[w]+value;state[w,20]=state[w,20]+value
    if state[w,1]<D(0) and param[w,1]*D(qpos[w,0])>=param[w,2] and exited:
        state[w,1]=t;state[w,2]=D(qpos[w,0]);state[w,3]=D(qpos[w,1])
    if state[w,1]>=D(0):
        dx=D(qpos[w,0])-state[w,2];dy=D(qpos[w,1])-state[w,3]
        state[w,6]=wp.max(state[w,6],wp.sqrt(dx*dx+dy*dy))
        if t-state[w,1]>=D(1.5):state[w,7]=wp.max(state[w,7],wp.sqrt(D(qvel[w,0])*D(qvel[w,0])+D(qvel[w,1])*D(qvel[w,1])))
    next_cmd=param[w,0]*wp.clamp(t-D(1),D(0),D(1))
    if state[w,1]>=D(0):next_cmd=D(0)
    slot=int(state[w,0])%history.shape[1]
    history[w,slot,0]=float(roll);history[w,slot,1]=float(pitch);history[w,slot,2]=float(yaw)
    for j in range(3):history[w,slot,3+j]=sensors[w,ids[10]+j]
    history[w,slot,6]=float(vx);history[w,slot,7]=float(-wp.sin(yaw)*D(qvel[w,0])+wp.cos(yaw)*D(qvel[w,1]));history[w,slot,8]=qvel[w,2]
    history[w,slot,9]=float(next_cmd);history[w,slot,10]=float(vx-next_cmd);history[w,slot,11]=float(yaw)
    if int(param[w,14]):history[w,slot,11]=float(param[w,13]-D(.3))
    for j in range(4):history[w,slot,12+j]=qpos[w,ids[j]];history[w,slot,16+j]=qvel[w,ids[4+j]]
    history[w,slot,20]=qvel[w,ids[8]];history[w,slot,21]=qvel[w,ids[9]]
    for side in range(2):
        qa=D(qpos[w,ids[2*side]]);qb=D(qpos[w,ids[2*side+1]])
        jac=polar_jac(qa,qb);r=fk(qa,qb)
        history[w,slot,22+side]=float(r[3])
        history[w,slot,24+side]=float(jac[0,0]*D(qvel[w,ids[4+2*side]])+jac[1,0]*D(qvel[w,ids[5+2*side]]))
    for j in range(6):
        scale=D(4.5)
        if j<4:scale=D(40)
        history[w,slot,26+j]=float(diag[w,6+j]/scale)
    delayed=(int(state[w,0])-int(param[w,4])+history.shape[1])%history.shape[1]
    for j in range(32):obs[w,j]=history[w,delayed,j]
    if param[w,15]>D(0):
        left_length=fk(D(qpos[w,ids[0]]),D(qpos[w,ids[1]]))[3]
        right_length=fk(D(qpos[w,ids[2]]),D(qpos[w,ids[3]]))[3]
        state[w,30]=wp.min(state[w,30],wp.min(left_length,right_length))
    if int(param[w,14]) and t>=D(1):
        height_error=(D(history[w,slot,22])+D(history[w,slot,23]))/D(2)-param[w,13]
        state[w,28]=state[w,28]+height_error*height_error*D(.0005)
        state[w,29]=state[w,29]+D(.0005)
    reason=int(0)
    if not wp.isfinite(qpos[w,0]) or not wp.isfinite(qpos[w,2]):reason=1
    elif wp.max(wp.abs(roll),wp.abs(pitch))>D(.6981317007977318) or qpos[w,2]<.02:reason=2
    elif nonwheel:reason=3
    elif diag[w,14]==D(1):reason=4
    elif state[w,1]>=D(0) and t-state[w,1]>=D(2)-D(1.e-10):reason=5
    elif state[w,1]<D(0) and t>=param[w,3]-D(1.e-10):reason=6
    # Training only: an irreversible evaluation failure has no recoverable success.
    # Full-trajectory evaluation leaves this flag off and preserves every safety gate.
    if reason==0 and int(param[w,11]) and not attitude_passed(state,param,w):reason=7
    if diag[w,12]<D(1)-D(1.e-10):state[w,16]=state[w,16]+D(1)
    state[w,17]=state[w,17]+diag[w,12]
    state[w,18]=state[w,18]+diag[w,13]
    if reason:
        done[w]=reason;active[w]=0
        attitude=attitude_passed(state,param,w)
        terrain_passed=reason==5 and (touch&int(param[w,6]))==int(param[w,6])
        state[w,27]=D(0)
        if terrain_passed and exited:state[w,27]=D(1)
        success=state[w,27]>D(0) and (touch&int(param[w,5]))==int(param[w,5]) and attitude and state[w,5]>D(0)
        if state[w,5]>D(0):success=success and wp.sqrt(state[w,4]/state[w,5])<=D(.2)*wp.abs(param[w,0])
        success=success and state[w,6]<=D(.6) and state[w,7]<=D(.03)
        if int(param[w,14]):
            success=success and state[w,29]>D(0) and wp.sqrt(state[w,28]/state[w,29])<=D(.02)
            success=success and wp.abs((D(history[w,slot,22])+D(history[w,slot,23]))/D(2)-param[w,13])<=D(.02)
            if state.shape[1]>31:success=success and physical_passed(state,param[w,15],w)
            elif param[w,15]>D(0):success=success and state[w,30]>=param[w,15]
        state[w,19]=D(0)
        if success:state[w,19]=D(1);reward[w]=reward[w]+D(10);state[w,20]=state[w,20]+D(10)
        else:reward[w]=reward[w]-D(10);state[w,20]=state[w,20]-D(10)
        for j in range(qpos.shape[1]):stopped_q[w,j]=qpos[w,j]
        for j in range(qvel.shape[1]):stopped_v[w,j]=qvel[w,j];stopped_w[w,j]=warm[w,j]


@wp.kernel
def reset_rows(mask:wp.array[int],q0:wp.array2d[float],q:wp.array2d[float],v:wp.array2d[float],warm:wp.array2d[float],clock:wp.array[float],sensors:wp.array2d[float],
               control_state:wp.array2d[D],state:wp.array2d[D],residual:wp.array2d[D],active:wp.array[int],done:wp.array[int],
               obs0:wp.array2d[float],obs:wp.array2d[float],history:wp.array3d[float],targets:wp.array2d[float],reference:wp.array2d[D]):
    w=wp.tid()
    if not mask[w]:return
    for j in range(q.shape[1]):q[w,j]=q0[w,j]
    for j in range(v.shape[1]):v[w,j]=0.;warm[w,j]=0.
    clock[w]=0.;active[w]=1;done[w]=0
    for j in range(sensors.shape[1]):sensors[w,j]=0.
    for j in range(control_state.shape[1]):control_state[w,j]=D(0)
    control_state[w,1]=reference[w,2];control_state[w,12]=D(-1)
    for j in range(state.shape[1]):state[w,j]=D(0)
    state[w,1]=D(-1)
    state[w,30]=D(1)
    if state.shape[1]>31:
        state[w,31]=D(1);state[w,32]=D(1);state[w,33]=D(1.e30)
    for j in range(6):residual[w,j]=D(0)
    for j in range(targets.shape[1]):targets[w,j]=0.
    for j in range(32):
        obs[w,j]=obs0[w,j]
        for t in range(history.shape[1]):history[w,t,j]=obs0[w,j]


class NativeEnv(VecEnv):
    def __init__(self,n=128,stage=3,seed=730000,scenario=None,bank_factory=bank,yaw_config=(.4,2.,.24,.3),residual_scale=1.,residual_mode='diff3',terminate_on_attitude_failure=False,project_clipped_base=False,grouped_residual=False,height_conditioned=False,height_design='legacy',height_safety=None):
        if not isinstance(n,int) or not 1 <= n <= 1024:raise ValueError('用户限制：批量环境数须为1～1024')
        if type(project_clipped_base) is not bool:raise ValueError('基础限幅后残差投影开关须为布尔值')
        self.project_clipped_base=project_clipped_base
        if type(grouped_residual) is not bool or (grouped_residual and (project_clipped_base or residual_mode!='diff3')):raise ValueError('分组残差仅允许独立的diff3试验')
        self.grouped_residual=grouped_residual
        if type(terminate_on_attitude_failure) is not bool:raise ValueError('训练姿态终止开关须为布尔值')
        self.terminate_on_attitude_failure=terminate_on_attitude_failure
        if residual_mode not in ('diff3','virtual6'):raise ValueError('无效残差模式')
        self.residual_mode=residual_mode;self.action_dim=3 if residual_mode=='diff3' else 6
        if not np.isscalar(residual_scale) or not np.isfinite(residual_scale) or not 0<=residual_scale<=1:raise ValueError('残差强度须为0～1有限数')
        self.residual_scale=float(residual_scale)
        wp.init();wp.set_device('cuda:0')
        self.num_envs=n;self.stage=stage
        self.cpu,self.model,self.data,self.scenarios=bank_factory(n,stage,seed,scenario)
        if type(height_conditioned) is not bool:raise ValueError('多高度开关须为布尔值')
        if any(hasattr(s,'stand_height_m')!=height_conditioned for s in self.scenarios):
            raise ValueError('多高度场景与多高度观测模式必须同时启用')
        self.height_conditioned=height_conditioned
        if height_design not in ('legacy','range115') or (height_design=='range115' and not height_conditioned):
            raise ValueError('高度控制设计须与多高度模式匹配')
        self.height_design=height_design
        self.height_safety=('physical_v1' if height_design=='range115' else 'legacy_fk') if height_safety is None else height_safety
        if self.height_safety not in ('legacy_fk','physical_v1') or (self.height_safety=='physical_v1' and height_design!='range115'):
            raise ValueError('真实几何安全契约只用于range115独立任务')
        self.stand_heights=np.asarray([s.stand_height_m if height_conditioned else sim.L_STAND for s in self.scenarios],dtype=float)
        low=HEIGHT_115_MIN if height_design=='range115' else sim.L_SQUAT_MIN
        if not np.isfinite(self.stand_heights).all() or np.any((self.stand_heights<low)|(self.stand_heights>sim.L_MAX)):
            raise ValueError(f'目标腿长超出{low:.3f}～{sim.L_MAX:.3f}m')
        if len(yaw_config)!=4 or not np.isfinite(yaw_config).all() or min(yaw_config)<=0:raise ValueError('无效偏航控制参数')
        self.yaw_config=tuple(float(x) for x in yaw_config);self.k=constants(self.cpu,n,self.yaw_config,self.action_dim,height_design)
        self.control_kernel=control
        self.control_extra=[]
        self.actuator_gain_upper=None
        if self.height_safety=='physical_v1':
            actuator_gains=self.model.actuator_gainprm.numpy()[:,:,0]
            if not np.isfinite(actuator_gains).all() or np.any(actuator_gains<=0):raise ValueError('实际扭矩分配需要有限正执行器增益')
            if (np.any(self.cpu.actuator_gaintype!=mujoco.mjtGain.mjGAIN_FIXED) or
                np.any(self.cpu.actuator_biastype!=mujoco.mjtBias.mjBIAS_NONE) or
                np.any(self.cpu.actuator_dyntype!=mujoco.mjtDyn.mjDYN_NONE) or
                not np.allclose(self.cpu.actuator_gear,np.tile([1.,0.,0.,0.,0.,0.],(6,1)))):
                raise ValueError('实际扭矩分配只支持无动态和偏置的直接力矩电机')
            # Frozen Scenario domain: wheel drive difference <=5%. Do not give the
            # controller each randomized world's hidden actuator calibration.
            self.actuator_gain_upper=np.array(PUBLIC_ACTUATOR_GAIN_UPPER)
            if np.any(actuator_gains>self.actuator_gain_upper+1e-7):raise ValueError('执行器增益超出公开力矩分配上界')
            self.control_kernel=control_physical
            self.control_extra=[wp.array(np.tile(self.actuator_gain_upper,(n,1)),dtype=D)]
        ids=self.k['ids'].numpy().tolist()+[self.cpu.geom(x).id for x in ('wheel_collide_L','wheel_collide_R','bump_L','bump_R')]
        terrain=mujoco.mj_name2id(self.cpu,mujoco.mjtObj.mjOBJ_GEOM,'terrain_00')
        ids += [terrain,mujoco.mj_name2id(self.cpu,mujoco.mjtObj.mjOBJ_GEOM,'terrain_15')] if terrain>=0 else [-1,-1]
        ids += [int(self.cpu.jnt_qposadr[self.cpu.joint('passA_'+side).id]) for side in ('L','R')]
        self.ids=wp.array(ids,dtype=wp.int32)
        self.wheel_geom_ids=np.asarray(ids[11:13],dtype=int)
        chains=[]
        for side in ('L','R'):
            bodies=[self.cpu.body(name).id for name in ('leg'+side,'kneeA_'+side,'wheel'+side)];chains.append(bodies)
            if list(self.cpu.body_parentid[bodies[1:]])!=bodies[:2] or self.cpu.body_parentid[bodies[0]]!=self.cpu.jnt_bodyid[0]:raise ValueError('不支持的轮心关节链拓扑')
            joints=[self.cpu.joint(name).id for name in ('alpha'+side,'passA_'+side)]
            if not np.allclose(self.cpu.jnt_axis[joints],[0,1,0]) or not np.allclose(self.cpu.jnt_pos[joints],0):raise ValueError('轮心链需过原点的Y轴铰链')
        if not np.allclose(self.model.body_quat.numpy()[:,chains],[1,0,0,0]):raise ValueError('轮心链需单位body旋转')
        if not np.allclose(self.model.geom_pos.numpy()[:,self.wheel_geom_ids],0):raise ValueError('轮碰撞几何必须以轮轴为中心')
        self.wheel_offsets=wp.array(self.model.body_pos.numpy()[:,chains],dtype=wp.vec3d)
        # Actual world-aligned box extents, including rotated thickness and entry/exit ramps.
        ends=[None]*n
        if terrain>=0:
            positions=self.data.geom_xpos.numpy()[:,ids[15]:ids[16]+1]
            matrices=self.data.geom_xmat.numpy()[:,ids[15]:ids[16]+1]
            sizes=self.model.geom_size.numpy()[:,ids[15]:ids[16]+1]
            radii=np.sum(abs(matrices[:,:,0,:])*sizes,axis=-1)
            for i,s in enumerate(self.scenarios):
                if getattr(s,'terrain','legacy')=='legacy':continue
                present=positions[i,:,2]>-1.
                if not present.any():raise ValueError('目标地形没有有效几何')
                ends[i]=float(np.max(np.sign(s.speed)*positions[i,present,0]+radii[i,present])-s.center)
        p=[]
        self.required_contact_masks=[];self.required_terrain_contact_masks=[];self.required_terrain_end=[];self.relative_attitude=[]
        for i,s in enumerate(self.scenarios):
            goal=s.center+abs(s.offset)/2+.75
            end=ends[i]
            if end is not None:goal=max(goal,s.center+end+.15)
            required=(1 if s.height_l else 0)|(2 if s.height_r else 0);terrain=12 if getattr(s,'terrain','legacy')!='legacy' else 0
            if getattr(s,'terrain','legacy')=='single_side_ramp':terrain=8 if s.grade_deg>0 else 4
            relative=bool(getattr(s,'relative_attitude',False));kind={'ramp':1,'cross_slope':2,'rolling_slope':3,'split_level':4}.get(getattr(s,'terrain','legacy'),0)
            self.required_contact_masks.append(required);self.required_terrain_contact_masks.append(terrain);self.required_terrain_end.append(end);self.relative_attitude.append(relative)
            p.append([s.speed,np.sign(s.speed),goal,1.5+1.5*goal/abs(s.speed),round(s.delay_ms*2),required,terrain,relative,kind,np.deg2rad(getattr(s,'grade_deg',0.)),s.center,terminate_on_attitude_failure,end or 0.,self.stand_heights[i],height_conditioned,HEIGHT_115_GEOMETRIC_MIN if height_design=='range115' else 0.])
        self.task_goals=[row[2] for row in p]
        self.param=wp.array(p,dtype=D);self.command=wp.zeros(n,dtype=D)
        self.active=wp.ones(n,dtype=wp.int32);self.done=wp.zeros(n,dtype=wp.int32)
        self.state=wp.zeros((n,38 if self.height_safety=='physical_v1' else 31),dtype=D);self.diag=wp.zeros((n,21 if height_conditioned else 15),dtype=D)
        self.residual=wp.zeros((n,6),dtype=D);self.reward=wp.zeros(n,dtype=D);self.contact_flags=wp.zeros((n,2),dtype=wp.int32)
        self.obs=wp.zeros((n,32));self.history=wp.zeros((n,max(round(s.delay_ms*2) for s in self.scenarios)+1,32))
        self.targets=wp.zeros((n,self.action_dim));self.stopped_q=wp.zeros((n,self.cpu.nq));self.stopped_v=wp.zeros((n,self.cpu.nv));self.stopped_w=wp.zeros((n,self.cpu.nv))
        q0=np.tile(self.cpu.key_qpos[self.cpu.keyframe('stand').id],(n,1))
        references=np.tile([0.,0.,sim.L_STAND],(n,1))
        passive_ids=np.asarray([self.cpu.jnt_qposadr[self.cpu.joint(name).id] for name in ('passA_L','passC_L','passA_R','passC_R')])
        avec=self.cpu.body_pos[self.cpu.body('wheelL').id][[0,2]]
        cvec=self.cpu.site_pos[self.cpu.site('couplerB_L_end').id][[0,2]]
        for i,h in enumerate(self.stand_heights):
            if h!=sim.L_STAND:
                alpha,beta=sim.ik(h)
                bpos=sim.L1*np.array([np.cos(sim.PHI1_STAND-alpha),np.sin(sim.PHI1_STAND-alpha)])
                dpos=np.array([sim.L5,0.])+sim.L4*np.array([np.cos(sim.PHI4_STAND-beta),np.sin(sim.PHI4_STAND-beta)])
                cpos=np.array([sim.L5/2,-h])
                passive_a=np.arctan2(avec[1],avec[0])-np.arctan2(*(cpos-bpos)[::-1])-alpha
                passive_c=np.arctan2(cvec[1],cvec[0])-np.arctan2(*(cpos-dpos)[::-1])-beta
                q0[i,2]+=h-sim.L_STAND
                q0[i,np.asarray(ids[:4])]=[alpha,beta,alpha,beta]
                q0[i,passive_ids]=[passive_a,passive_c,passive_a,passive_c]
                references[i]=[alpha,beta,h]
        self.k['reference'].assign(references)
        self.q0=wp.array(q0,dtype=wp.float32)
        initial=np.zeros((n,32),dtype=np.float32)
        q=self.q0.numpy()
        initial[:,12:16]=q[:,np.array(ids[:4])]
        from state_estimation import leg_kinematics
        for i in range(n):
            initial[i,22:24]=[np.linalg.norm(leg_kinematics(q[i,np.array(ids[:2])],np.zeros(2))[0]),np.linalg.norm(leg_kinematics(q[i,np.array(ids[2:4])],np.zeros(2))[0])]
        if height_conditioned:initial[:,11]=self.stand_heights-sim.L_STAND
        self.obs0=wp.array(initial,dtype=wp.float32)
        self.mask=wp.ones(n,dtype=wp.int32)
        self.reset_args=[self.mask,self.q0,self.data.qpos,self.data.qvel,self.data.qacc_warmstart,self.data.time,self.data.sensordata,self.k['state'],self.state,self.residual,self.active,self.done,self.obs0,self.obs,self.history,self.targets,self.k['reference']]
        if self.height_safety=='physical_v1':
            names=('alphaL','betaL','passA_L','passC_L','alphaR','betaR','passA_R','passC_R')
            joint_ids=np.array([self.cpu.joint(name).id for name in names])
            geometry_ids=[[self.cpu.body(name).id for name in ('leg'+s,'kneeA_'+s,'wheel'+s,'leg'+s+'_D','kneeB_'+s)] for s in ('L','R')]
            # Both chains use origin-centered Y hinges and identity body rotations.
            if not np.allclose(self.cpu.jnt_axis[joint_ids],[0,1,0]) or not np.allclose(self.cpu.jnt_pos[joint_ids],0) or not np.all(self.cpu.jnt_limited[joint_ids]):
                raise ValueError('真实链安全契约需受限的原点Y轴关节')
            if not np.allclose(self.model.body_quat.numpy()[:,geometry_ids],[1,0,0,0]):raise ValueError('真实链安全契约需单位body旋转')
            offsets=np.empty((n,2,6,3))
            offsets[:,:,:5]=self.model.body_pos.numpy()[:,geometry_ids]
            offsets[:,:,5]=self.model.site_pos.numpy()[:,[self.cpu.site('couplerB_'+s+'_end').id for s in ('L','R')]]
            self.physical_args=[self.data.qpos,self.stopped_v,self.data.actuator_force,self.data.ctrl,self.active,self.ids,
                wp.array(self.cpu.jnt_qposadr[joint_ids],dtype=wp.int32),wp.array(self.model.jnt_range.numpy()[:,joint_ids],dtype=D),
                wp.array(offsets,dtype=wp.vec3d),self.state]
        wp.launch(reset_rows,n,self.reset_args)
        mjw.forward(self.model,self.data)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,n,[self.reward])
            for _ in range(40):
                wp.launch(command_step,n,[self.state,self.param,self.command,self.active,self.data.qpos,self.data.qvel,self.data.qacc_warmstart,self.stopped_q,self.stopped_v,self.stopped_w,self.contact_flags])
                wp.launch(self.control_kernel,n,[self.data.qpos,self.data.qvel,self.data.sensordata,self.targets,self.command,self.active,
                    self.k['state'],self.ids,self.k['heights'],self.k['gains'],self.k['feed'],self.k['angles'],self.k['reference'],self.k['yaw'],self.data.ctrl,self.diag,int(self.project_clipped_base),int(self.grouped_residual)]+self.control_extra,block_dim=32)
                mjw.step(self.model,self.data)
                wp.launch(reduce_contacts,self.data.naconmax,[self.data.nacon,self.data.contact.worldid,self.data.contact.geom,self.ids,self.contact_flags])
                if self.height_safety=='physical_v1':wp.launch(collect_physical,n,self.physical_args)
                wp.launch(after,n,[self.data.qpos,self.data.qvel,self.data.sensordata,self.data.qacc_warmstart,self.data.time,self.contact_flags,self.ids,self.param,self.command,self.state,self.k['state'],self.diag,
                    self.residual,self.active,self.done,self.reward,self.obs,self.history,self.stopped_q,self.stopped_v,self.stopped_w,self.wheel_offsets],block_dim=32)
        self.graph=capture.graph
        super().__init__(n,gym.spaces.Box(-np.inf,np.inf,(32,),dtype=np.float32),gym.spaces.Box(-1.,1.,(self.action_dim,),dtype=np.float32))

    def reset(self):
        self.mask.fill_(1);wp.launch(reset_rows,self.num_envs,self.reset_args);mjw.forward(self.model,self.data)
        return self.obs.numpy().copy()

    def step_async(self,actions):
        a=np.asarray(actions,dtype=np.float32)
        if a.shape!=(self.num_envs,self.action_dim) or not np.isfinite(a).all() or np.any(abs(a)>1.000001):raise ValueError('无效动作')
        self.targets.assign(a*self.residual_scale);wp.capture_launch(self.graph)

    def step_wait(self):
        obs=self.obs.numpy().copy();reward=self.reward.numpy().copy();reasons=self.done.numpy();done=reasons!=0
        if np.any(self.data.overflow.numpy()):raise RuntimeError('GPU容量溢出')
        if not np.isfinite(obs).all() or not np.isfinite(reward).all():raise FloatingPointError('原生GPU非有限状态')
        infos=[{} for _ in range(self.num_envs)]
        if done.any():
            states=self.state.numpy()
            for i in np.flatnonzero(done):
                infos[i]=dict(terminal_observation=obs[i].copy(),reason=['ongoing','invalid','fall','nonwheel_contact','invalid_mapping','completed','timeout','attitude_failure'][int(reasons[i])],physical_steps=int(states[i,0]),arrival_s=float(states[i,1]) if states[i,1]>=0 else None,
                    rms_deg=(np.sqrt(states[i,11:14]/max(states[i,0]*.0005,.0005))*180/np.pi).tolist(),
                    velocity_rmse=float(np.sqrt(states[i,4]/states[i,5])) if states[i,5]>0 else None,stop_distance_m=float(states[i,6]),tail_speed_m_s=float(states[i,7]),success=bool(states[i,19]),
                    duration_s=states[i,0]*.0005,episode=dict(r=float(states[i,20]),l=int(np.ceil(states[i,0]/40))),peak_deg=(states[i,8:11]*180/np.pi).tolist(),TimeLimit_truncated=False)
                touched=int(states[i,14]);required_terrain=self.required_terrain_contact_masks[i]
                wheel_progress=states[i,24:26].tolist();end=self.required_terrain_end[i]
                infos[i]['required_contact_mask']=self.required_contact_masks[i];infos[i]['touched_contact_mask']=touched&3
                infos[i]['required_terrain_contact_mask']=required_terrain;infos[i]['touched_terrain_contact_mask']=touched&12
                infos[i]['terrain_passed']=int(reasons[i])==5 and (touched&required_terrain)==required_terrain
                infos[i]['terrain_required_end_m']=end;infos[i]['wheel_progress_m']=wheel_progress
                infos[i]['terrain_entry_profile']='ramped' if getattr(self.scenarios[i],'transition_run_m',0.) else 'abrupt_or_original'
                infos[i]['terrain_exit_passed']=bool(states[i,26])
                infos[i]['terrain_evidence_passed']=bool(states[i,27])
                infos[i]['task_contract_version']=TASK_CONTRACT_VERSION
                infos[i]['task_goal_progress_m']=float(self.task_goals[i])
                infos[i]['attitude_mode']='terrain_relative' if self.relative_attitude[i] else 'world'
                infos[i]['relative_peak_deg']=(states[i,21:24]*180/np.pi).tolist()
                infos[i]['residual_limited_steps']=int(states[i,16]);infos[i]['mean_residual_lambda']=float(states[i,17]/max(states[i,0],1));infos[i]['base_infeasible_steps']=int(states[i,18])
                infos[i]['yaw_config']=self.yaw_config
                infos[i]['residual_scale']=self.residual_scale
                infos[i]['residual_mode']=self.residual_mode
                infos[i]['project_clipped_base']=self.project_clipped_base
                infos[i]['grouped_residual']=self.grouped_residual
                infos[i]['control_limit_scope']='nominal_command'
                infos[i]['terminate_on_attitude_failure']=self.terminate_on_attitude_failure
                if self.height_conditioned:
                    infos[i]['target_leg_m']=float(self.stand_heights[i])
                    infos[i]['height_rmse_m']=float(np.sqrt(states[i,28]/states[i,29])) if states[i,29]>0 else None
                    infos[i]['height_tolerance_m']=.02
                    infos[i]['height_design']=self.height_design
                    if self.height_design=='range115':
                        infos[i]['min_leg_m']=float(states[i,30])
                        infos[i]['geometric_limit_m']=HEIGHT_115_GEOMETRIC_MIN
                        infos[i]['geometric_margin_passed']=bool(states[i,30]>=HEIGHT_115_GEOMETRIC_MIN)
                        infos[i]['height_safety_contract']=self.height_safety
                        if self.height_safety=='physical_v1':
                            infos[i]['min_fk_leg_m']=float(states[i,30])
                            infos[i]['min_actual_A_leg_m']=float(states[i,31]);infos[i]['min_actual_B_leg_m']=float(states[i,32])
                            infos[i]['min_leg_m']=float(min(states[i,31],states[i,32]))
                            infos[i]['geometric_margin_passed']=bool(infos[i]['min_leg_m']>=HEIGHT_115_GEOMETRIC_MIN)
                            infos[i]['min_eight_joint_margin_rad']=float(states[i,33]);infos[i]['max_loop_error_m']=float(states[i,34])
                            infos[i]['max_actual_torque_excess_Nm']=float(states[i,35]);infos[i]['max_command_torque_excess_Nm']=float(states[i,36])
                            infos[i]['physical_evidence_steps']=int(states[i,37])
                            infos[i]['physical_safety_passed']=bool(states[i,37]==states[i,0] and states[i,37]>0 and
                                infos[i]['geometric_margin_passed'] and states[i,33]>=0 and max(states[i,35],states[i,36])<=1e-6)
                            infos[i]['control_limit_scope']='actual_torque_and_nominal_command'
                            infos[i]['actuator_gain_upper']=self.actuator_gain_upper.tolist()
                # This deadline is task failure, not an external collection cutoff.
                # SB3 must not add gamma*V(terminal_observation) to its -10 reward.
                infos[i]['TimeLimit.truncated']=False
            self.mask.assign(done.astype(np.int32));wp.launch(reset_rows,self.num_envs,self.reset_args)
            refreshed=self.obs.numpy();obs[done]=refreshed[done]
        return obs,reward,done,infos

    def close(self):pass
    def get_attr(self,name,indices=None):
        return [getattr(self,name,None) for _ in self._get_indices(indices)]
    def set_attr(self,name,value,indices=None):
        if name=='stage' and value!=self.stage:raise ValueError('本实验场景库阶段固定，不支持暗中改变课程')
        setattr(self,name,value)
    def env_method(self,method_name,*args,indices=None,**kwargs):raise NotImplementedError(method_name)
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
