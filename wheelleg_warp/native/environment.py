"""GPU驻留地面环境：控制、奖励、观测、终止；每40物理步才与Python通信。"""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from mujoco_warp._src.types import vec5
from native.controller import control,constants,D,fk,polar_jac
from native.models import bank
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym


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


@wp.kernel
def after(qpos:wp.array2d[float],qvel:wp.array2d[float],sensors:wp.array2d[float],
          warm:wp.array2d[float],clock:wp.array[float],contact_flags:wp.array2d[int],
          ids:wp.array[int],param:wp.array2d[D],cmd:wp.array[D],state:wp.array2d[D],controller_state:wp.array2d[D],
          diag:wp.array2d[D],last_residual:wp.array2d[D],active:wp.array[int],done:wp.array[int],
          reward:wp.array[D],obs:wp.array2d[float],history:wp.array3d[float],
          stopped_q:wp.array2d[float],stopped_v:wp.array2d[float],stopped_w:wp.array2d[float]):
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
    vx=wp.cos(yaw)*D(qvel[w,0])+wp.sin(yaw)*D(qvel[w,1]);err=vx-cmd[w]
    state[w,8]=wp.max(state[w,8],wp.abs(roll));state[w,9]=wp.max(state[w,9],wp.abs(pitch));state[w,10]=wp.max(state[w,10],wp.abs(yaw))
    state[w,11]=state[w,11]+roll*roll*D(.0005);state[w,12]=state[w,12]+pitch*pitch*D(.0005);state[w,13]=state[w,13]+yaw*yaw*D(.0005)
    if cmd[w]!=D(0):state[w,4]=state[w,4]+err*err*D(.0005);state[w,5]=state[w,5]+D(.0005)
    touch=int(state[w,14])|contact_flags[w,1];nonwheel=contact_flags[w,0]!=0
    state[w,14]=D(touch)
    penalty=D(0)
    for j in range(6):
        scale=D(4.5)
        if j<4:scale=D(40)
        residual=diag[w,6+j]/scale;smooth=(residual-last_residual[w,j])/D(.0005)
        penalty=penalty+D(.05)*residual*residual+D(.0001)*smooth*smooth
        last_residual[w,j]=residual
    value=D(.0005)*(wp.exp(-(err/D(.25))*(err/D(.25)))-(roll*roll+pitch*pitch+yaw*yaw)/(D(.08726646)*D(.08726646))-penalty)
    reward[w]=reward[w]+value;state[w,20]=state[w,20]+value
    if state[w,1]<D(0) and param[w,1]*D(qpos[w,0])>=param[w,2]:
        state[w,1]=t;state[w,2]=D(qpos[w,0]);state[w,3]=D(qpos[w,1])
    if state[w,1]>=D(0):
        dx=D(qpos[w,0])-state[w,2];dy=D(qpos[w,1])-state[w,3]
        state[w,6]=wp.max(state[w,6],wp.sqrt(dx*dx+dy*dy))
        if t-state[w,1]>=D(1.5):state[w,7]=wp.max(state[w,7],wp.sqrt(D(qvel[w,0])*D(qvel[w,0])+D(qvel[w,1])*D(qvel[w,1])))
    next_cmd=param[w,0]*wp.clamp(t-D(1),D(0),D(1))
    if state[w,1]>=D(0):next_cmd=D(0)
    slot=int(state[w,0])%21
    history[w,slot,0]=float(roll);history[w,slot,1]=float(pitch);history[w,slot,2]=float(yaw)
    for j in range(3):history[w,slot,3+j]=sensors[w,ids[10]+j]
    history[w,slot,6]=float(vx);history[w,slot,7]=float(-wp.sin(yaw)*D(qvel[w,0])+wp.cos(yaw)*D(qvel[w,1]));history[w,slot,8]=qvel[w,2]
    history[w,slot,9]=float(next_cmd);history[w,slot,10]=float(vx-next_cmd);history[w,slot,11]=float(yaw)
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
    delayed=(int(state[w,0])-int(param[w,4])+21)%21
    for j in range(32):obs[w,j]=history[w,delayed,j]
    reason=int(0)
    if not wp.isfinite(qpos[w,0]) or not wp.isfinite(qpos[w,2]):reason=1
    elif wp.max(wp.abs(roll),wp.abs(pitch))>D(.6981317007977318) or qpos[w,2]<.02:reason=2
    elif nonwheel:reason=3
    elif diag[w,14]==D(1):reason=4
    elif state[w,1]>=D(0) and t-state[w,1]>=D(2)-D(1.e-10):reason=5
    elif state[w,1]<D(0) and t>=param[w,3]-D(1.e-10):reason=6
    if diag[w,12]<D(1)-D(1.e-10):state[w,16]=state[w,16]+D(1)
    state[w,18]=state[w,18]+diag[w,13]
    if reason:
        done[w]=reason;active[w]=0
        success=reason==5 and (touch&int(param[w,5]))==int(param[w,5]) and wp.max(state[w,8],wp.max(state[w,9],state[w,10]))<=D(.08726646259971647) and state[w,5]>D(0)
        if state[w,5]>D(0):success=success and wp.sqrt(state[w,4]/state[w,5])<=D(.2)*wp.abs(param[w,0])
        success=success and state[w,6]<=D(.6) and state[w,7]<=D(.03)
        state[w,19]=D(0)
        if success:state[w,19]=D(1);reward[w]=reward[w]+D(10);state[w,20]=state[w,20]+D(10)
        else:reward[w]=reward[w]-D(10);state[w,20]=state[w,20]-D(10)
        for j in range(qpos.shape[1]):stopped_q[w,j]=qpos[w,j]
        for j in range(qvel.shape[1]):stopped_v[w,j]=qvel[w,j];stopped_w[w,j]=warm[w,j]


@wp.kernel
def reset_rows(mask:wp.array[int],q0:wp.array[float],q:wp.array2d[float],v:wp.array2d[float],warm:wp.array2d[float],clock:wp.array[float],sensors:wp.array2d[float],
               control_state:wp.array2d[D],state:wp.array2d[D],residual:wp.array2d[D],active:wp.array[int],done:wp.array[int],
               obs0:wp.array2d[float],obs:wp.array2d[float],history:wp.array3d[float],targets:wp.array2d[float]):
    w=wp.tid()
    if not mask[w]:return
    for j in range(q.shape[1]):q[w,j]=q0[j]
    for j in range(v.shape[1]):v[w,j]=0.;warm[w,j]=0.
    clock[w]=0.;active[w]=1;done[w]=0
    for j in range(sensors.shape[1]):sensors[w,j]=0.
    for j in range(19):control_state[w,j]=D(0)
    control_state[w,1]=D(.3);control_state[w,12]=D(-1)
    for j in range(21):state[w,j]=D(0)
    state[w,1]=D(-1)
    for j in range(6):residual[w,j]=D(0)
    for j in range(3):targets[w,j]=0.
    for j in range(32):
        obs[w,j]=obs0[w,j]
        for t in range(21):history[w,t,j]=obs0[w,j]


class NativeEnv(VecEnv):
    def __init__(self,n=128,stage=3,seed=730000,scenario=None):
        if not isinstance(n,int) or not 1 <= n <= 1024:raise ValueError('用户限制：批量环境数须为1～1024')
        wp.init();wp.set_device('cuda:0')
        self.num_envs=n;self.stage=stage
        self.cpu,self.model,self.data,self.scenarios=bank(n,stage,seed,scenario)
        self.k=constants(self.cpu,n)
        ids=self.k['ids'].numpy().tolist()+[self.cpu.geom(x).id for x in ('wheel_collide_L','wheel_collide_R','bump_L','bump_R')]
        self.ids=wp.array(ids,dtype=wp.int32)
        p=[]
        for s in self.scenarios:
            goal=s.center+abs(s.offset)/2+.75
            p.append([s.speed,np.sign(s.speed),goal,1.5+1.5*goal/abs(s.speed),round(s.delay_ms*2),(1 if s.height_l else 0)|(2 if s.height_r else 0)])
        self.param=wp.array(p,dtype=D);self.command=wp.zeros(n,dtype=D)
        self.active=wp.ones(n,dtype=wp.int32);self.done=wp.zeros(n,dtype=wp.int32)
        self.state=wp.zeros((n,21),dtype=D);self.diag=wp.zeros((n,15),dtype=D)
        self.residual=wp.zeros((n,6),dtype=D);self.reward=wp.zeros(n,dtype=D);self.contact_flags=wp.zeros((n,2),dtype=wp.int32)
        self.obs=wp.zeros((n,32));self.history=wp.zeros((n,21,32))
        self.targets=wp.zeros((n,3));self.stopped_q=wp.zeros((n,self.cpu.nq));self.stopped_v=wp.zeros((n,self.cpu.nv));self.stopped_w=wp.zeros((n,self.cpu.nv))
        self.q0=wp.array(self.cpu.key_qpos[self.cpu.keyframe('stand').id],dtype=wp.float32)
        initial=np.zeros((n,32),dtype=np.float32)
        q=self.q0.numpy()
        initial[:,12:16]=q[np.array(ids[:4])]
        from state_estimation import leg_kinematics
        initial[:,22:24]=[np.linalg.norm(leg_kinematics(q[np.array(ids[:2])],np.zeros(2))[0]),np.linalg.norm(leg_kinematics(q[np.array(ids[2:4])],np.zeros(2))[0])]
        self.obs0=wp.array(initial,dtype=wp.float32)
        self.mask=wp.ones(n,dtype=wp.int32)
        self.reset_args=[self.mask,self.q0,self.data.qpos,self.data.qvel,self.data.qacc_warmstart,self.data.time,self.data.sensordata,self.k['state'],self.state,self.residual,self.active,self.done,self.obs0,self.obs,self.history,self.targets]
        wp.launch(reset_rows,n,self.reset_args)
        mjw.forward(self.model,self.data)
        with wp.ScopedCapture() as capture:
            wp.launch(begin,n,[self.reward])
            for _ in range(40):
                wp.launch(command_step,n,[self.state,self.param,self.command,self.active,self.data.qpos,self.data.qvel,self.data.qacc_warmstart,self.stopped_q,self.stopped_v,self.stopped_w,self.contact_flags])
                wp.launch(control,n,[self.data.qpos,self.data.qvel,self.data.sensordata,self.targets,self.command,self.active,
                    self.k['state'],self.ids,self.k['heights'],self.k['gains'],self.k['feed'],self.k['angles'],self.k['reference'],self.data.ctrl,self.diag],block_dim=32)
                mjw.step(self.model,self.data)
                wp.launch(reduce_contacts,self.data.naconmax,[self.data.nacon,self.data.contact.worldid,self.data.contact.geom,self.ids,self.contact_flags])
                wp.launch(after,n,[self.data.qpos,self.data.qvel,self.data.sensordata,self.data.qacc_warmstart,self.data.time,self.contact_flags,self.ids,self.param,self.command,self.state,self.k['state'],self.diag,
                    self.residual,self.active,self.done,self.reward,self.obs,self.history,self.stopped_q,self.stopped_v,self.stopped_w],block_dim=32)
        self.graph=capture.graph
        super().__init__(n,gym.spaces.Box(-np.inf,np.inf,(32,),dtype=np.float32),gym.spaces.Box(-1.,1.,(3,),dtype=np.float32))

    def reset(self):
        self.mask.fill_(1);wp.launch(reset_rows,self.num_envs,self.reset_args);mjw.forward(self.model,self.data)
        return self.obs.numpy().copy()

    def step_async(self,actions):
        a=np.asarray(actions,dtype=np.float32)
        if a.shape!=(self.num_envs,3) or not np.isfinite(a).all() or np.any(abs(a)>1.000001):raise ValueError('无效动作')
        self.targets.assign(a);wp.capture_launch(self.graph)

    def step_wait(self):
        obs=self.obs.numpy().copy();reward=self.reward.numpy().copy();reasons=self.done.numpy();done=reasons!=0
        if np.any(self.data.overflow.numpy()):raise RuntimeError('GPU容量溢出')
        if not np.isfinite(obs).all() or not np.isfinite(reward).all():raise FloatingPointError('原生GPU非有限状态')
        infos=[{} for _ in range(self.num_envs)]
        if done.any():
            states=self.state.numpy()
            for i in np.flatnonzero(done):
                infos[i]=dict(terminal_observation=obs[i].copy(),reason=['ongoing','invalid','fall','nonwheel_contact','invalid_mapping','completed','timeout'][int(reasons[i])],physical_steps=int(states[i,0]),arrival_s=float(states[i,1]) if states[i,1]>=0 else None,
                    rms_deg=(np.sqrt(states[i,11:14]/max(states[i,0]*.0005,.0005))*180/np.pi).tolist(),
                    velocity_rmse=float(np.sqrt(states[i,4]/states[i,5])) if states[i,5]>0 else None,stop_distance_m=float(states[i,6]),tail_speed_m_s=float(states[i,7]),success=bool(states[i,19]),
                    duration_s=states[i,0]*.0005,episode=dict(r=float(states[i,20]),l=int(np.ceil(states[i,0]/40))),peak_deg=(states[i,8:11]*180/np.pi).tolist(),TimeLimit_truncated=int(reasons[i])==6)
                infos[i]['TimeLimit.truncated']=int(reasons[i])==6
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
