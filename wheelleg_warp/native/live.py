"""1024世界真实姿态总览＋可选世界物理子步详情；训练世界0完整轨迹归档。"""
from dataclasses import asdict
from pathlib import Path
import json,struct,time
import numpy as np
import warp as wp
import mujoco_warp as mjw
from native.environment import NativeEnv,begin,command_step,reduce_contacts,after
from native.controller import control,D
from dashboard.live_env import atomic_json
SELECTED=Path(__file__).resolve().parents[1]/'dashboard/local_data/selected_environment.json'


@wp.kernel
def snapshot(q:wp.array2d[float],v:wp.array2d[float],u:wp.array2d[float],state:wp.array2d[D],obs:wp.array2d[float],out:wp.array2d[D],selected:wp.array[int],slot:int):
    which,j=wp.tid();w=int(0)
    if which==1:w=selected[0]
    row=which*8+slot;nq=q.shape[1];nv=v.shape[1];nu=u.shape[1]
    if j==0:out[row,j]=state[w,0]*D(.0005)
    elif j<=nq:out[row,j]=D(q[w,j-1])
    elif j<=nq+nv:out[row,j]=D(v[w,j-1-nq])
    elif j<=nq+nv+nu:out[row,j]=D(u[w,j-1-nq-nv])
    else:out[row,j]=D(obs[w,j-1-nq-nv-nu])


@wp.kernel
def overview(x:wp.array2d[wp.vec3],rotation:wp.array2d[wp.mat33],state:wp.array2d[D],done:wp.array[int],out:wp.array2d[float],root:int,wheels:wp.array[int],columns:wp.array[int]):
    w=wp.tid();out[w,0]=float(wp.max(D(0),state[w,0]*D(.0005)-D(.0005)))
    out[w,1]=float(done[w]);out[w,2]=float(state[w,19])
    for j in range(9):out[w,3+j]=rotation[w,root][j//3,j%3]
    for j in range(x.shape[1]):
        for k in range(3):out[w,12+j*3+k]=x[w,j][k]
    for j in range(wheels.shape[0]):
        for k in range(3):out[w,12+x.shape[1]*3+j*3+k]=rotation[w,wheels[j]][k,columns[j]]


class LiveNativeEnv(NativeEnv):
    def __init__(self,directory,start_steps=0,phase='formal_training',**kwargs):
        super().__init__(**kwargs)
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.start_steps=start_steps;self.policy_frames=0;self.phase=phase;self.selected=0
        self.episode_counts=np.zeros(self.num_envs,dtype=int);self.returns=np.zeros(self.num_envs)
        self.select_array=wp.zeros(1,dtype=wp.int32);self.last_select=0.;self.last_overview=0.
        self.width=1+self.cpu.nq+self.cpu.nv+self.cpu.nu+32
        self.frames=wp.zeros((16,self.width),dtype=D)
        self.wheel_bodies=[i for i in range(self.cpu.nbody) if 'wheel' in self.cpu.body(i).name.lower()]
        columns=[]
        for i in self.wheel_bodies:
            joint=int(self.cpu.body_jntadr[i]);columns.append(1 if joint>=0 and abs(self.cpu.jnt_axis[joint,0])>.9 else 0)
        self.wheel_ids=wp.array(self.wheel_bodies,dtype=wp.int32);self.wheel_columns=wp.array(columns,dtype=wp.int32)
        self.overview_array=wp.zeros((self.num_envs,12+self.cpu.nbody*3+3*len(columns)),dtype=wp.float32)
        self.root_body=int(self.cpu.jnt_bodyid[0]);n=self.num_envs
        with wp.ScopedCapture() as capture:
            wp.launch(begin,n,[self.reward])
            for i in range(40):
                wp.launch(command_step,n,[self.state,self.param,self.command,self.active,self.data.qpos,self.data.qvel,self.data.qacc_warmstart,self.stopped_q,self.stopped_v,self.stopped_w,self.contact_flags])
                wp.launch(control,n,[self.data.qpos,self.data.qvel,self.data.sensordata,self.targets,self.command,self.active,self.k['state'],self.ids,self.k['heights'],self.k['gains'],self.k['feed'],self.k['angles'],self.k['reference'],self.data.ctrl,self.diag],block_dim=32)
                mjw.step(self.model,self.data)
                wp.launch(reduce_contacts,self.data.naconmax,[self.data.nacon,self.data.contact.worldid,self.data.contact.geom,self.ids,self.contact_flags])
                wp.launch(after,n,[self.data.qpos,self.data.qvel,self.data.sensordata,self.data.qacc_warmstart,self.data.time,self.contact_flags,self.ids,self.param,self.command,self.state,self.k['state'],self.diag,self.residual,self.active,self.done,self.reward,self.obs,self.history,self.stopped_q,self.stopped_v,self.stopped_w],block_dim=32)
                if i%5==4:wp.launch(snapshot,(2,self.width),[self.data.qpos,self.data.qvel,self.data.ctrl,self.state,self.obs,self.frames,self.select_array,i//5])
        self.graph=capture.graph

    def metadata(self,index):
        return dict(kind='actual_training_physics_frames',backend='native',environment_index=index,environments=self.num_envs,
            episode=int(self.episode_counts[index]),phase=self.phase,scenario=asdict(self.scenarios[index]),source_run=str(self.directory.parent),
            recorded_policy_hz=50,recorded_physics_hz=400,started=time.time(),status='recording')

    def new_episode(self):
        self.rows=[];self.actions=[];self.rewards=[];self.last_time=0.
        self.folder=self.directory/'episodes'/f'{self.episode_counts[0]:06d}';self.folder.mkdir(parents=True,exist_ok=False)
        self.episode_metadata=self.metadata(0);atomic_json(self.folder/'metadata.json',self.episode_metadata)

    def reset(self):
        result=super().reset();self.episode_counts+=1;self.returns[:]=0;self.new_episode();return result

    def step_async(self,actions):
        if time.monotonic()-self.last_select>.2:
            self.last_select=time.monotonic()
            try:
                index=json.loads(SELECTED.read_text())['environment']
                if type(index) is int and 0<=index<self.num_envs and index!=self.selected:
                    self.selected=index;self.select_array.assign([index])
            except (FileNotFoundError,KeyError,ValueError):pass
        super().step_async(actions);self.recorded_action=np.asarray(actions[0]).copy()

    def split(self,frames):
        a=1+self.cpu.nq;b=a+self.cpu.nv;c=b+self.cpu.nu
        return dict(time=frames[:,0],qpos=frames[:,1:a],qvel=frames[:,a:b],ctrl=frames[:,b:c],observation=frames[:,c:])

    def save_overview(self):
        wp.launch(overview,self.num_envs,[self.data.xpos,self.data.xmat,self.state,self.done,self.overview_array,self.root_body,self.wheel_ids,self.wheel_columns])
        values=self.overview_array.numpy()
        meta=dict(environments=self.num_envs,bodies=self.cpu.nbody,width=values.shape[1],root=self.root_body,
            parents=self.cpu.body_parentid.tolist(),wheels=self.wheel_bodies,
            sequence=self.policy_frames,wall_time=time.time(),sample_steps=self.start_steps+self.policy_frames*self.num_envs,
            source_run=str(self.directory.parent),kinematics_lag_seconds=.0005)
        header=json.dumps(meta,separators=(',',':')).encode();header+=b' '*((-len(header))%4)
        target=self.directory/'overview.bin';temporary=target.with_suffix('.tmp')
        temporary.write_bytes(struct.pack('<I',len(header))+header+values.astype('<f4',copy=False).tobytes());temporary.replace(target)

    def step_wait(self):
        packed=self.frames.numpy().copy();self.policy_frames+=1
        if time.monotonic()-self.last_overview>=.1:
            self.save_overview();self.last_overview=time.monotonic()
        result=super().step_wait();obs,rewards,done,infos=result;self.returns+=rewards
        frames=packed[:8];mask=np.r_[True,np.diff(frames[:,0])>1e-10] & (frames[:,0]>self.last_time+1e-10);frames=frames[mask]
        if not len(frames) and done[0]:frames=packed[7:8]
        if len(frames):
            self.last_time=float(frames[-1,0]);self.rows.extend(frames.copy());self.actions.extend([self.recorded_action.copy() for _ in frames])
            self.rewards.extend([0.]*(len(frames)-1)+[float(rewards[0])])
        preview=packed[8:];preview=preview[np.r_[True,np.diff(preview[:,0])>1e-10]]
        metadata=self.metadata(self.selected);seq=self.policy_frames
        with (self.directory/'latest.tmp').open('wb') as f:np.savez(f,**self.split(preview),sequence=seq,episode=metadata['episode'],environment=self.selected)
        (self.directory/'latest.tmp').replace(self.directory/'latest.npz')
        atomic_json(self.directory/'latest.json',dict(**metadata,wall_time=time.time(),sequence=seq,frame=seq*8,
            simulation_seconds=float(preview[-1,0]),sample_steps=self.start_steps+seq*self.num_envs,
            cumulative_reward=float(self.returns[self.selected]),frames_in_chunk=len(preview)))
        if done[0]:
            metrics={k:v for k,v in infos[0].items() if k!='terminal_observation'}
            with (self.folder/'trajectory.tmp').open('wb') as f:
                np.savez_compressed(f,**self.split(np.asarray(self.rows)),action=np.asarray(self.actions),reward=np.asarray(self.rewards))
            (self.folder/'trajectory.tmp').replace(self.folder/'trajectory.npz')
            atomic_json(self.folder/'metadata.json',{**self.episode_metadata,'status':'completed','frames':len(self.rows),'finished':time.time(),'metrics':metrics})
        self.episode_counts[done]+=1;self.returns[done]=0
        if done[0]:self.new_episode()
        return result
