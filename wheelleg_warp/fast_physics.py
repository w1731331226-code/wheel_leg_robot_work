"""单环境闭环专用：只回读原控制/奖励实际使用字段，合并为一次CUDA传输。"""
from dataclasses import asdict,is_dataclass
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from gpu_env import WarpEnv
from mujoco_warp._src.types import vec5


@wp.kernel
def pack(qpos:wp.array2d[float],qvel:wp.array2d[float],sensors:wp.array2d[float],
         ctrl:wp.array2d[float],force:wp.array2d[float],com:wp.array2d[wp.vec3],
         ncon:wp.array[int],overflow:wp.array[int],clock:wp.array[float],
         geom:wp.array[wp.vec2i],friction:wp.array[vec5],
         out:wp.array[float],nq:int,nv:int,ns:int,nu:int,nb:int):
    i=wp.tid()
    a=nq;b=a+nv;c=b+ns;e=c+nu;f=e+nu;g=f+nb*3
    if i<a:out[i]=qpos[0,i]
    elif i<b:out[i]=qvel[0,i-a]
    elif i<c:out[i]=sensors[0,i-b]
    elif i<e:out[i]=ctrl[0,i-c]
    elif i<f:out[i]=force[0,i-e]
    elif i<g:out[i]=com[0,(i-f)//3][(i-f)%3]
    elif i==g:out[i]=float(ncon[0])
    elif i==g+1:out[i]=float(overflow[0])
    elif i==g+2:out[i]=clock[0]
    else:
        j=(i-g-3)//7;k=(i-g-3)%7
        if j<ncon[0]:
            if k<2:out[i]=float(geom[j][k])
            else:out[i]=friction[j][k-2]
        else:out[i]=0.


class PackedPhysics:
    """GPU物理不改；主机MjData只供已审计的WheelLegEnv/控制器读取。"""
    def __init__(self,model,data):
        self.model=mjw.put_model(model)
        self.data=mjw.put_data(model,data,nworld=1,nconmax=64,njmax=256)
        sizes=(model.nq,model.nv,model.nsensordata,model.nu,model.nu,model.nbody*3)
        self.edges=np.r_[0,np.cumsum(sizes)]
        self.header=int(self.edges[-1]);size=self.header+3+64*7
        self.output=wp.empty(size,dtype=wp.float32)
        self.host=wp.empty(size,dtype=wp.float32,device='cpu',pinned=True)
        self.host_view=self.host.numpy()
        self.control=wp.empty((1,model.nu),dtype=wp.float32,device='cpu',pinned=True)
        self.control_view=self.control.numpy()
        self.stream=wp.get_stream()
        with wp.ScopedCapture() as capture:
            wp.copy(self.data.ctrl,self.control)
            mjw.step(self.model,self.data)
            d=self.data
            wp.launch(pack,size,[d.qpos,d.qvel,d.sensordata,d.ctrl,d.actuator_force,d.subtree_com,
                d.nacon,d.overflow,d.time,d.contact.geom,d.contact.friction,
                self.output,model.nq,model.nv,model.nsensordata,model.nu,model.nbody])
            wp.copy(self.host,self.output)
        self.graph=capture.graph

    def step(self,model,data):
        self.control_view[0]=data.ctrl
        wp.capture_launch(self.graph)
        wp.synchronize_stream(self.stream)
        out=self.host_view;head=self.header
        count=int(out[head])
        if out[head+1] or not 0<=count<=64:raise RuntimeError('GPU接触/约束容量溢出')
        fields=(data.qpos,data.qvel,data.sensordata,data.ctrl,data.actuator_force,data.subtree_com.reshape(-1))
        for field,start,end in zip(fields,self.edges[:-1],self.edges[1:]):field[:]=out[start:end]
        data.time=float(out[head+2])
        if data.ncon!=count:
            mujoco._functions._realloc_con_efc(data,ncon=count,nefc=0,nJ=0)
            data.ncon=count
        contacts=out[head+3:].reshape(64,7)[:count]
        data.contact.geom[:]=contacts[:,:2]
        data.contact.geom1[:]=contacts[:,0];data.contact.geom2[:]=contacts[:,1]
        data.contact.friction[:]=contacts[:,2:]


class FastWarpEnv(WarpEnv):
    def reset(self,**kwargs):
        if is_dataclass(self.env.fixed_scenario):
            self.env.fixed_scenario=self.source.Scenario(**asdict(self.env.fixed_scenario))
        if (kwargs.get('options') or {}).get('scenario') is not None:
            kwargs['options']=dict(scenario=self.source.Scenario(**asdict(kwargs['options']['scenario'])))
        self.physics=None
        result=self.env.reset(**kwargs)
        self.physics=PackedPhysics(self.env.model,self.env.data)
        return result
