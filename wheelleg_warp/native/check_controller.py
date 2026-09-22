import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[2]/'wheelleg_ppo/tools')]
import json
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
import wheelleg_sim as sim
from ppo_env import build_model,Scenario,Residual
from native.controller import control,constants,D

wp.init();wp.set_device('cuda:0')
m=build_model(Scenario(height_l=.012,mu_l=.7,mu_r=.9));ref=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,ref,m.keyframe('stand').id);mujoco.mj_forward(m,ref)
gm=mjw.put_model(m);gd=mjw.put_data(m,ref,nconmax=64,njmax=256)
k=constants(m,1);target=wp.array([[.1,-.1,.1]],dtype=wp.float32);cmd=wp.zeros(1,dtype=D);active=wp.ones(1,dtype=wp.int32);diag=wp.zeros((1,15),dtype=D)
with wp.ScopedCapture() as captured:
    mjw.step(gm,gd)
st=sim.make_state(m,True,True);res=Residual();res.set_action([.1,-.1,.1]);peak=0.
for i in range(12000):
    mjw.get_data_into(ref,m,gd);ref.contact.geom1[:]=ref.contact.geom[:,0];ref.contact.geom2[:]=ref.contact.geom[:,1]
    velocity=min(1.,max(0.,(i*.0005-1))) if i*.0005<4.5 else 0.
    st.cmd_vel=velocity;sim.control(m,ref,st,res);cmd.assign([velocity])
    wp.launch(control,1,[gd.qpos,gd.qvel,gd.sensordata,target,cmd,active,k['state'],k['ids'],k['heights'],k['gains'],k['feed'],k['angles'],k['reference'],k['yaw'],gd.ctrl,diag])
    error=float(np.max(abs(ref.ctrl-gd.ctrl.numpy()[0])));peak=max(peak,error)
    if error>1e-4:
        print('MISMATCH',i,error,ref.ctrl,gd.ctrl.numpy()[0],flush=True);raise SystemExit(1)
    wp.capture_launch(captured.graph)
    if i%1000==0:print('progress',i,'peak',peak,flush=True)
print('PASS',peak,flush=True)
