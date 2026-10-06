"""Supplemental current-cone API and in-window/bypass sampling check, no rollout."""
import json
import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.types import vec5
from task_mode_recorder import OUT, WIDTH, D, contact_record, post, geometry
from train_height_comparison import raw_env
from dashboard.live_env import atomic_json


def check():
    cones=[]
    for cone,condim in ((c,d) for c in ('elliptic','pyramidal') for d in (3,4,6)):
        xml=f'<mujoco><option cone="{cone}"/><worldbody><geom type="plane" size="1 1 .1" condim="{condim}"/><body pos="0 0 .045"><freejoint/><geom type="sphere" size=".05" mass="1" condim="{condim}"/></body></worldbody></mujoco>'
        m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m); mujoco.mj_forward(m,d); assert d.ncon==1
        c=d.contact[0]; expected=np.zeros(6); mujoco.mj_contactForce(m,d,0,expected); assert expected[0]>0 and c.dim==condim
        ids=np.zeros(19,np.int32); ids[11:17]=[1,2,0,-1,3,2]
        for swap in (False,True):
            pair=np.array([[0,1]],np.int32); frame=np.asarray(c.frame,np.float32).reshape(1,3,3).copy()
            if swap: pair[:]=[1,0]; frame[:,0]*=-1; frame[:,1]*=-1
            a=[wp.array(np.array([1],np.int32)),wp.array(np.array([0],np.int32)),wp.array(pair,dtype=wp.vec2i),wp.array(np.array([c.dist],np.float32)),wp.array(np.asarray(c.pos,np.float32).reshape(1,3),dtype=wp.vec3),wp.array(frame,dtype=wp.mat33),wp.array(np.asarray(c.friction,np.float32).reshape(1,5),dtype=vec5),wp.array(np.array([c.dim],np.int32)),wp.array(np.arange(c.efc_address,c.efc_address+6,dtype=np.int32).reshape(1,6)),wp.array(np.asarray(d.efc_force,np.float32).reshape(1,-1))]
            b=[wp.array(ids),wp.array(np.asarray(m.geom_bodyid,np.int32)),wp.array(np.asarray(m.body_rootid,np.int32)),wp.array(np.asarray(d.cvel,np.float32).reshape(1,m.nbody,6),dtype=wp.spatial_vector),wp.array(np.asarray(d.subtree_com,np.float32).reshape(1,m.nbody,3),dtype=wp.vec3)]
            values=np.zeros((1,1,WIDTH)); values[0,0,0]=1; out=wp.array(values,dtype=D)
            before=[x.numpy().copy() for x in a+b]; wp.launch(contact_record,1,[0,*a,d.nefc,int(m.opt.cone),*b,out])
            result=out.numpy()[0,0]; np.testing.assert_allclose(result[[62,64]],expected[0],rtol=1e-6,atol=1e-5)
            for x,y in zip(a+b,before): np.testing.assert_array_equal(x.numpy(),y)
        cones.append(dict(cone=cone,condim=condim))
    p=json.loads((OUT/'proposal.json').read_text()); raw=raw_env(p['cases'][:1],'diff3')
    try:
        raw.reset(); windows,_=geometry(raw); q=raw.data.qpos.numpy().copy(); q[0,1]=100.
        clone=wp.array(q); trace=wp.zeros((1,1,WIDTH),dtype=D); count=wp.zeros((1,2),dtype=D)
        for broad,step,expected in [(False,1,0),(False,40,1),(True,1,1)]:
            values=np.zeros((1,1,WIDTH)); values[0,0,0]=1; values[0,0,1]=step; trace.assign(values)
            window=wp.array(np.array([[-100.,100.]]) if broad else windows,dtype=D)
            a=[clone,raw.data.qvel,raw.ids,raw.wheel_offsets,raw.physical_args[8],raw.data.actuator_force,window,raw.param]
            before=[x.numpy().copy() for x in a]; wp.launch(post,1,[0,*a,count,trace])
            assert trace.numpy()[0,0,0]==expected
            for x,y in zip(a,before): np.testing.assert_array_equal(x.numpy(),y)
        assert not raw.state.numpy()[:,0].any() and not raw.data.time.numpy().any()
        actual_cone=int(raw.model.opt.cone); actual_dims=raw.cpu.geom_condim[raw.ids.numpy()[11:17]].tolist()
    finally: raw.close()
    atomic_json(OUT/'supplemental_unit.json',dict(verified=True,force_cases=cones,actual_native_cone=actual_cone,actual_geom_condims=actual_dims,both_geom_orders_CPU_force_agree=True,input_identity=True,in_window_2kHz_even_far_lateral_and_zero_torque=True,outside_50Hz=True,physical_rollouts=0,training_updates=0))
    print('PASS both cone APIs/geom orders and sparse/bypass sampling;0 rollout',flush=True)


if __name__=='__main__': check()
