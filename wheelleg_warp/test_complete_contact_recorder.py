"""Runnable full-wrench, state, dense-stream and ownership admission; no mj_step."""
import json

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.types import vec5

import complete_contact_recorder as rec
from task_mode_recorder import pre as header_pre, post as header_post
from train_height_comparison import raw_env


def force_check():
    checked = []
    for cone in ('elliptic', 'pyramidal'):
        for dim in (3, 4, 6):
            xml = f'<mujoco><option cone="{cone}"/><worldbody><geom type="plane" size="1 1 .1" condim="{dim}" friction=".8 .01 .005"/>'
            for x in (-.1, .1):
                xml += f'<body pos="{x} 0 .045"><freejoint/><geom type="sphere" size=".05" mass="1" condim="{dim}" friction=".8 .01 .005"/></body>'
            m = mujoco.MjModel.from_xml_string(xml + '</worldbody></mujoco>')
            d = mujoco.MjData(m)
            d.qvel[:] = np.tile([1, .4, 0, .2, .3, 3], 2)
            mujoco.mj_forward(m, d)
            assert d.ncon == 2
            expected = np.zeros((2, 6))
            for j in range(2):
                mujoco.mj_contactForce(m, d, j, expected[j])
                assert expected[j, 0] > 0 and np.linalg.norm(expected[j, 1:3]) > 0
                world_force = expected[j, :3] @ d.contact[j].frame.reshape(3, 3)
                np.testing.assert_allclose(world_force, d.qfrc_constraint[6*j:6*j+3], atol=1e-8)
            # Force decoding also needs nonzero torsional/rolling components even
            # when this particular pyramidal physical fixture has zero torque.
            for synthetic in (False, True):
                if synthetic:
                    d.efc_force[:] = np.arange(1, d.nefc+1) * .25
                    for j in range(2):
                        mujoco.mj_contactForce(m, d, j, expected[j])
                for swap in (False, True):
                    pair = np.asarray(d.contact.geom, np.int32).copy()
                    frame = np.asarray(d.contact.frame, np.float32).reshape(2,3,3).copy()
                    efc = np.asarray(d.efc_force, np.float32).copy()
                    want = expected.copy()
                    if swap:
                        pair = pair[:, ::-1].copy()
                        frame[:, :2] *= -1
                        want[:, [2, 5]] *= -1
                        for c in d.contact:
                            for k in (2, 5):
                                if k >= c.dim:
                                    continue
                                if cone == 'elliptic':
                                    efc[c.efc_address+k] *= -1
                                else:
                                    adr = c.efc_address+2*(k-1)
                                    efc[adr:adr+2] = efc[adr:adr+2][::-1]
                    args = [wp.array(np.array([2],np.int32)),wp.array(np.zeros(2,np.int32)),
                        wp.array(pair,dtype=wp.vec2i),wp.array(np.asarray(d.contact.dist,np.float32)),
                        wp.array(np.asarray(d.contact.pos,np.float32),dtype=wp.vec3),wp.array(frame,dtype=wp.mat33),
                        wp.array(np.asarray(d.contact.friction,np.float32),dtype=vec5),wp.array(np.asarray(d.contact.dim,np.int32)),
                        wp.array(np.array([np.arange(c.efc_address,c.efc_address+6) for c in d.contact],np.int32)),
                        wp.array(efc.reshape(1,-1)),d.nefc,int(m.opt.cone)]
                    trace = wp.zeros((1,1,rec.old.WIDTH),dtype=rec.D)
                    t = trace.numpy();t[0,0,0]=1;trace.assign(t)
                    out = wp.zeros((1,3,len(rec.CONTACT_COL)),dtype=rec.D)
                    saved = [x.numpy().copy() for x in args if isinstance(x,wp.array)]
                    wp.launch(rec.contact_copy,3,[0,*args,trace,out])
                    got = out.numpy()[0]
                    assert np.array_equal(got[:,0],[1,1,0])
                    np.testing.assert_array_equal(got[:2,2:4],pair)
                    np.testing.assert_allclose(got[:2,23:29],want,rtol=2e-6,atol=2e-5)
                    for j in range(2):
                        world = got[j,23:26] @ got[j,9:18].reshape(3,3)
                        original = expected[j,:3] @ d.contact[j].frame.reshape(3,3)
                        np.testing.assert_allclose(world,-original if swap else original,rtol=2e-6,atol=2e-5)
                    for x,y in zip([x for x in args if isinstance(x,wp.array)],saved):
                        np.testing.assert_array_equal(x.numpy(),y)
                    # Zero-load contacts must still be present, and inactive worlds absent.
                    args[9].zero_();wp.launch(rec.contact_copy,3,[0,*args,trace,out])
                    assert np.array_equal(out.numpy()[0,:,0],[1,1,0]) and not out.numpy()[0,:2,23:29].any()
                    trace.zero_();wp.launch(rec.contact_copy,3,[0,*args,trace,out]);assert not out.numpy()[:,:,0].any()
            checked.append(dict(cone=cone,condim=dim,physical_and_synthetic=True,both_orders=True,contacts=2))
    return checked


def stream_fixture(raw):
    trace,mask,count,_,pre,post,meta,contacts = raw._complete_buffers
    n = raw.num_envs
    for step in (1,2):
        t=np.zeros(trace.shape);t[0,0,0]=1;t[0,0,1]=step;t[0,0,2]=(step-1)*.0005;t[0,0,3]=step*.0005
        trace.assign(t);counts=np.zeros((n,2));counts[0]=[step,step];count.assign(counts)
        if step==2:
            state=raw.state.numpy();state[0,0]=2;raw.state.assign(state)
            done=raw.done.numpy();done[0]=5;raw.done.assign(done)
        result=raw.step_wait()
    assert result[3][0]['complete_counts']==dict(steps=2,contacts=0) and mask.numpy()[0]==0
    done=raw.done.numpy();done[0]=5;raw.done.assign(done)
    assert 'complete_counts' not in raw.step_wait()[3][0]
    raw.reset()
    assert np.all(mask.numpy()==1)
    for a in (trace,count,pre,post,meta,contacts):
        assert not a.numpy().any()


def check():
    p=json.loads((rec.OUT/'proposal.json').read_text())
    owners=[]
    for mode in ('diff3','virtual6'):
        raw=rec.instrument(raw_env,p['cases'],mode)
        try:
            raw.reset()
            trace,mask,count,window,pre,post,meta,contacts=raw._complete_buffers
            args=[0,raw.data.qpos,raw.data.qvel,raw.ids,raw.wheel_offsets,raw.active,mask,raw.state,
                  raw.k['state'],raw.k['reference'],raw.param,raw.targets,raw.diag,raw.data.ctrl,raw.command,
                  raw.control_extra[0],raw.nominal_correction,count,trace]
            inputs=[a for a in args[1:-2] if isinstance(a,wp.array)]
            saved=[a.numpy().copy() for a in inputs]
            wp.launch(header_pre,raw.num_envs,args)
            wp.launch(rec.state_copy,raw.num_envs,[0,raw.data.qpos,raw.data.qvel,raw.data.ctrl,trace,pre])
            wp.launch(header_post,raw.num_envs,[0,raw.data.qpos,raw.data.qvel,raw.ids,raw.wheel_offsets,
                       raw.physical_args[8],raw.data.actuator_force,window,raw.param,count,trace])
            wp.launch(rec.state_copy,raw.num_envs,[0,raw.data.qpos,raw.data.qvel,raw.data.actuator_force,trace,post])
            got=trace.numpy()[0];assert np.all(got[:,0]==1) and np.all(count.numpy()[:,1]==1)
            expected=np.c_[raw.data.qpos.numpy(),raw.data.qvel.numpy(),raw.data.ctrl.numpy()]
            np.testing.assert_array_equal(pre.numpy()[0],expected)
            np.testing.assert_array_equal(post.numpy()[0],np.c_[expected[:,:raw.cpu.nq+raw.cpu.nv],raw.data.actuator_force.numpy()])
            q=raw.data.qpos.numpy();v=raw.data.qvel.numpy()
            for w in range(raw.num_envs):
                data=mujoco.MjData(raw.cpu);data.qpos[:]=q[w];data.qvel[:]=v[w];mujoco.mj_forward(raw.cpu,data)
                for side,name in enumerate(('wheelL','wheelR')):
                    pos=data.xpos[raw.cpu.body(name).id]
                    np.testing.assert_allclose(got[w,10+3*side:13+3*side],pos,rtol=0,atol=1e-6)
                    np.testing.assert_allclose(got[w,16+3*side:19+3*side],pos,rtol=0,atol=1e-6)
            for a,b in zip(inputs,saved):np.testing.assert_array_equal(a.numpy(),b)
            # No position window: even a far-lateral root records every first-episode step.
            q[:,0]=q[:,1]=100.;clone=wp.array(q)
            wp.launch(header_post,raw.num_envs,[0,clone,raw.data.qvel,raw.ids,raw.wheel_offsets,
                       raw.physical_args[8],raw.data.actuator_force,window,raw.param,count,trace])
            assert np.all(trace.numpy()[0,:,0]==1)
            raw.reset();stream_fixture(raw)
            assert not raw.state.numpy()[:,0].any() and not raw.data.time.numpy().any()
            owners.append(dict(mode=mode,worlds=raw.num_envs,**raw._complete_topology))
        finally:
            raw.close()
    force=force_check()
    # Complete-contact cardinality is checked even when every solved force is zero.
    t=np.zeros((2,rec.old.WIDTH));t[:,0]=1;t[:,1]=[1,2];t[:,2]=[0,.0005];t[:,3]=[.0005,.001]
    meta=np.zeros((2,6));meta[:,1]=[2,0]
    data=(t,np.zeros((2,3)),np.zeros((2,3)),meta,np.zeros((2,29)),np.array([1,1]))
    rec.validate(data)
    for bad in ('missing_contact','overflow','step_gap'):
        values=tuple(a.copy() for a in data)
        if bad=='missing_contact':values[3][0,1]=3
        if bad=='overflow':values[3][0,2]=1
        if bad=='step_gap':values[0][1,1]=3
        try:rec.validate(values)
        except AssertionError:pass
        else:raise AssertionError('Accepted '+bad)
    rec.atomic_json(rec.OUT/'unit.json',dict(verified=True,owners=owners,force_cases=force,
        full_CPU_force_torque_and_world_signs=True,complete_zero_load_and_inactive_contacts=True,
        every_world_CPU_forward_centres=True,dense_far_lateral_steps=True,
        synthetic_terminal_freeze_reset=True,missing_contact_overflow_step_gap_rejected=True,
        physics_rollouts=0,training_updates=0,limits='Constructor/capture, CPU forward and synthetic states only; does not prove live complete rollout recording or physical validity of a new method.'))
    print('PASS dense source/pose/owner/reset,12 physical+synthetic force/torque/order fixtures; 0 rollout',flush=True)


if __name__=='__main__':
    check()
