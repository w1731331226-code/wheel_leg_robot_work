"""Readonly physical witnesses on the current Native pipeline, never policy input."""
import gc
import inspect
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

import route_pilot_env as evaluator
from native.controller import D, V6, command_bounds, control_physical_nominal
from native.environment import begin, command_step, reduce_contacts, collect_physical, after, wheel_center
from native.shared_reference import update as update_shared_reference
from trace_failure_chain import point_velocity
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/task_mode_witness_v1'
COL = ['valid','step','pre_s','post_s','body_x','body_y','body_z','roll','pitch','yaw']
COL += [f'{stem}_{a}' for stem in ('pre_left','pre_right','post_left','post_right') for a in 'xyz']
COL += ['left_leg_m','right_leg_m','design_margin','height_command','height_reference','roll_offset_requested',
        'roll_offset_selected','left_reference','right_reference','raw_nom_left','raw_nom_right',
        'nom_left','nom_right','accepted_left','accepted_right','command_left','command_right',
        'actual_left','actual_right','post_wheel_speed_left','post_wheel_speed_right','speed_command',
        'post_body_vx','post_body_vy','post_body_vz']
COL += [f'filtered_{s}' for s in ('F_left','F_right','H_left','H_right','wheel_left','wheel_right')]
COL += ['original_lambda','nom_correction_left','nom_correction_right','pre_wheel_speed_left',
        'pre_wheel_speed_right','pre_command_bound_left','pre_command_bound_right']
CONTACT = ['candidates','target_candidates','normal_N','target_normal_N','vertical_normal_N',
           'target_vertical_normal_N','max_target_normal_N','target_geom','point_x','point_y','point_z',
           'normal_x','normal_y','normal_z','mu','dim','distance','slip',
           'max_target_vertical_normal_N','vertical_target_geom','vertical_point_x','vertical_point_y',
           'vertical_point_z','vertical_normal_x','vertical_normal_y','vertical_normal_z']
assert len(COL) == 60 and len(CONTACT) == 26
COL += [side+'_'+name for side in ('left','right') for name in CONTACT]
WIDTH = len(COL)


@wp.kernel
def pre(slot:int, q:wp.array2d[float], v:wp.array2d[float], ids:wp.array[int],
        offsets:wp.array3d[wp.vec3d], active:wp.array[int], tracking:wp.array[int], state:wp.array2d[D],
        memory:wp.array2d[D], reference:wp.array2d[D], param:wp.array2d[D], targets:wp.array2d[float],
        diag:wp.array2d[D], ctrl:wp.array2d[float], command:wp.array[D], gains:wp.array2d[D],
        correction:wp.array2d[D], count:wp.array2d[D], trace:wp.array3d[D]):
    w=wp.tid()
    for j in range(WIDTH): trace[slot,w,j]=D(0)
    for side in range(2):
        trace[slot,w,60+26*side+7]=D(-1); trace[slot,w,60+26*side+19]=D(-1)
    if active[w]==0 or tracking[w]==0: return
    trace[slot,w,0]=D(1); trace[slot,w,1]=state[w,0]+D(1)
    trace[slot,w,2]=state[w,0]*D(.0005); count[w,0]+=D(1)
    for side in range(2):
        p=wheel_center(q,ids,offsets,w,side)
        for j in range(3): trace[slot,w,10+3*side+j]=p[j]
    qw=D(q[w,3]); qx=D(q[w,4]); qy=D(q[w,5]); qz=D(q[w,6])
    roll=wp.atan2(D(2)*(qw*qx+qy*qz),D(1)-D(2)*(qx*qx+qy*qy))
    trace[slot,w,25]=param[w,13]; trace[slot,w,26]=reference[w,2]
    trace[slot,w,27]=wp.clamp(D(.3)*roll+D(.12)*memory[w,5],D(-.035),D(.035))
    trace[slot,w,28]=(diag[w,21]-diag[w,22])/D(2)
    trace[slot,w,29]=diag[w,21]; trace[slot,w,30]=diag[w,22]
    for j in range(2):
        trace[slot,w,31+j]=diag[w,19+j]; trace[slot,w,33+j]=diag[w,4+j]
        trace[slot,w,35+j]=diag[w,10+j]; trace[slot,w,37+j]=D(ctrl[w,4+j])
        trace[slot,w,54+j]=correction[w,4+j]; trace[slot,w,56+j]=D(v[w,ids[8+j]])
    trace[slot,w,43]=command[w]; trace[slot,w,53]=diag[w,12]
    speeds=V6(); scale=V6()
    for j in range(6): speeds[j]=D(v[w,ids[4+j]]); scale[j]=gains[w,j]
    bounds=command_bounds(speeds,scale)
    trace[slot,w,58]=bounds[4]; trace[slot,w,59]=bounds[5]
    if targets.shape[1]==6:
        for j in range(6): trace[slot,w,47+j]=memory[w,16+j]
    else:
        for j in range(3):
            trace[slot,w,47+2*j]=memory[w,16+j]; trace[slot,w,48+2*j]=-memory[w,16+j]


@wp.kernel
def post(slot:int, q:wp.array2d[float], v:wp.array2d[float], ids:wp.array[int],
         offsets:wp.array3d[wp.vec3d], physical_offsets:wp.array3d[wp.vec3d],
         torque:wp.array2d[float], window:wp.array2d[D], param:wp.array2d[D],
         count:wp.array2d[D], trace:wp.array3d[D]):
    w=wp.tid()
    if trace[slot,w,0]==D(0): return
    keep=int(trace[slot,w,1])%40==0
    rotation=wp.quatd(D(q[w,4]),D(q[w,5]),D(q[w,6]),D(q[w,3]))
    origin=wp.vec3d(D(q[w,0]),D(q[w,1]),D(q[w,2]))
    for side in range(2):
        p=wheel_center(q,ids,offsets,w,side)
        x=param[w,1]*p[0]; before=param[w,1]*trace[slot,w,10+3*side]
        if (x>=window[w,0] and x<=window[w,1]) or (before>=window[w,0] and before<=window[w,1]): keep=True
        for j in range(3): trace[slot,w,16+3*side+j]=p[j]
        mid=(physical_offsets[w,side,0]+physical_offsets[w,side,3])/D(2)
        trace[slot,w,22+side]=wp.length(p-(origin+wp.quat_rotate(rotation,mid)))
    if not keep:
        trace[slot,w,0]=D(0); return
    count[w,1]+=D(1); trace[slot,w,3]=trace[slot,w,1]*D(.0005)
    for j in range(3): trace[slot,w,4+j]=D(q[w,j])
    qw=D(q[w,3]); qx=D(q[w,4]); qy=D(q[w,5]); qz=D(q[w,6])
    roll=wp.atan2(D(2)*(qw*qx+qy*qz),D(1)-D(2)*(qx*qx+qy*qy))
    pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    trace[slot,w,7]=roll; trace[slot,w,8]=pitch; trace[slot,w,9]=yaw
    margin=D(1.e30)
    for j in range(4): margin=wp.min(margin,D(1.4)-wp.abs(D(q[w,ids[j]])))
    trace[slot,w,24]=margin
    for j in range(2): trace[slot,w,39+j]=D(torque[w,4+j]); trace[slot,w,41+j]=D(v[w,ids[8+j]])
    trace[slot,w,44]=wp.cos(yaw)*D(v[w,0])+wp.sin(yaw)*D(v[w,1])
    trace[slot,w,45]=-wp.sin(yaw)*D(v[w,0])+wp.cos(yaw)*D(v[w,1]); trace[slot,w,46]=D(v[w,2])


@wp.kernel
def contact_record(slot:int, ncon:wp.array[int], world:wp.array[int], geom:wp.array[wp.vec2i],
                   dist:wp.array[float], pos:wp.array[wp.vec3], frame:wp.array[wp.mat33],
                   friction:wp.array[vec5], dim:wp.array[int], address:wp.array2d[int],
                   force:wp.array2d[float], njmax:int, cone:int, ids:wp.array[int],
                   body:wp.array[int], root:wp.array[int], cvel:wp.array2d[wp.spatial_vector],
                   com:wp.array2d[wp.vec3], trace:wp.array3d[D]):
    w=wp.tid()
    if trace[slot,w,0]==D(0): return
    # ponytail: scan this small diagnostic batch; indexed contacts if larger banks need it.
    for j in range(ncon[0]):
        if world[j]!=w: continue
        a=geom[j][0]; b=geom[j][1]
        f=contact_force_fn(cone,frame,friction,dim,address,force,njmax,ncon,w,j,False)
        load=D(wp.max(f[0],0.)); normal=wp.vec3(frame[j][0,0],frame[j][0,1],frame[j][0,2])
        relative=point_velocity(body[b],w,pos[j],cvel,com,root)-point_velocity(body[a],w,pos[j],cvel,com,root)
        slip=wp.length(relative-wp.dot(relative,normal)*normal)
        for side in range(2):
            if a!=ids[11+side] and b!=ids[11+side]: continue
            other=a; sign=D(1)
            if a==ids[11+side]: other=b; sign=D(-1)
            col=60+26*side; vertical=wp.max(D(0),sign*load*D(normal[2]))
            trace[slot,w,col]+=D(1); trace[slot,w,col+2]+=load; trace[slot,w,col+4]+=vertical
            target=other==ids[13] or other==ids[14] or (other>=ids[15] and other<=ids[16])
            if not target: continue
            trace[slot,w,col+1]+=D(1); trace[slot,w,col+3]+=load; trace[slot,w,col+5]+=vertical
            if trace[slot,w,col+1]==D(1) or load>trace[slot,w,col+6]:
                trace[slot,w,col+6]=load; trace[slot,w,col+7]=D(other)
                for k in range(3): trace[slot,w,col+8+k]=D(pos[j][k]); trace[slot,w,col+11+k]=D(normal[k])
                trace[slot,w,col+14]=D(friction[j][0]); trace[slot,w,col+15]=D(dim[j]); trace[slot,w,col+16]=D(dist[j]); trace[slot,w,col+17]=D(slip)
            if vertical>trace[slot,w,col+18]:
                trace[slot,w,col+18]=vertical; trace[slot,w,col+19]=D(other)
                for k in range(3): trace[slot,w,col+20+k]=D(pos[j][k]); trace[slot,w,col+23+k]=D(normal[k])


def geometry(raw):
    position=raw.data.geom_xpos.numpy(); rotation=raw.data.geom_xmat.numpy(); size=raw.model.geom_size.numpy()
    ids=raw.ids.numpy(); windows=[]; manifest=[]
    for w,scene in enumerate(raw.scenarios):
        boxes=[]; endpoints=[]; direction=np.sign(scene.speed)
        for j in range(int(ids[13]),int(ids[16])+1):
            half=size[w if len(size)>1 else 0,j]; p=position[w,j]; R=rotation[w,j]
            extent=abs(R)@half
            if p[2]+extent[2]<=0: continue
            boxes.append(dict(geom=j,position=p.tolist(),rotation=R.tolist(),size=half.tolist()))
            endpoints.extend([direction*p[0]-extent[0],direction*p[0]+extent[0]])
        assert boxes and endpoints
        windows.append([min(endpoints)-.2,max(endpoints)+.2]); manifest.append(boxes)
    return np.asarray(windows),manifest


def signature(value):
    if isinstance(value,wp.array): return ('array',value.ptr,tuple(value.shape),str(value.dtype))
    return ('scalar',type(value).__name__,repr(value))


def instrument(factory,cases,mode,directory=None):
    original=[]; rebuilding=[]; current=original
    launch,step=wp.launch,mjw.step
    watched={begin,command_step,control_physical_nominal,reduce_contacts,collect_physical,after,update_shared_reference}
    def spy(kernel,dim,inputs=None,**kwargs):
        if kernel in watched:
            current.append((kernel.key,repr(dim),tuple(signature(x) for x in (inputs or [])),repr(sorted(kwargs.items()))))
        return launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
    def spy_step(model,data,**kwargs):
        current.append(('mjw.step',id(model),id(data),repr(sorted(kwargs.items()))))
        return step(model,data,**kwargs)
    wp.launch,mjw.step=spy,spy_step
    try: raw=factory(cases,mode)
    finally: wp.launch,mjw.step=launch,step
    n=raw.num_envs; windows,boxes=geometry(raw)
    trace=wp.zeros((40,n,WIDTH),dtype=D); mask=wp.array(np.ones(n,np.int32)); count=wp.zeros((n,2),dtype=D); window=wp.array(windows,dtype=D)
    raw._task_mode_buffers=(trace,mask,count,window)
    protected=[raw.data.qpos,raw.data.qvel,raw.data.ctrl,raw.diag,raw.targets,raw.param,raw.state,
               raw.k['state'],raw.k['reference'],raw.nominal_correction,raw.active]
    snapshot=[x.numpy().copy() for x in protected]
    current=rebuilding; wp.launch,mjw.step=spy,spy_step
    try:
        # Same original calls/arguments in the same order; only readonly kernels inserted.
        with wp.ScopedCapture() as capture:
            wp.launch(begin,n,[raw.reward])
            for slot in range(40):
                if raw.shared_reference_enabled and slot%10==0: wp.launch(update_shared_reference,n,raw.shared_reference_args)
                wp.launch(command_step,n,[raw.state,raw.param,raw.command,raw.active,raw.data.qpos,raw.data.qvel,raw.data.qacc_warmstart,raw.stopped_q,raw.stopped_v,raw.stopped_w,raw.contact_flags])
                wp.launch(raw.control_kernel,n,[raw.data.qpos,raw.data.qvel,raw.data.sensordata,raw.targets,raw.command,raw.active,raw.k['state'],raw.ids,raw.k['heights'],raw.k['gains'],raw.k['feed'],raw.k['angles'],raw.k['reference'],raw.k['yaw'],raw.data.ctrl,raw.diag,int(raw.project_clipped_base),int(raw.grouped_residual)]+raw.control_extra,block_dim=32)
                launch(pre,n,[slot,raw.data.qpos,raw.data.qvel,raw.ids,raw.wheel_offsets,raw.active,mask,raw.state,raw.k['state'],raw.k['reference'],raw.param,raw.targets,raw.diag,raw.data.ctrl,raw.command,raw.control_extra[0],raw.nominal_correction,count,trace])
                mjw.step(raw.model,raw.data)
                launch(post,n,[slot,raw.data.qpos,raw.data.qvel,raw.ids,raw.wheel_offsets,raw.physical_args[8],raw.data.actuator_force,window,raw.param,count,trace])
                c=raw.data.contact
                launch(contact_record,n,[slot,raw.data.nacon,c.worldid,c.geom,c.dist,c.pos,c.frame,c.friction,c.dim,c.efc_address,raw.data.efc.force,raw.data.njmax,raw.model.opt.cone,raw.ids,raw.model.geom_bodyid,raw.model.body_rootid,raw.data.cvel,raw.data.subtree_com,trace])
                wp.launch(reduce_contacts,raw.data.naconmax,[raw.data.nacon,c.worldid,c.geom,raw.ids,raw.contact_flags])
                if raw.height_safety=='physical_v1': wp.launch(collect_physical,n,raw.physical_args)
                wp.launch(after,n,[raw.data.qpos,raw.data.qvel,raw.data.sensordata,raw.data.qacc_warmstart,raw.data.time,raw.contact_flags,raw.ids,raw.param,raw.command,raw.state,raw.k['state'],raw.diag,raw.residual,raw.active,raw.done,raw.reward,raw.obs,raw.history,raw.stopped_q,raw.stopped_v,raw.stopped_w,raw.wheel_offsets],block_dim=32)
        raw.graph=capture.graph
    finally: wp.launch,mjw.step=launch,step
    assert original==rebuilding and sum(x[0]=='mjw.step' for x in original)==40
    assert sum(x[0]==update_shared_reference.key for x in original)==4
    for a,b in zip(protected,snapshot): np.testing.assert_array_equal(a.numpy(),b)
    gc.collect(); assert raw._task_mode_buffers[0] is trace
    wait,reset=raw.step_wait,raw.reset; chunks=[[] for _ in cases]; frozen=set()
    def reset_all():
        result=reset(); trace.zero_(); count.zero_(); mask.fill_(1); frozen.clear()
        for x in chunks: x.clear()
        return result
    def step_wait():
        result=wait(); frames=trace.numpy(); counters=count.numpy(); tracking=mask.numpy()
        for w in range(n):
            if w in frozen: continue
            selected=frames[:,w][frames[:,w,0]==1].copy()
            if len(selected): chunks[w].append(selected)
            if result[2][w]:
                data=np.concatenate(chunks[w]) if chunks[w] else np.empty((0,WIDTH))
                assert len(data)==int(counters[w,1]) and np.isfinite(data).all()
                assert int(counters[w,0])==result[3][w]['physical_steps']
                result[3][w]['task_mode_counts']=counters[w].tolist()
                if directory is not None:
                    file=directory/f'case_{cases[w]["seed"]}.npz'
                    np.savez_compressed(file,trace=data,columns=np.array(COL))
                    result[3][w]['task_mode_trace']=dict(path=file.name,sha256=sha(file),rows=len(data))
                tracking[w]=0; frozen.add(w); chunks[w].clear()
        mask.assign(tracking); return result
    raw.reset,raw.step_wait=reset_all,step_wait
    if directory is not None: atomic_json(directory/'geometry.json',dict(boxes=boxes,windows=windows.tolist(),case_ids=[x['seed'] for x in cases],timing='Static GPU geometry; contact/force before integration, moving wheel pose independently labeled pre/post'))
    raw._task_mode_topology=dict(verified=True,original_calls=len(original),physics_calls=40,shared_reference_calls=4,input_pointer_and_scalar_identity=True,protected_arrays_unchanged=True,buffer_owner=True)
    return raw


def evaluate(cases,arm,model=None,norm=None,classical=None,directory=None):
    factory=evaluator.raw_env
    def wrapped(cases,mode): return instrument(factory,cases,mode,directory)
    evaluator.raw_env=wrapped
    try: return evaluator.evaluate(cases,arm,model=model,normalization=norm,classical=classical)
    finally: evaluator.raw_env=factory


def unit():
    from train_height_comparison import raw_env
    p=json.loads((OUT/'proposal.json').read_text()); owners=[]
    for mode in ('diff3','virtual6'):
        raw=instrument(raw_env,p['cases'],mode)
        try:
            raw.reset(); assert not raw.state.numpy()[:,0].any() and not raw.data.time.numpy().any()
            owners.append(dict(mode=mode,worlds=len(p['cases']),**raw._task_mode_topology))
            # CPU forward only: check passive-chain centres in the actual first model.
            data=mujoco.MjData(raw.cpu); data.qpos[:]=raw.data.qpos.numpy()[0]; mujoco.mj_forward(raw.cpu,data)
            inputs=[0,raw.data.qpos,raw.data.qvel,raw.ids,raw.wheel_offsets,raw.active,raw._task_mode_buffers[1],raw.state,raw.k['state'],raw.k['reference'],raw.param,raw.targets,raw.diag,raw.data.ctrl,raw.command,raw.control_extra[0],raw.nominal_correction,raw._task_mode_buffers[2],raw._task_mode_buffers[0]]
            before=[a.numpy().copy() for a in inputs[1:-2]]
            wp.launch(pre,raw.num_envs,inputs); out=raw._task_mode_buffers[0].numpy()
            for side,name in enumerate(('wheelL','wheelR')): np.testing.assert_allclose(out[0,0,10+3*side:13+3*side],data.xpos[raw.cpu.body(name).id],rtol=0,atol=1e-6)
            for a,b in zip(inputs[1:-2],before): np.testing.assert_array_equal(a.numpy(),b)
        finally: raw.close()
    force_unit()
    atomic_json(OUT/'unit.json',dict(verified=True,owners=owners,pose_CPU_forward_equivalence=True,force_contact_API_CPU_equivalence_and_geom_order=True,physical_rollouts=0,training_updates=0))
    print('PASS2x41 current graph/input/owner/reset and CPU pose/force checks',flush=True)


def force_unit():
    xml='<mujoco><option cone="elliptic"/><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .045"><freejoint/><geom type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m); mujoco.mj_forward(m,d); assert d.ncon==1
    contact=d.contact[0]; cpu=np.zeros(6); mujoco.mj_contactForce(m,d,0,cpu); assert cpu[0]>0
    ids=np.zeros(19,np.int32); ids[11:17]=[1,2,0,-1,3,2]
    frame=np.array(contact.frame).reshape(1,3,3).astype(np.float32)
    for swap in (False,True):
        geom=np.array([[0,1]],np.int32); f=frame.copy()
        if swap: geom[:]=[1,0]; f[:,0]*=-1; f[:,1]*=-1
        trace=np.zeros((1,1,WIDTH)); trace[0,0,0]=1; trace[0,0,67]=trace[0,0,79]=-1
        arrays=[wp.array(np.array([1],np.int32)),wp.array(np.array([0],np.int32)),wp.array(geom),wp.array(np.array([contact.dist],np.float32)),wp.array(np.array(contact.pos,np.float32).reshape(1,3),dtype=wp.vec3),wp.array(f,dtype=wp.mat33),wp.array(np.array(contact.friction,np.float32).reshape(1,5),dtype=vec5),wp.array(np.array([contact.dim],np.int32)),wp.array(np.arange(contact.efc_address,contact.efc_address+6,dtype=np.int32).reshape(1,6)),wp.array(np.asarray(d.efc_force,np.float32).reshape(1,-1))]
        tail=[wp.array(ids),wp.array(np.asarray(m.geom_bodyid,np.int32)),wp.array(np.asarray(m.body_rootid,np.int32)),wp.array(np.asarray(d.cvel,np.float32).reshape(1,m.nbody,6),dtype=wp.spatial_vector),wp.array(np.asarray(d.subtree_com,np.float32).reshape(1,m.nbody,3),dtype=wp.vec3)]
        out=wp.array(trace,dtype=D); before=[a.numpy().copy() for a in arrays+tail]
        wp.launch(contact_record,1,[0,*arrays,d.nefc,int(m.opt.cone),*tail,out]); value=out.numpy()[0,0]
        np.testing.assert_allclose(value[62],cpu[0],rtol=1e-6,atol=1e-5); np.testing.assert_allclose(value[64],cpu[0],rtol=1e-6,atol=1e-5)
        assert value[60]==value[61]==1 and value[67]==0
        for a,b in zip(arrays+tail,before): np.testing.assert_array_equal(a.numpy(),b)


def freeze():
    p=json.loads((OUT/'proposal.json').read_text()); parent=ROOT/p['source_parent']; c=json.loads(parent.read_text())
    assert sha(parent)==p['source_parent_sha256'] and all(sha(ROOT/n)==v for n,v in c['source_sha256'].items())
    assert not (OUT/'source_contract.json').exists(); unit()
    sources=dict(c['source_sha256'])
    for n in ('task_mode_recorder.py','route_pilot_env.py','route_state.py','trace_failure_chain.py'): sources['wheelleg_warp/'+n]=sha(ROOT/'wheelleg_warp'/n)
    atomic_json(OUT/'source_contract.json',dict(proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,unit_sha256=sha(OUT/'unit.json'),contact_api_sha256=sha(Path(inspect.getfile(contact_force_fn.func))),columns=COL,budget_first_episode_evaluations=205,training_updates=0,
        graph_semantics='Fresh capture with identical original launch/step sequence and exact input pointer/scalar arguments; adds readonly kernels, keeps diag/shared-reference. Not a trajectory bitwise claim.',contact_semantics='Solver force and contact data belong to pre-integration layer; normal load and wheel-side vertical normal component are not total friction-inclusive support forces. Max-force and max-vertical target witnesses separate.'))
    print('ADMITTED205 source, no evaluations run',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['unit','freeze']); globals()[parser.parse_args().command]()
