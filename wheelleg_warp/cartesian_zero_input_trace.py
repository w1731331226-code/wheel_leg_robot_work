"""Identical read-only observer kernels around the original light control chain."""
import warp as wp
import native.environment as native
import light_phase_reference as phase
import cartesian_light_runtime as cart
from native.controller import D
from task_mode_recorder import signature

PRE_WIDTH=66  # valid,active,step,memory5,gyro,q17,v16,sensor28
BOUNDARIES=('clear_before','clear_after','Nom_before','Nom_after','deliver_before','deliver_after')


@wp.kernel
def pre(slot:int,q:wp.array2d[float],v:wp.array2d[float],sensor:wp.array2d[float],
        memory:wp.array2d[D],state:wp.array2d[D],ids:wp.array[int],active:wp.array[int],out:wp.array3d[D]):
    w=wp.tid();out[slot,w,0]=D(1);out[slot,w,1]=D(active[w]);out[slot,w,2]=state[w,0]+D(1)
    out[slot,w,3]=memory[w,5];out[slot,w,4]=D(sensor[w,ids[10]])
    for j in range(17):out[slot,w,5+j]=D(q[w,j])
    for j in range(16):out[slot,w,22+j]=D(v[w,j])
    for j in range(28):out[slot,w,38+j]=D(sensor[w,j])


@wp.kernel
def boundary(slot:int,stage:int,memory:wp.array2d[D],sensor:wp.array2d[float],ids:wp.array[int],
             active:wp.array[int],out:wp.array4d[D]):
    w=wp.tid();out[slot,w,stage,0]=D(1);out[slot,w,stage,1]=D(active[w])
    out[slot,w,stage,2]=memory[w,5];out[slot,w,stage,3]=D(sensor[w,ids[10]])
    for j in range(6):out[slot,w,stage,4+j]=memory[w,16+j]


@wp.kernel
def control(slot:int,stage:int,ctrl:wp.array2d[float],diag:wp.array2d[D],out:wp.array4d[D]):
    w=wp.tid()
    for j in range(6):out[slot,w,stage,j]=D(ctrl[w,j])
    for j in range(38):out[slot,w,stage,6+j]=diag[w,j]


def instrument(factory,cases,cartesian):
    n=len(cases);assert n in (3,10) and type(cartesian) is bool
    wp.init();wp.set_device('cuda:0')
    before=wp.zeros((40,n,PRE_WIDTH),dtype=D);edges=wp.zeros((40,n,6,10),dtype=D);controls=wp.zeros((40,n,2,44),dtype=D)
    launch=wp.launch;counts=dict(phase=0,Nom=0,clear=0,deliver=0);command_args=phase_args=None;slot=-1;bindings=[]
    def mark(stage):
        launch(boundary,n,[slot,stage,phase_args[6],phase_args[4],phase_args[5],phase_args[8],edges])
    def intercept(kernel,dim,inputs=None,**kwargs):
        nonlocal command_args,phase_args,slot
        if kernel is native.command_step:command_args=inputs
        if kernel is phase.phase.roles.prepare:
            slot=counts['phase'];counts['phase']+=1;phase_args=inputs
            assert slot<40 and inputs[0]==slot and command_args is not None
            assert inputs[3] is command_args[4] and inputs[7] is command_args[0] and inputs[8] is command_args[3]
            assert inputs[3].shape==(n,17) and command_args[5].shape==(n,16) and inputs[4].shape==(n,28)
            launch(pre,n,[slot,inputs[3],command_args[5],inputs[4],inputs[6],inputs[7],inputs[5],inputs[8],before])
        if kernel is cart.clear_native:
            assert cartesian and inputs[0] is phase_args[8] and inputs[1] is phase_args[6]
            mark(0);value=launch(kernel,dim,inputs,**kwargs);mark(1);counts['clear']+=1;return value
        if kernel is cart.NOMINAL:
            assert counts['Nom']==slot
            assert all(inputs[i] is phase_args[j] for i,j in ((0,3),(2,4),(5,8),(6,6),(7,5)))
            assert inputs[1] is command_args[5] and inputs[15].shape==(n,38)
            bindings.append([signature(inputs[i]) for i in (0,1,2,5,6,7,14,15)])
            # B0 receives the same observer calls at the two absent Cart boundaries.
            if not cartesian:mark(0);mark(1)
            mark(2);launch(control,n,[slot,0,inputs[14],inputs[15],controls])
            value=launch(kernel,dim,inputs,**kwargs)
            mark(3);launch(control,n,[slot,1,inputs[14],inputs[15],controls]);counts['Nom']+=1
            if not cartesian:mark(4);mark(5)
            return value
        if kernel is cart.deliver:
            assert cartesian and inputs[0]==slot and inputs[9] is phase_args[6]
            mark(4);value=launch(kernel,dim,inputs,**kwargs);mark(5);counts['deliver']+=1;return value
        return launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
    wp.launch=intercept
    try:
        raw=factory()
        assert counts==dict(phase=40,Nom=40,clear=40 if cartesian else 0,deliver=40 if cartesian else 0)
        assert all(b==bindings[0] for b in bindings)
        assert bindings[0]==[signature(b) for b in (raw.data.qpos,raw.data.qvel,raw.data.sensordata,raw.active,raw.k['state'],raw.ids,raw.data.ctrl,raw.diag)]
    finally:wp.launch=launch
    raw._zero_input_buffers=dict(pre=before,boundaries=edges,control=controls)
    raw._zero_input_topology=dict(counts=counts,observer_calls_each_slot=9,cartesian=cartesian,
        bound_to_actual_control=True,pre_width=PRE_WIDTH,boundary_names=BOUNDARIES,actual_control_signatures=bindings[0])
    reset,wait=raw.reset,raw.step_wait;raw._zero_input_terminal=None
    def reset_all():
        value=reset()
        for b in raw._zero_input_buffers.values():b.zero_()
        raw._zero_input_terminal=None
        return value
    def step_wait():
        raw._zero_input_terminal={name:b.numpy() for name,b in dict(state=raw.state,public=raw.obs,done=raw.done,
            controller_memory=raw.k['state'],q=raw.data.qpos,v=raw.data.qvel,sensors=raw.data.sensordata,
            stopped_q=raw.stopped_q,stopped_v=raw.stopped_v).items()}
        return wait()
    raw.reset,raw.step_wait=reset_all,step_wait
    return raw
