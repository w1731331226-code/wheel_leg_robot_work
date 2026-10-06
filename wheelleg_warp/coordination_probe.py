"""Experimental shadow-query coordinator within one existing Nom authority pool."""
import numpy as np
import warp as wp
import phase_support_probe as phase
import coordinated_nominal_query as query
import coordination_nominal_budget as budget
from speed_yaw_coordination import propose

D=phase.D
BASE_FACTORY=phase.instrument
COL=['valid','step','public_command','gamma','motion_reference','updated5ms']
COL += [f'{name}_{j}' for name in ('original_Nom','motion_delta','damping_request','total_Nom') for j in range(6)]
COL += ['original_query_error','motion_query_error']


def prepare_packet(raw,condition):
    if raw._coord_packet is None:raise RuntimeError('Reset required')
    gamma,damping,*_=raw._coord_buffers
    packet=np.c_[raw._coord_packet,np.zeros(raw.num_envs,np.float32)]
    result=propose(packet,gamma.numpy())
    gamma.assign(result['gamma'] if condition=='Cgamma' else np.ones(raw.num_envs))
    value=np.zeros((raw.num_envs,6));value[:,4:]=result['wheel_damping_request_Nm'] if condition!='B0' else 0
    damping.assign(value)


@wp.kernel
def motion_reference(command:wp.array[D],gamma:wp.array[D],reference:wp.array2d[D],shadow:wp.array2d[D]):
    w=wp.tid()
    for j in range(16):shadow[w,j]=reference[w,j]
    shadow[w,16]=gamma[w]*command[w]


@wp.kernel
def difference(original:wp.array2d[D],shadow:wp.array2d[D],active:wp.array[int],delta:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    valid=original[w,14]!=D(2) and shadow[w,14]!=D(2)
    for j in range(6):
        if not wp.isfinite(original[w,15+j]) or not wp.isfinite(shadow[w,15+j]):valid=False
    for j in range(6):
        delta[w,j]=D(0)
        if valid:delta[w,j]=shadow[w,15+j]-original[w,15+j]


@wp.kernel
def record(slot:int,update:int,task:wp.array2d[D],command:wp.array[D],gamma:wp.array[D],active:wp.array[int],
           original:wp.array2d[D],delta:wp.array2d[D],damping:wp.array2d[D],total:wp.array2d[D],query0:wp.array2d[D],query1:wp.array2d[D],log:wp.array3d[D]):
    w=wp.tid()
    for j in range(32):log[slot,w,j]=D(0)
    if active[w]==0:return
    log[slot,w,0]=D(1);log[slot,w,1]=task[w,0]+D(1);log[slot,w,2]=command[w]
    log[slot,w,3]=gamma[w];log[slot,w,4]=gamma[w]*command[w];log[slot,w,5]=D(update)
    log[slot,w,30]=query0[w,14];log[slot,w,31]=query1[w,14]
    for j in range(6):
        log[slot,w,6+j]=original[w,j];log[slot,w,12+j]=delta[w,j]
        log[slot,w,18+j]=damping[w,j];log[slot,w,24+j]=total[w,j]


def instrument(cases,mode,condition,directory=None):
    if condition not in ('B0','Bomega','Cgamma') or mode!='virtual6':raise ValueError('Registered analytic conditions/virtual6 only')
    n=len(cases);gamma=wp.ones(n,dtype=D);damping=wp.zeros((n,6),dtype=D);delta=wp.zeros((n,6),dtype=D)
    total=wp.zeros((n,6),dtype=D);previous=wp.zeros((n,6),dtype=D);shadow_ref=wp.zeros((n,17),dtype=D)
    states=[wp.zeros((n,30),dtype=D) for _ in range(2)];diags=[wp.zeros((n,38),dtype=D) for _ in range(2)]
    controls=[wp.zeros((n,6),dtype=wp.float32) for _ in range(2)];log=wp.zeros((40,n,32),dtype=D)
    for module in (__name__,query.__name__,budget.__name__):wp.load_module(module=module,device='cuda')
    launch=wp.launch;count=0;env_state=command=None;actual=[]
    def intercept(kernel,dim,inputs=None,**kwargs):
        nonlocal count,env_state,command
        if kernel is phase.rec.old.command_step:env_state,command=inputs[0],inputs[2]
        if kernel is phase.roles.experimental.control_physical_nominal:
            assert command is inputs[4] and inputs[6].shape==(n,30) and inputs[15].shape==(n,38)
            assert inputs[3].shape==(n,6) and inputs[12].shape==(n,16)
            slot=count%40;count+=1;update=int(slot%10==0)
            if condition=='B0':wp.copy(total,inputs[-1])
            elif update:
                for i in range(2):
                    wp.copy(states[i],inputs[6]);args=list(inputs);args[6]=states[i];args[14]=controls[i];args[15]=diags[i]
                    if i==1 and condition=='Cgamma':
                        launch(motion_reference,dim,[command,gamma,inputs[12],shadow_ref]);args[12]=shadow_ref
                        launch(query.control_physical_nominal,dim,args,**kwargs)
                    else:launch(kernel,dim,args,**kwargs)
                launch(difference,dim,[diags[0],diags[1],inputs[5],delta]);wp.copy(previous,total)
                launch(budget.compose,dim,[previous,inputs[-1],delta,damping,inputs[5],total])
            launch(record,dim,[slot,update,env_state,command,gamma,inputs[5],inputs[-1],delta,damping,total,*diags,log])
            args=list(inputs);args[-1]=total;actual.append(tuple(phase.rec.old.signature(x) for x in args))
            return launch(kernel,dim,args,**kwargs)
        return launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
    wp.launch=intercept
    try:raw=BASE_FACTORY(cases,mode,directory=directory)
    finally:wp.launch=launch
    assert count==80 and actual[:40]==actual[40:]
    raw._coord_buffers=(gamma,damping,delta,total,previous,log);raw._coord_chunks=[[] for _ in cases];raw._coord_frozen=set()
    raw._coord_contract=dict(condition=condition,columns=COL,packet_age='Same delayed raw38 subset plus unused route placeholder;20ms hold',
        Nom_budget='One total request +/-1Nm,delta L1<=.1Nm at slots0/10/20/30;B0 exact original pool',public_command_unchanged=True,shadow_state_private=True,
        shadow_velocity_tilt_integral_consistent=True,actor_dimension_unchanged=True)
    if directory is not None:phase.rec.atomic_json(directory/'coordination_contract.json',raw._coord_contract)
    raw._coord_packet=None;reset,async_step,wait=raw.reset,raw.step_async,raw.step_wait
    def clear():
        gamma.fill_(1)
        for a in (damping,delta,total,previous,log):a.zero_()
        raw._coord_frozen.clear()
        for c in raw._coord_chunks:c.clear()
    def reset_all():clear();value=reset();raw._coord_packet=value.copy();return value
    def step_async(actions):
        prepare_packet(raw,condition);async_step(actions)
    def step_wait():
        result=wait();raw._coord_packet=result[0].copy();frames=raw._complete_buffers[0].numpy();values=log.numpy()
        for w in range(n):
            if w in raw._coord_frozen:continue
            selected=values[frames[:,w,0]==1,w].copy()
            np.testing.assert_array_equal(selected[:,:2],frames[frames[:,w,0]==1,w,:2])
            if len(selected):raw._coord_chunks[w].append(selected)
            if result[2][w]:
                data=np.concatenate(raw._coord_chunks[w]);check_log(data,condition)
                if directory is not None:
                    file=directory/f'case_coordination_{raw._broad_contract["recording_ids"][w]}.npz';assert not file.exists()
                    np.savez_compressed(file,trace=data,columns=np.array(COL));result[3][w]['coordination_trace']=dict(path=file.name,sha256=phase.rec.sha(file),rows=len(data))
                raw._coord_frozen.add(w);raw._coord_chunks[w].clear()
        # Reset private per-world history after Native autoreset,before the next graph.
        if result[2].any():
            mask=result[2];g=gamma.numpy();g[mask]=1;gamma.assign(g)
            for a in (damping,delta,total,previous):v=a.numpy();v[mask]=0;a.assign(v)
        return result
    raw.reset,raw.step_async,raw.step_wait=reset_all,step_async,step_wait
    return raw


def check_log(data,condition):
    assert data.ndim==2 and data.shape[1]==32 and np.isfinite(data).all()
    np.testing.assert_array_equal(data[:,1],np.arange(1,len(data)+1))
    np.testing.assert_array_equal(data[:,5],(np.arange(len(data))%10)==0)
    assert np.all((data[:,3]>=0)&(data[:,3]<=1)) and np.all(abs(data[:,24:30])<=1+1e-14)
    np.testing.assert_array_equal(data[:,4],data[:,3]*data[:,2])
    if condition=='B0':np.testing.assert_array_equal(data[:,24:30],data[:,6:12])
    else:
        previous=np.zeros((1,6))
        for row in data:
            if row[5]:previous=budget.bounded_total(previous,row[6:12][None],row[12:18][None],row[18:24][None])
            np.testing.assert_allclose(row[24:30],previous[0],atol=1e-14,rtol=0)


def preserve(raw,directory):
    file=directory/'interrupted_coordination_buffers.npz'
    np.savez_compressed(file,**{n:a.numpy() for n,a in zip(('gamma','damping','delta','total','previous','log'),raw._coord_buffers)})
    for w,c in enumerate(raw._coord_chunks):
        if c:np.savez_compressed(directory/f'interrupted_coordination_prefix_{w}.npz',trace=np.concatenate(c),columns=np.array(COL))
    return dict(last_buffers_sha256=phase.rec.sha(file),complete_worlds=sorted(raw._coord_frozen))
