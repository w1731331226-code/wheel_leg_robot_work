"""Experimental request withdrawal; unchanged controller/filter and actor packet."""
import numpy as np
import warp as wp
import phase_support_probe as phase

D=phase.D
BASE_FACTORY=phase.instrument
COL=['valid','step','current_command','seen_motion','withdraw']
COL += [f'{name}_{j}' for name in ('requested','effective','filtered_before','filtered_after') for j in range(6)]
COL += ['control_error']


@wp.kernel
def prepare(slot:int, enabled:int, command:wp.array[D], active:wp.array[int],
            env_state:wp.array2d[D], control_state:wp.array2d[D], requested:wp.array2d[float],
            seen:wp.array[int], effective:wp.array2d[float], log:wp.array3d[D]):
    w=wp.tid()
    for j in range(30):log[slot,w,j]=D(0)
    for j in range(6):effective[w,j]=requested[w,j]
    if active[w]==0:return
    if command[w]!=D(0):seen[w]=1
    withdraw=enabled==1 and seen[w]==1 and command[w]==D(0)
    log[slot,w,0]=D(1);log[slot,w,1]=env_state[w,0]+D(1)
    log[slot,w,2]=command[w];log[slot,w,3]=D(seen[w]);log[slot,w,4]=D(int(withdraw))
    for j in range(6):
        if withdraw:effective[w,j]=0.
        log[slot,w,5+j]=D(requested[w,j]);log[slot,w,11+j]=D(effective[w,j])
        log[slot,w,17+j]=control_state[w,16+j]


@wp.kernel
def finish(slot:int,state:wp.array2d[D],diag:wp.array2d[D],log:wp.array3d[D]):
    w=wp.tid()
    if log[slot,w,0]==D(0):return
    for j in range(6):log[slot,w,23+j]=state[w,16+j]
    log[slot,w,29]=diag[w,14]


@wp.kernel
def clear_rows(mask:wp.array[int],seen:wp.array[int]):
    w=wp.tid()
    if mask[w]:seen[w]=0


def check_log(data,condition):
    assert condition in ('original','parking_request_withdrawal')
    assert data.ndim==2 and data.shape[1]==len(COL) and np.isfinite(data).all()
    np.testing.assert_array_equal(data[:,0],1)
    np.testing.assert_array_equal(data[:,1],np.arange(1,len(data)+1))
    expected=np.maximum.accumulate(data[:,2]!=0)
    np.testing.assert_array_equal(data[:,3],expected)
    gate=expected & (data[:,2]==0) & (condition=='parking_request_withdrawal')
    np.testing.assert_array_equal(data[:,4],gate)
    np.testing.assert_array_equal(data[:,11:17],np.where(gate[:,None],0,data[:,5:11]))
    after=data[:,17:23]+np.clip(data[:,11:17]-data[:,17:23],-.01,.01)
    processed=np.all(data[:,23:29]==after,axis=1)
    assert np.all(processed | ((data[:,29]==2)&np.all(data[:,23:29]==data[:,17:23],axis=1)))


def instrument(cases,mode,condition,directory=None):
    if condition not in ('original','parking_request_withdrawal') or mode!='virtual6':
        raise ValueError('Registered two-condition diagnostic requires virtual6')
    n=len(cases);seen=wp.zeros(n,dtype=wp.int32);effective=wp.zeros((n,6),dtype=wp.float32)
    log=wp.zeros((40,n,len(COL)),dtype=D)
    wp.load_module(module=__name__,device='cuda')
    launch=wp.launch;env_state=command=None;count=0;events=[];actual=[]
    def intercept(kernel,dim,inputs=None,**kwargs):
        nonlocal env_state,command,count
        if kernel is phase.rec.old.command_step:
            env_state,command=inputs[0],inputs[2];events.append('command')
        if kernel is phase.roles.experimental.control_physical_nominal:
            assert env_state is not None and command is inputs[4] and events[-1]=='command'
            slot=count%40;count+=1
            launch(prepare,dim,[slot,int(condition!='original'),command,inputs[5],env_state,inputs[6],inputs[3],seen,effective,log])
            args=list(inputs);args[3]=effective
            actual.append(tuple(phase.rec.old.signature(x) for x in args))
            result=launch(kernel,dim,args,**kwargs);launch(finish,dim,[slot,inputs[6],inputs[15],log])
            events.append('control');return result
        return launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
    wp.launch=intercept
    try:raw=BASE_FACTORY(cases,mode,directory=directory)
    finally:wp.launch=launch
    assert count==80 and actual[:40]==actual[40:] and events==['command','control']*80
    raw._parking_buffers=(seen,effective,log)
    raw._parking_chunks=[[] for _ in cases];raw._parking_frozen=set()
    raw._parking_contract=dict(verified=True,condition=condition,columns=COL,controller_calls_per_capture=40,
        actual_control_signatures_identical_in_captures=True,only_controller_target_pointer_replaced=True,
        current_command_owner_verified=True,latch_private_to_controller=True,actor_observation_dimensions=39,
        filter='Original state16:22 slew limiter:delta clipped to0.01 per0.5ms. No state/output hard reset on parking.')
    if directory is not None:phase.rec.atomic_json(directory/'parking_contract.json',raw._parking_contract)
    reset,wait=raw.reset,raw.step_wait
    def reset_all():
        seen.zero_();effective.zero_();log.zero_();raw._parking_frozen.clear()
        for c in raw._parking_chunks:c.clear()
        return reset()
    def step_wait():
        result=wait();frames=raw._complete_buffers[0].numpy();values=log.numpy()
        for w in range(n):
            if w in raw._parking_frozen:continue
            keep=frames[:,w,0]==1;selected=values[keep,w].copy()
            np.testing.assert_array_equal(selected[:,:2],frames[keep,w,:2])
            np.testing.assert_array_equal(selected[:,2],frames[keep,w,43])
            if len(selected):raw._parking_chunks[w].append(selected)
            if result[2][w]:
                data=np.concatenate(raw._parking_chunks[w]);check_log(data,condition)
                assert len(data)==result[3][w]['physical_steps']
                if directory is not None:
                    file=directory/f'case_parking_{raw._broad_contract["recording_ids"][w]}.npz';assert not file.exists()
                    np.savez_compressed(file,trace=data,columns=np.array(COL))
                    result[3][w]['parking_trace']=dict(path=file.name,sha256=phase.rec.sha(file),rows=len(data))
                raw._parking_frozen.add(w);raw._parking_chunks[w].clear()
        # Native autoreset has completed; private latch is cleared before the next control call.
        if result[2].any():wp.launch(clear_rows,n,[raw.mask,seen])
        return result
    raw.reset,raw.step_wait=reset_all,step_wait
    return raw


def preserve(raw,directory):
    seen,effective,log=raw._parking_buffers
    f=directory/'interrupted_parking_buffers.npz'
    np.savez_compressed(f,seen=seen.numpy(),effective=effective.numpy(),log=log.numpy())
    for w,c in enumerate(raw._parking_chunks):
        if c:np.savez_compressed(directory/f'interrupted_parking_prefix_{w}.npz',trace=np.concatenate(c),columns=np.array(COL))
    return dict(last_buffers_sha256=phase.rec.sha(f),complete_worlds=sorted(raw._parking_frozen))
