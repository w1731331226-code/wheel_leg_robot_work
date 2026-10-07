"""Map-only outer capture hook on unchanged complete B0 evaluation pipeline."""
import numpy as np
import warp as wp
import execution_history_evaluation as evaluation
import continuous_nominal_map as mapping
import continuous_nominal_control as candidate
from native.controller import D
from task_mode_recorder import signature
from review_yaw_sector import sha

BASE_MAKE_ENV=evaluation.make_env
phase=evaluation.parking.phase
# ponytail: one sequential queue; use per-evaluation ownership if concurrency is added.
LAST_RAW=None


@wp.kernel
def record(slot:int,state:wp.array2d[D],command:wp.array[D],active:wp.array[int],
           reference:wp.array2d[D],log:wp.array3d[D]):
    w=wp.tid()
    for j in range(9):log[slot,w,j]=D(0)
    if active[w]==0:return
    log[slot,w,0]=D(1);log[slot,w,1]=state[w,0]+D(1)
    log[slot,w,2]=command[w];log[slot,w,3]=reference[w,2]
    for j in range(5):log[slot,w,4+j]=reference[w,16+j]


def make_map(cases,arm,normalization,directory):
    global LAST_RAW
    assert arm=='B0' and normalization is None
    n=len(cases);table=mapping.arrays('cuda:0')
    private=wp.zeros((n,21),dtype=D,device='cuda:0')
    log=wp.zeros((40,n,9),dtype=D,device='cuda:0')
    launch=wp.launch;command=state=None;actual=[]
    def intercept(kernel,dim,inputs=None,**kwargs):
        nonlocal command,state
        if kernel is phase.rec.old.command_step:state,command=inputs[0],inputs[2]
        if kernel is phase.roles.experimental.control_physical_nominal:
            assert command is inputs[4] and state is not None and inputs[12].shape[1]==16
            slot=len(actual)%40
            launch(mapping.prepare,dim,[inputs[12],command,inputs[5],*table,private])
            launch(record,dim,[slot,state,command,inputs[5],private,log])
            args=list(inputs);args[12]=private
            actual.append(tuple(signature(x) for x in args))
            return launch(candidate.control_physical_nominal,dim,args,**kwargs)
        return launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
    wp.launch=intercept
    try:result=BASE_MAKE_ENV(cases,arm,normalization,directory)
    finally:wp.launch=launch
    raw=result[0]
    assert len(actual)==80 and actual[:40]==actual[40:]
    assert all(s[4]==signature(raw.command) and s[14]==signature(raw.data.ctrl) for s in actual)
    raw._continuous_map_buffers=(private,log,*table)
    raw._continuous_map_topology=dict(verified=True,captured_controller_calls=80,captures=2,
        calls_per_capture=40,current_command_owner=True,final_ctrl_owner_unchanged=True,
        actual_reference_shape=[n,21],original_role_reference_shape=[n,16],
        signatures_same_in_both_captures=True,GPU_table_only=True,source_table_sha256=sha(mapping.OUT/'delta_table.npz'))
    chunks=[[] for _ in cases];frozen=set()
    reset,wait=raw.reset,raw.step_wait
    def reset_all():
        private.zero_();log.zero_();frozen.clear()
        for part in chunks:part.clear()
        return reset()
    def step_wait():
        output=wait();frames=raw._complete_buffers[0].numpy();values=log.numpy()
        for w in range(n):
            if w in frozen:continue
            keep=frames[:,w,0]==1;selected=values[keep,w].copy()
            np.testing.assert_array_equal(selected[:,:2],frames[keep,w,:2])
            np.testing.assert_array_equal(selected[:,2],frames[keep,w,43])
            if len(selected):chunks[w].append(selected)
            if output[2][w]:
                data=np.concatenate(chunks[w]);assert len(data)==output[3][w]['physical_steps']
                np.testing.assert_array_equal(data[:,1],np.arange(1,len(data)+1))
                assert np.isfinite(data).all()
                assert np.all((data[:,3]>=.115)&(data[:,3]<=.38))
                assert np.all((data[:,2]>=-1.)&(data[:,2]<=1.))
                np.testing.assert_array_equal(data[:,8],(data[:,2]!=0.).astype(float))
                np.testing.assert_array_equal(data[data[:,2]==0.,4:],0.)
                path=directory/f'case_map_{cases[w]["seed"]}.npz'
                assert not path.exists()
                np.savez_compressed(path,trace=data,columns=np.array([
                    'valid','step','current_command','public_height','delta_wheel','delta_hub',
                    'delta_support','delta_theta','enabled']))
                output[3][w]['map_trace']=dict(path=path.name,sha256=sha(path),rows=len(data))
                frozen.add(w);chunks[w].clear()
        return output
    raw.reset,raw.step_wait=reset_all,step_wait
    raw._continuous_map_chunks=chunks
    raw._continuous_map_frozen=frozen
    LAST_RAW=raw
    return result


def preserve(raw,directory):
    if not hasattr(raw,'_continuous_map_buffers'):return
    private,log,*table=raw._continuous_map_buffers
    np.savez_compressed(directory/'interrupted_map_buffers.npz',reference=private.numpy(),log=log.numpy())
    for i,parts in enumerate(raw._continuous_map_chunks):
        if parts:np.savez_compressed(directory/f'interrupted_map_prefix_{i}.npz',trace=np.concatenate(parts))
