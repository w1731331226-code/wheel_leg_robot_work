"""Declared reference-role arms on the existing dense/noisy measurement pipeline."""
import inspect
import json
from pathlib import Path
import numpy as np
import warp as wp
import native.controller as original
import native.environment as environment
import reference_role_control as experimental
import nom_yaw_filter_probe as noise

rec=noise.rec;D=rec.D
OUT=rec.ROOT/'wheelleg_warp/results/paper_recovery_20261004/reference_role_probe_v1'
COL=['valid','step','base_mean','tracking_mean','tracking_lower','guard_lower',
     'requested_offset','left_target','right_target','guard_requested','guard_applied']


@wp.kernel
def prepare(slot:int,mode:int,base:wp.array2d[D],q:wp.array2d[float],sensor:wp.array2d[float],
            ids:wp.array[int],memory:wp.array2d[D],state:wp.array2d[D],active:wp.array[int],
            ref:wp.array2d[D],log:wp.array3d[D]):
    w=wp.tid()
    for j in range(16):ref[w,j]=D(0)
    for j in range(base.shape[1]):ref[w,j]=base[w,j]
    for j in range(11):log[slot,w,j]=D(0)
    if active[w]==0:return
    roll=wp.atan2(D(2)*(D(q[w,3])*D(q[w,4])+D(q[w,5])*D(q[w,6])),D(1)-D(2)*(D(q[w,4])*D(q[w,4])+D(q[w,5])*D(q[w,5])))
    rate=memory[w,5]+(D(sensor[w,ids[10]])-memory[w,5])*D(.05)
    offset=wp.clamp(D(.3)*roll+D(.12)*rate,D(-.035),D(.035))
    if mode>=1:ref[w,15]=D(.115)
    if mode==2:
        left=wp.clamp(base[w,2]+offset,D(.115),base[w,6])
        right=wp.clamp(base[w,2]-offset,D(.115),base[w,6])
        ref[w,2]=(left+right)/D(2);ref[w,3]=D(.115)
    floor=wp.min(ref[w,2],D(.160))
    if mode>=1:floor=D(.115)
    log[slot,w,0]=D(1);log[slot,w,1]=state[w,0]+D(1)
    log[slot,w,2]=base[w,2];log[slot,w,3]=ref[w,2];log[slot,w,4]=ref[w,3]
    log[slot,w,5]=floor;log[slot,w,6]=offset


@wp.kernel
def finish(slot:int,diag:wp.array2d[D],log:wp.array3d[D]):
    w=wp.tid()
    if log[slot,w,0]==D(0):return
    log[slot,w,7]=diag[w,21];log[slot,w,8]=diag[w,22]
    log[slot,w,9]=diag[w,31];log[slot,w,10]=diag[w,32]


def check_control_copy():
    expected=inspect.getsource(original.control_step.func)
    needle='        radial_reference=wp.min(reference[w,2],D(sim.L_SQUAT_MIN))\n'
    patch=needle+'        if reference.shape[1]>15 and reference[w,15]>D(0):\n            radial_reference=reference[w,15]\n'
    assert expected.count(needle)==1
    assert inspect.getsource(experimental.control_step.func)==expected.replace(needle,patch,1)
    assert inspect.getsource(experimental.control_physical_nominal.func)==inspect.getsource(original.control_physical_nominal.func)


def check_log(data,mode):
    assert np.isfinite(data).all() and np.array_equal(data[:,1],np.arange(1,len(data)+1))
    np.testing.assert_allclose((data[:,7]+data[:,8])/2,data[:,3],atol=1e-12,rtol=0)
    assert np.all(data[:,10]>=0) and np.all(data[:,10]<=data[:,9]+1e-9)
    if mode==2:
        left=np.clip(data[:,2]+data[:,6],.115,.38);right=np.clip(data[:,2]-data[:,6],.115,.38)
        np.testing.assert_allclose(data[:,7:9],np.c_[left,right],atol=1e-12,rtol=0)
        assert np.all(abs(data[:,3]-data[:,2])<=.0175+1e-12)
        assert np.all(data[:,5]<=np.minimum(data[:,7],data[:,8])+1e-12)
    else:np.testing.assert_array_equal(data[:,3],data[:,2])
    if mode>=1:np.testing.assert_array_equal(data[:,5],.115)


def instrument(cases,native_mode,arm,std,directory=None):
    check_control_copy();mode={'original':0,'floor_only':1,'consistent_pair':2}[arm]
    ref=wp.zeros((len(cases),16),dtype=D);log=wp.zeros((40,len(cases),11),dtype=D)
    launch=wp.launch;kernel=environment.control_physical_nominal;watched=rec.old.control_physical_nominal
    count=0;envstate=None;actual=[]
    def intercept(k,dim,inputs=None,**kwargs):
        nonlocal count,envstate
        if k is rec.old.command_step:envstate=inputs[0]
        if k is experimental.control_physical_nominal:
            slot=count%40;count+=1
            assert inputs[12].shape[1]==10 and envstate is not None
            launch(prepare,dim,[slot,mode,inputs[12],inputs[0],inputs[2],inputs[7],inputs[6],envstate,inputs[5],ref,log])
            args=list(inputs);args[12]=ref
            actual.append(tuple(rec.old.signature(x) for x in args))
            result=launch(k,dim,args,**kwargs);launch(finish,dim,[slot,inputs[15],log]);return result
        return launch(k,dim,**kwargs) if inputs is None else launch(k,dim,inputs,**kwargs)
    wp.launch=intercept;environment.control_physical_nominal=experimental.control_physical_nominal
    rec.old.control_physical_nominal=experimental.control_physical_nominal
    try:
        raw=noise.instrument(cases,native_mode,dict(label=arm,alpha=.025,noise_std_rad_s=std),directory)
    finally:
        wp.launch=launch;environment.control_physical_nominal=kernel;rec.old.control_physical_nominal=watched
    assert count==80 and actual[:40]==actual[40:]
    raw._role_buffers=(ref,log);raw._role_chunks=[[] for _ in cases];raw._role_frozen=set()
    raw._role_topology=dict(verified=True,actual_controller_input_signatures_identical_in_captures=True,
        controller_calls_per_capture=40,ref_shape=list(ref.shape),mode=mode,arm=arm,gyro_alpha=.025,
        note='Actual control uses owned16-column role reference; original112 header height_reference remains base mean. Role log is authoritative for governed tracking/protection.')
    reset,wait=raw.reset,raw.step_wait
    def reset_all():
        result=reset();ref.zero_();log.zero_();raw._role_frozen.clear()
        for c in raw._role_chunks:c.clear()
        return result
    def step_wait():
        result=wait();frames=raw._complete_buffers[0].numpy();logs=log.numpy()
        for w in range(raw.num_envs):
            if w in raw._role_frozen:continue
            keep=frames[:,w,0]==1;selected=logs[keep,w].copy()
            assert np.all(selected[:,0]==1) and np.array_equal(selected[:,1],frames[keep,w,1])
            if len(selected):raw._role_chunks[w].append(selected)
            if result[2][w]:
                data=np.concatenate(raw._role_chunks[w]);check_log(data,mode)
                assert len(data)==result[3][w]['physical_steps']
                if directory is not None:
                    file=directory/f'case_role_{cases[w]["seed"]}.npz';assert not file.exists()
                    np.savez_compressed(file,trace=data,columns=np.array(COL))
                    result[3][w]['role_trace']=dict(path=file.name,sha256=rec.sha(file),rows=len(data))
                raw._role_frozen.add(w);raw._role_chunks[w].clear()
        return result
    raw.reset,raw.step_wait=reset_all,step_wait
    if directory is not None:rec.atomic_json(directory/'role_contract.json',raw._role_topology)
    return raw


def freeze():
    from test_reference_role_probe import check
    assert not (OUT/'source_contract.json').exists();check_control_copy()
    p=json.loads((OUT/'proposal.json').read_text());parent=rec.ROOT/p['parent_source_contract']
    assert rec.sha(parent)==p['parent_source_contract_sha256']
    sources=dict(json.loads(parent.read_text())['source_sha256'])
    assert all(rec.sha(rec.ROOT/n)==v for n,v in sources.items())
    check()
    for name in ('reference_role_control.py','reference_role_probe.py','test_reference_role_probe.py'):
        sources['wheelleg_warp/'+name]=rec.sha(rec.ROOT/'wheelleg_warp'/name)
    rec.atomic_json(OUT/'source_contract.json',dict(verified=True,proposal_sha256=rec.sha(OUT/'proposal.json'),source_sha256=sources,
        unit_sha256=rec.sha(OUT/'unit.json'),noise_file_sha256=p['noise_file_sha256'],role_columns=COL,new_evaluations=0,training_updates=0,
        status='Role source qualified, original branch no-op supports conditional80 reuse;160runner not yet frozen/run.'))
    print('SOURCE ADMITTED;0physics evaluations/learning;160queue not run',flush=True)


if __name__=='__main__':freeze()
