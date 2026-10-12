"""Read-only kernels and real light wrapper routing; never constructs a robot."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import warp as wp
import native.environment as native
import joint_state_guard as guard
import light_parking_request as parking
import cartesian_light_runtime as cart
import cartesian_zero_input_trace as trace
from native.controller import D


def kernels():
    n=3;q=wp.array(np.arange(n*17).reshape(n,17).astype(np.float32));v=wp.array(np.arange(n*16).reshape(n,16).astype(np.float32))
    sensor=wp.array(np.arange(n*28).reshape(n,28).astype(np.float32));memory=wp.array(np.arange(n*30).reshape(n,30).astype(float),dtype=D)
    state=wp.zeros((n,39),dtype=D);ids=wp.array(np.arange(18),dtype=int);active=wp.array([1,0,1],dtype=int)
    ctrl=wp.ones((n,6));diag=wp.ones((n,38),dtype=D)
    before=[b.numpy().copy() for b in (q,v,sensor,memory,state,active,ctrl,diag)]
    pre=wp.zeros((40,n,66),dtype=D);edges=wp.zeros((40,n,6,10),dtype=D);controls=wp.zeros((40,n,2,44),dtype=D)
    wp.launch(trace.pre,n,[0,q,v,sensor,memory,state,ids,active,pre])
    wp.launch(trace.boundary,n,[0,2,memory,sensor,ids,active,edges]);wp.launch(trace.control,n,[0,1,ctrl,diag,controls])
    p=pre.numpy()[0];np.testing.assert_array_equal(p[:,5:22],q.numpy());np.testing.assert_array_equal(p[:,22:38],v.numpy())
    np.testing.assert_array_equal(p[:,38:],sensor.numpy());np.testing.assert_array_equal(p[:,3],memory.numpy()[:,5])
    np.testing.assert_array_equal(p[:,4],sensor.numpy()[:,10]);np.testing.assert_array_equal(edges.numpy()[0,:,2,2],memory.numpy()[:,5])
    np.testing.assert_array_equal(controls.numpy()[0,:,1,:6],ctrl.numpy());np.testing.assert_array_equal(controls.numpy()[0,:,1,6:],diag.numpy())
    for old,b in zip(before,(q,v,sensor,memory,state,active,ctrl,diag)):np.testing.assert_array_equal(old,b.numpy())


def routing(cartesian):
    n=3;events=[]
    q=wp.zeros((n,17));v=wp.zeros((n,16));sensor=wp.zeros((n,28));target=wp.zeros((n,6));command=wp.zeros(n,dtype=D)
    active=wp.ones(n,dtype=int);state=wp.zeros((n,30),dtype=D);envstate=wp.zeros((n,39),dtype=D);ids=wp.array(np.arange(18),dtype=int)
    other=wp.zeros((n,4),dtype=D);reference=wp.zeros((n,10),dtype=D);ctrl=wp.zeros((n,6));diag=wp.zeros((n,38),dtype=D)
    obs=wp.zeros((n,38));obs0=wp.zeros((n,32));mask=wp.ones(n,dtype=int)
    raw=SimpleNamespace(num_envs=n,observation_spec={'dimension':38},obs=obs,state=envstate,done=wp.zeros(n,dtype=int),active=active,ids=ids,diag=diag,
        data=SimpleNamespace(qpos=q,qvel=v,sensordata=sensor,ctrl=ctrl),k={'state':state},stopped_q=q,stopped_v=v,
        nominal_correction_enabled=True,shared_reference_enabled=True,actuator_gain_upper=np.array(native.PUBLIC_ACTUATOR_GAIN_UPPER),
        reset=lambda:obs.numpy(),step_wait=lambda:(obs.numpy(),np.zeros(n),np.zeros(n,bool),[{} for _ in range(n)]))
    def factory(cases,mode):
        raw.control_kernel=native.control_physical_nominal
        wp.launch(native.reset_rows,n,[mask]+[other]*11+[obs0]+[other]*5)
        for slot in range(40):
            wp.launch(native.command_step,n,[envstate,other,command,active,q,v])
            wp.launch(native.control_physical_nominal,n,[q,v,sensor,target,command,active,state,ids,other,other,other,other,reference,other,ctrl,diag,0,0,other,other])
            args=[other]*22;args[10]=state;args[11]=diag;args[16]=obs;wp.launch(native.after,n,args)
        return raw
    def spy(kernel,dim,inputs=None,**kwargs):events.append((kernel,inputs))
    cases=[dict(seed=i) for i in range(n)]
    with patch.object(wp,'launch',spy):
        guarded=lambda:guard.instrument(lambda:(parking.instrument(factory,cases,'virtual6'),),cases,None,dense=False)
        build=lambda:cart.instrument(guarded,cases,0)[0] if cartesian else guarded()[0]
        result=trace.instrument(build,cases,cartesian);assert result is raw and wp.launch is spy
        assert len([k for k,_ in events if k in (trace.pre,trace.boundary,trace.control)])==40*9
        for slot in range(40):
            marks=[a[1] for k,a in events if k is trace.boundary and a[0]==slot];assert marks==list(range(6))
        observer_events=[(k.__name__,a[0],a[1] if k is not trace.pre else None) for k,a in events if k in (trace.pre,trace.boundary,trace.control)]
        def broken():raise RuntimeError('constructor failure')
        try:trace.instrument(broken,cases,cartesian)
        except RuntimeError:pass
        else:raise AssertionError('Missing constructor failure')
        assert wp.launch is spy
    return observer_events


def run():
    wp.init();wp.set_device('cuda:0');kernels();assert routing(False)==routing(True)
    import run_cartesian_zero_input as runner
    from run_cartesian_zero_input import Budget,secondary,blob_hashes,classify
    p=dict(worlds=10,actual_graph_world_step_budget=400,constructor_FD_budget=1);b=Budget(p);calls=[]
    raw=SimpleNamespace(graph=object(),num_envs=10,data=SimpleNamespace(qpos=SimpleNamespace(shape=(10,17))));b.bind(raw)
    for graph in (object(),):
        try:b.capture(lambda x:calls.append('bad'),graph)
        except RuntimeError:pass
        else:raise AssertionError('Unknown graph accepted')
    b.capture(lambda x:calls.append('good'),raw.graph)
    try:b.capture(lambda x:calls.append('bad'),raw.graph)
    except RuntimeError:pass
    else:raise AssertionError('Repeated graph accepted')
    raw2=SimpleNamespace(graph=object(),num_envs=10,data=raw.data);b.bind(raw2)
    try:b.capture(lambda x:calls.append('bad'),raw2.graph)
    except RuntimeError:pass
    else:raise AssertionError('Total graph cap exceeded')
    changed=Budget(p);changed.bind(raw2);raw2.graph=object()
    try:changed.capture(lambda x:calls.append('bad'),raw2.graph)
    except RuntimeError:pass
    else:raise AssertionError('Mutable raw.graph changed admitted identity')
    b.fd(lambda *x:calls.append('FD'),None,None,1e-6)
    try:b.fd(lambda *x:calls.append('bad'),None,None,1e-6)
    except RuntimeError:pass
    else:raise AssertionError('FD overflow accepted')
    assert calls==['good','FD'] and b.steps==400
    errors=[]
    def broken_write():raise OSError('secondary write')
    try:
        try:raise RuntimeError('primary')
        except BaseException:
            secondary(errors,'write',broken_write);raise
    except RuntimeError as error:assert str(error)=='primary' and errors[0]['error']=="OSError('secondary write')"
    writes=[];saved_errors=[]
    dummy=SimpleNamespace(_zero_input_buffers={},_cartesian_buffers={},_zero_input_terminal={'state':np.zeros(1)})
    def write(path,**values):
        writes.append(path.name)
        if path.name=='preserved_common.npz':raise OSError('common write failed')
    from pathlib import Path
    with patch.object(runner,'snapshot',lambda *a:{}),patch.object(runner.np,'savez_compressed',write):
        runner.save_buffers(dummy,None,Path('mock'),saved_errors)
    assert writes==['preserved_common.npz','preserved_observer.npz','preserved_cartesian.npz','preserved_terminal.npz'] and len(saved_errors)==1
    import hashlib
    value=b'abc\n';payload=b'0'*40+b' blob 4\n'+value+b'\n'
    assert blob_hashes(payload,['one'])=={'one':hashlib.sha256(value).hexdigest()}
    try:blob_hashes(b'HEAD:one missing\n',['one'])
    except ValueError:pass
    else:raise AssertionError('Uncommitted source accepted')
    original=dict(pre=np.zeros((40,3,66)),boundaries=np.zeros((40,3,6,10)))
    changed={k:v.copy() for k,v in original.items()};changed['pre'][7,1,4]=1
    assert classify(original,changed)['status']=='input_chain_divergence'
    changed={k:v.copy() for k,v in original.items()};changed['boundaries'][2,1,3,2]=1
    assert classify(original,changed)['status']=='Nom_execution_context_error'
    changed={k:v.copy() for k,v in original.items()};changed['boundaries'][3,1,5,2]=1
    assert classify(original,changed)['status']=='Cart_boundary_memory5_error'
    assert classify(original,original)['status']=='inconclusive_not_reproduced_under_instrumentation'
    print('PASS371 read-only CUDA records,identical9observer calls/slot on real light wrappers,restoration,graph/FD/HEAD caps,primary preservation and diagnosis fixtures;0physics/learning',flush=True)


if __name__=='__main__':run()
