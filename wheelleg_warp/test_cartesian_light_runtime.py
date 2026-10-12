"""Real light parking/phase/guard wrappers with a non-executing launch spy."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import warp as wp
import native.environment as native
import joint_state_guard as guard
import light_parking_request as parking
import cartesian_light_runtime as cart
from native.controller import D


def run():
    wp.init();wp.set_device('cuda:0');events=[];n=3
    def spy(kernel,dim,inputs=None,**kwargs):events.append((kernel,inputs))
    q=wp.zeros((n,17));v=wp.zeros((n,16));sensor=wp.zeros((n,16));target=wp.zeros((n,6));command=wp.zeros(n,dtype=D)
    active=wp.ones(n,dtype=int);state=wp.zeros((n,30),dtype=D);envstate=wp.zeros((n,39),dtype=D);ids=wp.array(np.arange(18),dtype=int)
    other=wp.zeros((n,4),dtype=D);reference=wp.zeros((n,10),dtype=D);ctrl=wp.zeros((n,6));diag=wp.zeros((n,38),dtype=D)
    obs=wp.zeros((n,38));obs0=wp.zeros((n,32));mask=wp.ones(n,dtype=int)
    failure=[False]
    def wait():
        if failure[0]:
            raw.obs.fill_(99);raw._execution_buffers[1].fill_(88);raw._execution_previous_sensor_steps.fill(77)
            raise RuntimeError('wait failure')
        return obs.numpy(),np.zeros(n),np.zeros(n,bool),[{} for _ in range(n)]
    raw=SimpleNamespace(num_envs=n,observation_spec={'dimension':38},obs=obs,k={'state':state},
        nominal_correction_enabled=True,shared_reference_enabled=True,actuator_gain_upper=np.array(native.PUBLIC_ACTUATOR_GAIN_UPPER),
        reset=lambda:obs.numpy(),step_wait=wait,done=wp.zeros(n,dtype=int),state=envstate,param=wp.zeros((n,5),dtype=D),
        _execution_buffers=(wp.zeros((n,6)),wp.zeros((n,2,6)),wp.zeros((n,2),dtype=int)),
        _execution_previous_sensor_steps=np.arange(n,dtype=np.int64))
    def factory(cases,mode):
        raw.control_kernel=native.control_physical_nominal
        wp.launch(native.reset_rows,n,[mask]+[other]*11+[obs0]+[other]*5)
        for _ in range(40):
            wp.launch(native.command_step,n,[envstate,other,command,active])
            wp.launch(native.control_physical_nominal,n,[q,v,sensor,target,command,active,state,ids,other,other,other,other,reference,other,ctrl,diag,0,0,other,other])
            args=[other]*22;args[10]=state;args[11]=diag;args[16]=obs;wp.launch(native.after,n,args)
        return raw
    cases=[dict(seed=i) for i in range(n)]
    with patch.object(wp,'launch',spy):
        result=cart.instrument(lambda:guard.instrument(lambda:(parking.instrument(factory,cases,'virtual6'),),cases,None,dense=False),cases,0)
        assert result[0] is raw and wp.launch is spy
        expected=[cart.NOMINAL,cart.deliver,guard.guard]*40
        assert [k for k,_ in events if k in (cart.NOMINAL,cart.deliver,guard.guard)]==expected
        for kernel,args in events:
            if kernel is cart.NOMINAL:assert args[3] is raw._cartesian_buffers['zero']
            if kernel is cart.parking.prepare:assert args[5] is raw._cartesian_buffers['shadow']
            if kernel is cart.parking.finish:assert args[1] is raw._cartesian_buffers['shadow']
        def broken():raise RuntimeError('constructor failure')
        try:cart.instrument(broken,cases,1)
        except RuntimeError:pass
        else:raise AssertionError('Missing failure')
        assert wp.launch is spy
        raw.step_wait();saved=raw._cartesian_last
        for key,buffer in [('guard',raw._joint_guard_buffers[1]),('parking',raw._light_parking_buffers[2]),
                           ('command_prefix',raw._execution_buffers[1]),('command_stamp',raw._execution_buffers[2])]:
            np.testing.assert_array_equal(saved[key],buffer.numpy())
        failure[0]=True
        try:raw.step_wait()
        except RuntimeError as error:assert str(error)=='wait failure'
        else:raise AssertionError('Missing wait failure')
        assert raw._cartesian_last is not saved
        np.testing.assert_array_equal(raw._cartesian_last['terminal_packet'],np.zeros((n,38)))
        np.testing.assert_array_equal(raw._cartesian_last['command_prefix'],np.zeros((n,2,6)))
        np.testing.assert_array_equal(raw._cartesian_last['previous_sensor_steps'],np.arange(n))
    print('PASS368 actual light parking/phase/guard wrappers:40 Nom-candidate-guard sequences,shadow/zero pointers,failure restoration and pre-wait snapshot preservation;0physics',flush=True)


if __name__=='__main__':run()
