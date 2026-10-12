"""CPU/mock launch routing and ownership checks; no CUDA or robot construction."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

import cartesian_ppo_recorder as recorder


class Buffer:
    def __init__(self,shape,dtype=None):
        self.shape=tuple(shape);self.dtype=dtype;self.ptr=id(self);self.values=np.zeros(self.shape)
    def numpy(self):return self.values.copy()
    def zero_(self):self.values.fill(0)
    def fill_(self,value):self.values.fill(value)


def run():
    n=3;events=[];wp=recorder.wp;cart=recorder.cart
    q=Buffer((n,17));v=Buffer((n,16));sensor=Buffer((n,28));active=Buffer((n,));ids=Buffer((19,))
    memory=Buffer((n,30));ctrl=Buffer((n,6));diag=Buffer((n,38));other=Buffer((n,4))
    effective=Buffer((n,6));private=[Buffer((n,6)) for _ in range(9)]
    raw=SimpleNamespace(num_envs=n,data=SimpleNamespace(qpos=q,qvel=v,sensordata=sensor,ctrl=ctrl),
        active=active,ids=ids,diag=diag,k={'state':memory,'reference':other},observation_spec={},
        nominal_correction_enabled=True,shared_reference_enabled=True,actuator_gain_upper=np.array([1,1,1,1,1.05,1.05]),
        reset=lambda:np.zeros((n,38)),step_wait=lambda:None,obs=Buffer((n,38)),
        _joint_guard_buffers=(private[0],other,other),_light_parking_buffers=(private[1],effective,other),
        _light_phase_buffers=tuple(private[3:6]),_execution_buffers=tuple(private[6:9]),
        _execution_previous_sensor_steps=np.arange(n),param=other,q0=q,obs0=other)
    def launch(kernel,dim,inputs=None,**kwargs):events.append((kernel,inputs))
    def factory(bad=False):
        raw.control_kernel=cart.NOMINAL
        wp.launch(cart.native.reset_rows,n,[other]+[other]*11+[other]+[other]*5)
        for slot in range(40):
            wp.launch(cart.native.command_step,n,[other,other,other,active])
            wp.launch(cart.parking.prepare,n,[slot,1,other,active,other,memory,other,other,effective,other])
            wp.launch(cart.NOMINAL,n,[q,v,sensor,effective,other,active,memory,ids]+[other]*6+[ctrl,diag,0,0,other,other])
            wp.launch(recorder.guard.guard,n,[slot,q,v,active,ids,ctrl if not bad else other,diag,other,other,other])
            wp.launch(cart.parking.finish,n,[slot,memory,diag,other])
            args=[other]*22;args[10]=memory;args[11]=diag;wp.launch(cart.native.after,n,args)
        return (raw,)
    with patch.object(wp,'init'),patch.object(wp,'set_device'),patch.object(wp,'zeros',lambda shape,dtype=None:Buffer(shape,dtype)),\
         patch.object(wp,'ones',lambda shape,dtype=None:Buffer((shape,) if isinstance(shape,int) else shape,dtype)),patch.object(wp,'launch',launch):
        result=recorder.instrument(factory,[{}]*n,0);assert result[0] is raw and wp.launch is launch
        expected=[cart.NOMINAL,recorder.record,cart.deliver,recorder.guard.guard]*40
        assert [k for k,_ in events if k in (cart.NOMINAL,recorder.record,cart.deliver,recorder.guard.guard)]==expected
        assert recorder.signature(raw._cartesian_buffers['zero'])==raw._cartesian_ppo_topology['zero_target']
        assert len(raw._cartesian_ppo_topology['slots'])==40
        raw._cartesian_ppo_buffers['nominal'].fill_(2);raw.reset();assert not raw._cartesian_ppo_buffers['nominal'].numpy().any()
        try:recorder.instrument(lambda:factory(True),[{}]*n,1)
        except AssertionError:pass
        else:raise AssertionError('Foreign guard control pointer accepted')
        assert wp.launch is launch
        def broken():raise RuntimeError('factory failed')
        try:recorder.instrument(broken,[{}]*n,0)
        except RuntimeError:pass
        else:raise AssertionError('Constructor failure hidden')
        assert wp.launch is launch
    values=recorder.post_private(raw);assert len(values)==10
    values['guard_memory'].fill(9);assert not private[0].numpy().any()
    bank=recorder.bank_actual(raw);assert set(bank)=={'param','q0','obs0','native_reference'}
    bank['q0'].fill(8);assert not q.numpy().any()
    print('PASS376 CPU/mock: actual Nom-zero/record/candidate/guard routing,ownership/restoration/reset and independent post/bank snapshots;0CUDA/physics/FD/learning')


if __name__=='__main__':run()
