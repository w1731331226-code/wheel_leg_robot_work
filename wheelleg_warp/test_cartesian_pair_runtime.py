"""Check actual wrapper ordering with a non-executing launch spy; no robot construction."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import warp as wp
import cartesian_pair_runtime as cart
from native.controller import D
import native.environment as native
import joint_state_guard as guard
import parking_withdrawal_probe as parking

control_physical_nominal=cart.NOMINAL


def run():
    wp.init();wp.set_device('cuda:0');calls=[]
    def spy(kernel,dim,inputs=None,**kwargs):calls.append((kernel,inputs))
    with tempfile.TemporaryDirectory(prefix='cartesian-routing-') as td,patch.object(wp,'launch',spy):
        directory=Path(td);cases=[dict(seed=1)]
        q=wp.zeros((1,17));v=wp.zeros((1,16));sensor=wp.zeros((1,16));targets=wp.zeros((1,6));effective=wp.zeros((1,6))
        command=wp.zeros(1,dtype=D);active=wp.ones(1,dtype=int);state=wp.zeros((1,30),dtype=D);ids=wp.array(np.arange(18),dtype=int)
        other=wp.zeros((1,4),dtype=D);ctrl=wp.zeros((1,6));diag=wp.zeros((1,38),dtype=D);env_state=wp.zeros((1,39),dtype=D)
        obs=wp.zeros((1,38));obs0=wp.zeros((1,32));mask=wp.ones(1,dtype=int);log=wp.zeros((40,1,30),dtype=D)
        raw=SimpleNamespace(num_envs=1,observation_spec={'dimension':38},_parking_contract={},obs=obs,k={'state':state},
            reset=lambda:obs.numpy(),step_wait=lambda:(obs.numpy(),np.zeros(1),np.zeros(1,bool),[{}]))
        args=[q,v,sensor,effective,command,active,state,ids,other,other,other,other,other,other,ctrl,diag,0,0,other,other]
        def factory():
            reset=[mask]+[other]*11+[obs0]+[other]*5;wp.launch(native.reset_rows,1,reset)
            for k in range(80):
                wp.launch(native.command_step,1,[env_state,other,command,active])
                wp.launch(parking.prepare,1,[k%40,1,command,active,env_state,state,targets,mask,effective,log])
                wp.launch(control_physical_nominal,1,args)
                wp.launch(parking.finish,1,[k%40,state,diag,log])
                after=[other]*22;after[10]=state;after[11]=diag;after[16]=obs
                wp.launch(native.after,1,after)
            return (raw,)
        result=cart.instrument(lambda:guard.instrument(factory,cases,directory),cases,directory,0)
        assert result[0] is raw and wp.launch is spy
        buffers=raw._cartesian_buffers
        for kernel,inputs in calls:
            if kernel is control_physical_nominal:assert inputs[3] is buffers['zero'] and inputs[6] is state
            if kernel is parking.prepare:assert inputs[5] is buffers['shadow']
            if kernel is parking.finish:assert inputs[1] is buffers['shadow']
        important=[k for k,_ in calls if k in (control_physical_nominal,cart.deliver,guard.guard)]
        assert important==[control_physical_nominal,cart.deliver,guard.guard]*80
        def broken():raise RuntimeError('injected construction failure')
        try:cart.instrument(broken,cases,directory,1)
        except RuntimeError as error:assert str(error)=='injected construction failure'
        else:raise AssertionError('Construction failure swallowed')
        assert wp.launch is spy and (directory/'cartesian_construction_failure.npz').exists()
    print('PASS364 real guard wrapper delegates Nom->candidate->guard; shadow/zero pointer separation; restoration and construction failure preservation;0physics',flush=True)


if __name__=='__main__':run()
