"""Single-capture Cartesian adapter for the existing light curriculum pipeline."""
import numpy as np
import warp as wp
import native.environment as native
import joint_state_guard as guard
import parking_withdrawal_probe as parking
from cartesian_pair_kernel import D,clear_native,reset_aux,observe_latent,TRACE_WIDTH
from cartesian_pair_kernel_v2 import deliver_accepted_nominal as deliver
from cartesian_pair_runtime import NOMINAL
from run_cartesian_repair import require_accepted_nominal


def instrument(factory,cases,mode):
    if mode not in (0,1) or not 1<=len(cases)<=100:raise ValueError('Light Cartesian mode and1..100 worlds required')
    n=len(cases);wp.init();wp.set_device('cuda:0')
    latent=wp.zeros((n,3),dtype=D);shadow=wp.zeros((n,22),dtype=D);reference=wp.zeros((n,4),dtype=D)
    zero=wp.zeros((n,6));trace=wp.zeros((40,n,TRACE_WIDTH),dtype=D);mask=wp.ones(n,dtype=int)
    launch=wp.launch;counts=dict(reset=0,control=0,guard=0,prepare=0,finish=0,after=0);phase=None;last=None;command=None
    def intercept(kernel,dim,inputs=None,**kwargs):
        nonlocal phase,last,command
        if kernel is native.command_step:
            assert phase in (None,'after');command=inputs[2];phase='command'
        if kernel is parking.prepare:
            assert phase=='command' and inputs[2] is command;args=list(inputs);args[5]=shadow;counts['prepare']+=1
            return launch(kernel,dim,args,**kwargs)
        if kernel is NOMINAL:
            assert phase=='command' and inputs[4] is command;last=inputs;slot=counts['control'];counts['control']+=1
            assert slot<40
            launch(clear_native,dim,[inputs[5],inputs[6]]);args=list(inputs);args[3]=zero
            result=launch(kernel,dim,args,**kwargs)
            launch(deliver,dim,[slot,mode,inputs[0],inputs[1],inputs[7],inputs[3],inputs[5],latent,reference,inputs[6],shadow,inputs[14],inputs[15],trace])
            phase='candidate';return result
        if kernel is guard.guard:
            assert phase=='candidate' and all(inputs[i] is last[j] for i,j in [(1,0),(2,1),(3,5),(4,7),(5,14),(6,15)])
            result=launch(kernel,dim,inputs,**kwargs);counts['guard']+=1;phase='guard';return result
        if kernel is parking.finish:
            assert phase=='guard';args=list(inputs);args[1]=shadow;counts['finish']+=1
            return launch(kernel,dim,args,**kwargs)
        result=launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
        if kernel is native.reset_rows:
            launch(reset_aux,dim,[inputs[0],inputs[12],latent,shadow,reference]);counts['reset']+=1
        if kernel is native.after:
            assert phase=='guard' and inputs[10] is last[6] and inputs[11] is last[15]
            launch(observe_latent,dim,[latent,inputs[16]]);counts['after']+=1;phase='after'
        return result
    wp.launch=intercept
    try:
        result=factory();raw=result[0]
        assert counts['reset']>=1 and all(counts[k]==40 for k in ('control','guard','prepare','finish','after'))
        require_accepted_nominal(raw)
    finally:wp.launch=launch
    raw._cartesian_buffers=dict(latent=latent,shadow=shadow,reference=reference,zero=zero,trace=trace,mask=mask)
    raw._cartesian_topology=dict(mode=mode,counts=counts,worlds=n,dense=False,accepted_nominal_mode=1,
        candidate_kernel='cartesian_pair_kernel_v2.deliver_accepted_nominal')
    raw.observation_spec={**raw.observation_spec,'version':'cartesian_latent_memory_v1',
        'memory32_38':'current filtered [ax,-ax,az,-az,aw,-aw]; actual canonical diagnostic only'}
    reset,wait=raw.reset,raw.step_wait;raw._cartesian_last=None
    def reset_all():
        value=reset();mask.fill_(1);launch(reset_aux,n,[mask,raw.obs,latent,shadow,reference]);trace.zero_();raw._cartesian_last=None
        return value
    def step_wait():
        before_done=raw.done.numpy()!=0;values=trace.numpy();valid=values[:,:,0]==1
        assert not values[:,:,33][valid].any(),'Cartesian control-stage error'
        saved=dict(trace=values,latent=latent.numpy(),reference=reference.numpy(),shadow=shadow.numpy(),
            terminal_packet=raw.obs.numpy(),state=raw.state.numpy(),done=before_done.copy(),
            guard=raw._joint_guard_buffers[1].numpy(),parking=raw._light_parking_buffers[2].numpy(),
            guard_memory=raw._joint_guard_buffers[0].numpy(),parking_seen=raw._light_parking_buffers[0].numpy(),
            parking_effective=raw._light_parking_buffers[1].numpy(),phase_reference=raw._light_phase_buffers[0].numpy(),
            phase_role=raw._light_phase_buffers[1].numpy(),phase_monitor=raw._light_phase_buffers[2].numpy(),
            command_total=raw._execution_buffers[0].numpy(),
            command_prefix=raw._execution_buffers[1].numpy(),command_stamp=raw._execution_buffers[2].numpy(),
            previous_sensor_steps=raw._execution_previous_sensor_steps.copy(),delay_steps=raw.param.numpy()[:,4].copy())
        raw._cartesian_last=saved
        value=wait();np.testing.assert_array_equal(value[2],before_done)
        for w in np.flatnonzero(before_done):np.testing.assert_array_equal(value[3][w]['terminal_observation'],saved['terminal_packet'][w])
        if before_done.any():mask.assign(before_done.astype(np.int32));launch(reset_aux,n,[mask,raw.obs,latent,shadow,reference])
        return value
    raw.reset,raw.step_wait=reset_all,step_wait
    return result
