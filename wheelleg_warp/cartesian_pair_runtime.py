"""Versioned dense adapter: original zero-target Nom, Cartesian residual, original guard."""
import json
import numpy as np
import warp as wp
import native.environment as native
import joint_state_guard as guard
import parking_withdrawal_probe as parking
from native.controller import D
from cartesian_pair_kernel import clear_native,reset_aux,observe_latent,deliver,TRACE_WIDTH
from dashboard.live_env import atomic_json
from review_yaw_sector import sha
from task_mode_recorder import signature

NOMINAL=parking.phase.roles.experimental.control_physical_nominal


def instrument(factory,cases,directory,mode):
    if mode not in (0,1) or not 1<=len(cases)<=20 or directory is None:
        raise ValueError('Dense Cartesian mode and bounded case list required')
    n=len(cases);wp.init();wp.set_device('cuda:0')
    latent=wp.zeros((n,3),dtype=D);shadow=wp.zeros((n,22),dtype=D);reference=wp.zeros((n,4),dtype=D)
    zero=wp.zeros((n,6),dtype=float);trace=wp.zeros((40,n,TRACE_WIDTH),dtype=D);mask=wp.ones(n,dtype=int)
    launch=wp.launch;counts=dict(reset=0,control=0,guard=0,prepare=0,finish=0,after=0)
    phase=None;last=None;command=None;nom_signatures=[]
    def intercept(kernel,dim,inputs=None,**kwargs):
        nonlocal phase,last,command
        if kernel is native.command_step:
            assert phase in (None,'after');command=inputs[2];phase='command'
        if kernel is parking.prepare:
            assert phase=='command' and inputs[2] is command
            args=list(inputs);args[5]=shadow;counts['prepare']+=1
            return launch(kernel,dim,args,**kwargs)
        if kernel is NOMINAL:
            assert phase=='command' and inputs[4] is command
            last=inputs;slot=counts['control']%40;counts['control']+=1
            launch(clear_native,dim,[inputs[5],inputs[6]])
            args=list(inputs);args[3]=zero
            result=launch(kernel,dim,args,**kwargs)
            nom_signatures.append(tuple(signature(x) for x in args))
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
        result=factory()
        assert counts['reset']>=1 and all(counts[k]==80 for k in ('control','guard','prepare','finish','after'))
        assert nom_signatures[:40]==nom_signatures[40:]
    except BaseException:
        np.savez_compressed(directory/'cartesian_construction_failure.npz',latent=latent.numpy(),shadow=shadow.numpy(),reference=reference.numpy(),trace=trace.numpy())
        atomic_json(directory/'cartesian_construction_counts.json',counts)
        raise
    finally:wp.launch=launch
    raw=result[0];assert raw.num_envs==n
    raw._cartesian_buffers=dict(latent=latent,shadow=shadow,reference=reference,zero=zero,trace=trace,mask=mask)
    raw._cartesian_topology=dict(mode=mode,counts=counts,nominal_signatures_equal_between_captures=True,
        zero_nom_target_signature=signature(zero),latent_shadow_signature=signature(shadow),original_nom_and_guard_kernels=True)
    atomic_json(directory/'cartesian_topology.json',raw._cartesian_topology)
    raw.observation_spec={**raw.observation_spec,'version':'cartesian_latent_memory_v1',
        'memory32_38':'current filtered [ax,-ax,az,-az,aw,-aw]; actual canonical/alpha diagnostic only'}
    raw._parking_contract={**raw._parking_contract,'filter':'paired latent3 shadow, +/-0.01 per0.5ms before Cartesian mapping',
        'only_controller_target_pointer_replaced':False,'canonical_slew_unchanged':False,
        'target_signatures_describe':'parked input to adapter; original Nom receives separate zero6', 'separate_latent_shadow':True}
    atomic_json(directory/'parking_contract.json',raw._parking_contract)
    reset,wait=raw.reset,raw.step_wait;chunks=[[] for _ in cases];times=[[] for _ in cases];frozen=set()
    raw._cartesian_chunks=chunks;raw._cartesian_times=times;raw._cartesian_terminal={}
    def reset_all():
        value=reset();mask.fill_(1);launch(reset_aux,n,[mask,raw.obs,latent,shadow,reference]);trace.zero_();frozen.clear()
        for part in chunks+times:part.clear()
        raw._cartesian_terminal.clear()
        return value
    def step_wait():
        values=trace.numpy();frames=raw._complete_buffers[0].numpy();done_before=raw.done.numpy()!=0
        for w in range(n):
            if w in frozen:continue
            np.testing.assert_array_equal(values[:,w,0],frames[:,w,0])
            keep=values[:,w,0]==1
            if keep.any():chunks[w].append(values[keep,w].copy());times[w].append(frames[keep,w,1:4].copy())
            if done_before[w]:
                raw._cartesian_terminal[w]=dict(latent=latent.numpy()[w].copy(),shadow=shadow.numpy()[w].copy(),reference=reference.numpy()[w].copy(),canonical=raw.k['state'].numpy()[w,16:22].copy())
        value=wait();np.testing.assert_array_equal(value[2],done_before)
        for w in np.flatnonzero(done_before):
            if w in frozen:continue
            combined=np.concatenate(chunks[w]);clock=np.concatenate(times[w]);row=value[3][w]
            assert len(combined)==row['physical_steps'] and not combined[:,33].any()
            np.testing.assert_array_equal(clock[:,0],np.arange(1,len(clock)+1))
            file=directory/f'cartesian_{cases[w]["seed"]}.npz'
            np.savez_compressed(file,trace=combined,clock=clock,clock_columns=np.array(['step','pre_s','post_s']))
            terminal=directory/f'cartesian_terminal_{cases[w]["seed"]}.npz';np.savez_compressed(terminal,**raw._cartesian_terminal[w])
            row['cartesian_trace']=dict(path=file.name,sha256=sha(file),rows=len(combined))
            row['cartesian_terminal']=dict(path=terminal.name,sha256=sha(terminal))
            row['cartesian_mode']='live' if mode==0 else 'reset';frozen.add(w)
        if done_before.any():mask.assign(done_before.astype(np.int32));launch(reset_aux,n,[mask,raw.obs,latent,shadow,reference])
        return value
    raw.reset,raw.step_wait=reset_all,step_wait
    return result


def preserve(raw,directory):
    if not hasattr(raw,'_cartesian_buffers'):return
    np.savez_compressed(directory/'cartesian_buffers.npz',**{k:v.numpy() for k,v in raw._cartesian_buffers.items()})
    for w,part in enumerate(raw._cartesian_chunks):
        if part:np.savez_compressed(directory/f'cartesian_prefix_{w}.npz',trace=np.concatenate(part),clock=np.concatenate(raw._cartesian_times[w]))
