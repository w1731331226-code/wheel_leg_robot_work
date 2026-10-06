"""Isolated latest-gyro/measurement-noise intervention around frozen original calls."""
import inspect
import json
from pathlib import Path

import numpy as np
import warp as wp

import complete_contact_recorder as rec
from train_height_comparison import raw_env

OUT=rec.ROOT/'wheelleg_warp/results/paper_recovery_20261004/nom_yaw_filter_probe_v1'
D=rec.D
COL=['valid','step','Nom_measurement_index','Nom_noisy_input','Nom_filtered_output',
     'fresh_clean_measurement','noise_added','fresh_noisy_measurement']


def noise_table(cases,std):
    horizons=[]
    for case in cases:
        s=case['scenario'];assert s['terrain']=='legacy' and s['delay_ms']==0
        goal=s['center']+abs(s['offset'])/2+.75
        horizons.append(1.5+1.5*goal/abs(s['speed']))
    length=(int(np.ceil((max(horizons)+2)/.02))+2)*40+1
    table=np.zeros((len(cases),length),np.float32)
    if std:
        for i,c in enumerate(cases):
            rng=np.random.Generator(np.random.PCG64(17500000+c['seed']))
            table[i]=(rng.standard_normal(length)*std).astype(np.float32)
    return table


@wp.kernel
def initial_measurement(mask:wp.array[int],sensor:wp.array2d[float],ids:wp.array[int],
                        obs:wp.array2d[float],history:wp.array3d[float],
                        noise:wp.array2d[float],noisy:int):
    w=wp.tid()
    if mask[w]==0 or noisy==0:return
    g=ids[10]+2
    sensor[w,g]=sensor[w,g]+noise[w,0]
    obs[w,5]=sensor[w,g]
    history[w,0,5]=sensor[w,g]


@wp.kernel
def before_control(slot:int,sensor:wp.array2d[float],ids:wp.array[int],state:wp.array2d[D],
                   memory:wp.array2d[D],active:wp.array[int],latest:int,capacity:int,
                   monitor:wp.array3d[D],error:wp.array[int]):
    w=wp.tid()
    for j in range(8):monitor[slot,w,j]=D(0)
    if active[w]==0:return
    index=int(state[w,0])
    if index>=capacity:error[w]=1;return
    gyro=D(sensor[w,ids[10]+2])
    monitor[slot,w,0]=D(1);monitor[slot,w,1]=state[w,0]+D(1)
    monitor[slot,w,2]=D(wp.max(0,index-1));monitor[slot,w,3]=gyro
    # The original update then computes gyro + .025*(gyro-gyro) exactly.
    if latest:memory[w,8]=gyro


@wp.kernel
def filtered_measurement(slot:int,memory:wp.array2d[D],monitor:wp.array3d[D]):
    w=wp.tid()
    if monitor[slot,w,0]!=D(0):monitor[slot,w,4]=memory[w,8]


@wp.kernel
def fresh_measurement(slot:int,sensor:wp.array2d[float],ids:wp.array[int],state:wp.array2d[D],
                      active:wp.array[int],noise:wp.array2d[float],noisy:int,
                      monitor:wp.array3d[D],error:wp.array[int]):
    w=wp.tid()
    if active[w]==0:return
    index=int(state[w,0])
    if index>=noise.shape[1]:error[w]=1;return
    g=ids[10]+2;clean=sensor[w,g]
    if noisy:sensor[w,g]=clean+noise[w,index]
    monitor[slot,w,5]=D(clean);monitor[slot,w,6]=D(noise[w,index])
    monitor[slot,w,7]=D(sensor[w,g])


def instrument(cases,mode,condition,directory=None):
    assert condition['alpha'] in (.025,1.) and condition['noise_std_rad_s'] in (0.,.02)
    noisy=int(condition['noise_std_rad_s']!=0);latest=int(condition['alpha']==1.)
    samples=noise_table(cases,condition['noise_std_rad_s'])
    table=wp.array(samples);monitor=wp.zeros((40,len(cases),8),dtype=D)
    error=wp.zeros(len(cases),dtype=int);mask=wp.ones(len(cases),dtype=int)
    launch=wp.launch;calls=0;envstate=None;events=[]
    def intercepted(kernel,dim,inputs=None,**kwargs):
        nonlocal calls,envstate
        if kernel is rec.old.command_step:envstate=inputs[0]
        if kernel is rec.old.control_physical_nominal:
            slot=calls%40;calls+=1;assert envstate is not None
            sensor,active,memory,ids=inputs[2],inputs[5],inputs[6],inputs[7]
            launch(before_control,dim,[slot,sensor,ids,envstate,memory,active,latest,samples.shape[1],monitor,error])
            events.append(('control',slot,rec.old.signature(envstate),rec.old.signature(sensor),rec.old.signature(ids)))
            result=launch(kernel,dim,inputs,**kwargs)
            launch(filtered_measurement,dim,[slot,memory,monitor]);return result
        if kernel is rec.old.after:
            assert rec.old.signature(inputs[9])==rec.old.signature(envstate)
            slot=(calls-1)%40
            launch(fresh_measurement,dim,[slot,inputs[2],inputs[6],inputs[9],inputs[13],table,noisy,monitor,error])
            events.append(('after',slot,rec.old.signature(envstate),rec.old.signature(inputs[2]),rec.old.signature(inputs[6])))
        return launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
    wp.launch=intercepted
    try:raw=rec.instrument(raw_env,cases,mode,directory)
    finally:wp.launch=launch
    assert calls==80 and len(events)==160 and events[:80]==events[80:]
    assert raw._complete_topology['original_calls']==245
    deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
    assert samples.shape[1]>deadline*40
    raw._gyro_buffers=(table,monitor,error,mask)
    raw._gyro_chunks=[[] for _ in cases];raw._gyro_frozen=set()
    raw._gyro_topology=dict(verified=True,core=raw._complete_topology,control_interventions_per_capture=40,
        fresh_measurements_per_capture=40,identical_insertion_slots_and_inputs=True,
        table_shape=list(samples.shape),table_sha256=__import__('hashlib').sha256(samples.tobytes()).hexdigest(),
        numpy_version=np.__version__,noise_seeds=[17500000+c['seed'] for c in cases],condition=condition)
    reset,wait=raw.reset,raw.step_wait
    def initialize(which):
        mask.assign(np.asarray(which,np.int32))
        wp.launch(initial_measurement,raw.num_envs,[mask,raw.data.sensordata,raw.ids,raw.obs,raw.history,table,noisy])
    def reset_all():
        reset();monitor.zero_();error.zero_();raw._gyro_frozen.clear()
        for c in raw._gyro_chunks:c.clear()
        initialize(np.ones(raw.num_envs,np.int32))
        return raw.obs.numpy().copy()
    def step_wait():
        result=wait()
        if error.numpy().any():raise RuntimeError('Gyro noise-table capacity exceeded; no clipping/resume')
        frames=raw._complete_buffers[0].numpy();logs=monitor.numpy()
        for w in range(raw.num_envs):
            if w in raw._gyro_frozen:continue
            keep=frames[:,w,0]==1
            selected=logs[keep,w].copy()
            assert np.all(selected[:,0]==1) and np.array_equal(selected[:,1],frames[keep,w,1])
            if len(selected):raw._gyro_chunks[w].append(selected)
            if result[2][w]:
                data=np.concatenate(raw._gyro_chunks[w]);assert len(data)==result[3][w]['physical_steps']
                check_log(data,samples[w],condition['alpha'])
                if directory is not None:
                    file=directory/f'case_gyro_{cases[w]["seed"]}.npz';assert not file.exists()
                    np.savez_compressed(file,trace=data,columns=np.array(COL))
                    result[3][w]['gyro_trace']=dict(path=file.name,sha256=rec.sha(file),rows=len(data))
                raw._gyro_frozen.add(w);raw._gyro_chunks[w].clear()
        if result[2].any():
            initialize(result[2]);result[0][result[2]]=raw.obs.numpy()[result[2]]
        return result
    raw.reset,raw.step_wait=reset_all,step_wait
    if directory is not None:rec.atomic_json(directory/'gyro_contract.json',raw._gyro_topology)
    return raw


def check_log(data,noise,alpha):
    assert np.isfinite(data).all() and np.array_equal(data[:,1],np.arange(1,len(data)+1))
    np.testing.assert_array_equal(data[:,2],np.maximum(0,np.arange(len(data))-1))
    np.testing.assert_array_equal(data[:,6],noise[:len(data)])
    # Raw measurement rounding happens once in the existing float32 sensor array.
    np.testing.assert_array_equal(data[:,7],(data[:,5].astype(np.float32)+noise[:len(data)]).astype(np.float64))
    np.testing.assert_array_equal(data[1:,3],data[:-1,7])
    value=0.
    for input_value,actual in data[:,3:5]:
        value=input_value if alpha==1. else value+.025*(input_value-value)
        assert abs(actual-value)<1e-12


def freeze():
    from test_nom_yaw_filter_probe import check
    assert not (OUT/'source_contract.json').exists()
    p=json.loads((OUT/'proposal.json').read_text())
    parent=rec.ROOT/p['parent_source_contract'];assert rec.sha(parent)==p['parent_source_contract_sha256']
    sources=dict(json.loads(parent.read_text())['source_sha256'])
    assert all(rec.sha(rec.ROOT/n)==v for n,v in sources.items())
    check()
    samples=noise_table(p['cases'],.02)
    file=OUT/'noise_table.npz';assert not file.exists()
    np.savez_compressed(file,noise=samples,case_ids=np.array([c['seed'] for c in p['cases']]))
    for name in ('nom_yaw_filter_probe.py','test_nom_yaw_filter_probe.py'):
        sources['wheelleg_warp/'+name]=rec.sha(rec.ROOT/'wheelleg_warp'/name)
    rec.atomic_json(OUT/'source_contract.json',dict(verified=True,proposal_sha256=rec.sha(OUT/'proposal.json'),
        source_sha256=sources,unit_sha256=rec.sha(OUT/'unit.json'),noise_file_sha256=rec.sha(file),
        contact_API_sha256=json.loads(parent.read_text())['contact_api_sha256'],
        numpy_version=np.__version__,columns=COL,new_evaluations=0,training_updates=0,
        status='Source qualified; runner freeze and120 episode accounting still required. Clean original40 reuse qualified by no-op source/measurement identity, not fresh trajectory bitwise comparison.'))
    print('SOURCE ADMITTED;0 new evaluations/learning; sole120 queue not executed',flush=True)


if __name__=='__main__':freeze()
