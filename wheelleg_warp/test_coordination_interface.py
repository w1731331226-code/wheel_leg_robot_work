"""Static CPU/Warp shadow reference and shared-budget checks;no integration."""
import inspect
import json
import tempfile
from pathlib import Path
import numpy as np
import warp as wp
import run_parking_withdrawal as runner
import coordination_probe as probe
import coordinated_nominal_query as query
import coordination_nominal_budget as budget
from test_nom_yaw_filter_probe import control_args


def packet_hold(raw,condition):
    raw.reset();packet=raw._coord_packet.copy();packet[:,6]=packet[:,9]=.7;packet[:,20:22]=[72,14];raw._coord_packet=packet
    arrays=(raw.state,raw.k['state'],raw.targets,raw.data.qpos,raw.data.qvel,raw.command,raw.obs,raw.history)
    saved=[a.numpy().copy() for a in arrays]
    probe.prepare_packet(raw,condition)
    gamma=raw._coord_buffers[0].numpy().copy();damp=raw._coord_buffers[1].numpy().copy()
    if condition=='Cgamma':assert np.all((gamma>0)&(gamma<1))
    else:np.testing.assert_array_equal(gamma,1)
    if condition=='B0':assert not damp.any()
    else:assert np.all(damp[:,4]<0)
    for a,b in zip(arrays,saved):np.testing.assert_array_equal(a.numpy(),b)
    np.testing.assert_array_equal(raw._coord_buffers[0].numpy(),gamma)
    np.testing.assert_array_equal(raw._coord_buffers[1].numpy(),damp)


def lifecycle(case,condition):
    with tempfile.TemporaryDirectory() as tmp:
        directory=Path(tmp);raw=probe.instrument([case],'virtual6',condition,directory)
        try:
            raw.reset();gamma,damp,delta,total,previous,log=raw._coord_buffers
            ref,role=raw._role_buffers;table,gyro,error,_=raw._gyro_buffers;trace,_,count,*_=raw._complete_buffers
            for step in range(1,13):
                slot=step-1;state=raw.state.numpy();state[0,0]=step-1;raw.state.assign(state)
                command=.5 if 2<=step<12 else 0.;raw.command.fill_(command);probe.phase.prepare_step(raw,slot,'phase_support')
                if step==11:raw.nominal_correction.fill_(.1/6)
                args=control_args(raw);args[12]=ref
                wp.launch(probe.phase.roles.noise.before_control,1,[slot,args[2],args[7],raw.state,args[6],args[5],0,table.shape[1],gyro,error])
                update=int(slot%10==0)
                if condition=='B0':wp.copy(total,raw.nominal_correction)
                elif update:
                    wp.copy(previous,total);wp.launch(budget.compose,1,[previous,raw.nominal_correction,delta,damp,raw.active,total])
                wp.launch(probe.record,1,[slot,update,raw.state,raw.command,gamma,raw.active,raw.nominal_correction,delta,damp,total,raw.diag,raw.diag,log])
                args[-1]=total;wp.launch(probe.phase.roles.experimental.control_physical_nominal,1,args,block_dim=32)
                wp.launch(probe.phase.roles.finish,1,[slot,raw.diag,role]);wp.launch(probe.phase.roles.noise.filtered_measurement,1,[slot,raw.k['state'],gyro])
                wp.launch(probe.phase.roles.noise.fresh_measurement,1,[slot,raw.data.sensordata,raw.ids,raw.state,raw.active,table,0,gyro,error])
                if step==12:
                    frames=np.zeros(trace.shape);frames[:12,0,:4]=np.c_[np.ones(12),np.arange(1,13),np.arange(12)*.0005,np.arange(1,13)*.0005]
                    frames[:12,0,43]=[0]+[.5]*10+[0];trace.assign(frames);count.assign(np.array([[12,12]],float))
                    state[0,0]=12;raw.state.assign(state);raw.done.fill_(5)
                elif step==11:
                    partial=probe.preserve(raw,directory);assert partial['complete_worlds']==[]
            result=raw.step_wait();assert raw._coord_frozen=={0} and not raw._coord_chunks[0]
            file=directory/result[3][0]['coordination_trace']['path'];digest=probe.phase.rec.sha(file)
            with np.load(file,allow_pickle=False) as z:probe.check_log(z['trace'],condition);assert len(z['trace'])==12
            np.testing.assert_array_equal(gamma.numpy(),1)
            assert not total.numpy().any() and not damp.numpy().any()
            raw.done.fill_(5);raw.step_wait();assert probe.phase.rec.sha(file)==digest
            raw.reset();assert not log.numpy().any() and not raw._coord_frozen
            np.testing.assert_array_equal(raw.data.time.numpy(),0)
        finally:raw.close()


def source_copy():
    expected=inspect.getsource(probe.phase.roles.experimental.control_step.func)
    patches={'    cmd=D(command[w]);error=':'    cmd=D(command[w]);motion_cmd=cmd\n    if reference.shape[1]>16:motion_cmd=reference[w,16]\n    error=',
             'state[w,11]+(state[w,7]-cmd)*dt':'state[w,11]+(state[w,7]-motion_cmd)*dt',
             'ktha*(cmd-state[w,7])':'ktha*(motion_cmd-state[w,7])',
             'position,vx-cmd,pitch,pitch_rate':'position,vx-motion_cmd,pitch,pitch_rate'}
    for a,b in patches.items():assert expected.count(a)==1;expected=expected.replace(a,b)
    assert expected==inspect.getsource(query.control_step.func)
    assert inspect.getsource(query.control_physical_nominal.func)==inspect.getsource(probe.phase.roles.experimental.control_physical_nominal.func)


def shadow_query(raw,command,gamma,device):
    raw.reset();raw.command.fill_(command);probe.phase.prepare_step(raw,0,'phase_support')
    state=raw.k['state'].numpy();state[:,0]=2.;state[:,11]=.05;raw.k['state'].assign(state)
    args=control_args(raw);args[12]=raw._role_buffers[0]
    # Transfer the real model/controller query to CPU as an independent execution path.
    if device=='cpu':args=[wp.array(a.numpy(),dtype=a.dtype,device='cpu') if isinstance(a,wp.array) else a for a in args]
    arrays=[a for a in args if isinstance(a,wp.array)];saved=[a.numpy().copy() for a in arrays]
    wp.launch(probe.phase.roles.experimental.control_physical_nominal,raw.num_envs,args,device=device,block_dim=32)
    original=[args[i].numpy().copy() for i in (6,14,15)]
    for a,b in zip(arrays,saved):a.assign(b)
    shadow=wp.zeros((raw.num_envs,17),dtype=probe.D,device=device)
    effective=wp.array(raw.command.numpy()*gamma,dtype=probe.D,device=device)
    wp.launch(budget.reference_copy,raw.num_envs,[args[12],effective,shadow],device=device)
    altered=list(args);altered[12]=shadow
    wp.launch(query.control_physical_nominal,raw.num_envs,altered,device=device,block_dim=32)
    result=[args[i].numpy().copy() for i in (6,14,15)]
    if gamma==1:
        for a,b in zip(original,result):np.testing.assert_array_equal(a,b)
    else:
        # Public mode state remains unchanged even when motion reference becomes zero.
        np.testing.assert_array_equal(original[0][:,[0,9,10,12,13,14,15]],result[0][:,[0,9,10,12,13,14,15]])
        assert np.any(original[2][:,15:21]!=result[2][:,15:21])
    return result


def run():
    runner.verify();source_copy();budget.unit()
    p=json.loads((runner.main.OUT/'proposal.json').read_text());cases=[next(c for c in p['controlled'] if c['scenario']['stand_height_m']==h) for h in (.115,.16,.24,.30,.38)]
    count=0;worlds=0
    for condition in ('B0','Bomega','Cgamma'):
        raw=probe.instrument(cases,'virtual6',condition)
        try:
            assert raw.obs.shape[1]==38
            packet_hold(raw,condition)
            for command,gamma in ((0.,1.),(.8,1.),(-.8,1.),(.8,0.),(-.8,.4)):
                cpu=shadow_query(raw,command,gamma,'cpu');gpu=shadow_query(raw,command,gamma,'cuda')
                for a,b in zip(cpu,gpu):np.testing.assert_allclose(a,b,atol=2e-10,rtol=2e-10)
                count+=raw.num_envs*2
            # No source/query launch has integrated physics.
            np.testing.assert_array_equal(raw.data.time.numpy(),0)
            raw._coord_buffers[0].fill_(.2);raw.reset()
            np.testing.assert_array_equal(raw._coord_buffers[0].numpy(),1)
            assert not raw._coord_frozen and all(not c for c in raw._coord_chunks)
            worlds+=raw.num_envs
        finally:raw.close()
        lifecycle(cases[0],condition)
    runner.write(runner.OUT/'round218_interface_unit.json',dict(verified=True,static_worlds=worlds,nonintegrating_CPU_CUDA_queries=count,
        shadow_source_exact_four_substitutions=True,original_sources_unmodified=True,gamma1_same_device_exact_noop=True,
        shadow_velocity_tilt_integral_consistent=True,zero_motion_reference_does_not_trigger_public_parking_mode=True,
        CPU_CUDA_static_query_tolerance=dict(atol=2e-10,rtol=2e-10),one_Nom_pool1Nm_L1_point1_per5ms=True,private_reset_gamma1=True,
        source_sha256={f'wheelleg_warp/{n}':runner.sha(runner.ROOT/'wheelleg_warp'/n) for n in ('coordinated_nominal_query.py','coordination_nominal_budget.py','coordination_probe.py','test_coordination_interface.py')},
        parent_proposal_sha256=runner.sha(runner.OUT/'round217_coordination_proposal.json'),new_training_or_evaluations=0,
        synthetic_terminal_repeat_autoreset_and_explicit_reset_checked=True,packet20ms_hold_and_raw_input_unchanged=True,source_stream5ms_cadence_and_partial_saver_checked=True,
        limits='Source/static/synthetic admission only;492 runner not frozen/run. B0 actual replay comparability and original gates remain future scientific checks. CPU-GPU local query equality does not establish wholetrajectory backend equivalence.'))
    print('PASS218 component interface',worlds,'worlds',count,'CPU/CUDA staticqueries;492 not admitted',flush=True)


if __name__=='__main__':run()
