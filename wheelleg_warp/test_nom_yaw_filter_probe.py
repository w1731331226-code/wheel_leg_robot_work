"""Source/measurement/capture/no-op admission without physics integration."""
import gc
import json
import numpy as np
import warp as wp

import nom_yaw_filter_probe as p


def control_args(raw):
    return [raw.data.qpos,raw.data.qvel,raw.data.sensordata,raw.targets,raw.command,raw.active,
        raw.k['state'],raw.ids,raw.k['heights'],raw.k['gains'],raw.k['feed'],raw.k['angles'],
        raw.k['reference'],raw.k['yaw'],raw.data.ctrl,raw.diag,int(raw.project_clipped_base),
        int(raw.grouped_residual)]+raw.control_extra


def after_args(raw):
    return [raw.data.qpos,raw.data.qvel,raw.data.sensordata,raw.data.qacc_warmstart,raw.data.time,
        raw.contact_flags,raw.ids,raw.param,raw.command,raw.state,raw.k['state'],raw.diag,
        raw.residual,raw.active,raw.done,raw.reward,raw.obs,raw.history,raw.stopped_q,raw.stopped_v,
        raw.stopped_w,raw.wheel_offsets]


def check():
    proposal=json.loads((p.OUT/'proposal.json').read_text());cases=proposal['cases']
    noise=p.noise_table(cases,.02)
    np.testing.assert_array_equal(noise,p.noise_table(cases,.02))
    np.testing.assert_array_equal(noise[::-1],p.noise_table(cases[::-1],.02))
    assert not p.noise_table(cases,0.).any()
    owners=[];same_original_outputs=None
    for condition in proposal['conditions']:
        raw=p.instrument(cases,'diff3',condition)
        try:
            obs=raw.reset();table,monitor,error,mask=raw._gyro_buffers
            n=raw.num_envs;ids=raw.ids.numpy();g=int(ids[10])+2
            samples=p.noise_table(cases,condition['noise_std_rad_s'])
            np.testing.assert_array_equal(raw.data.sensordata.numpy()[:,g],samples[:,0])
            np.testing.assert_array_equal(obs[:,5],samples[:,0])
            np.testing.assert_array_equal(raw.history.numpy()[:,0,5],samples[:,0])
            arrays=[a for a in control_args(raw) if isinstance(a,wp.array)]
            saved=[a.numpy().copy() for a in arrays]
            wp.launch(p.before_control,n,[0,raw.data.sensordata,raw.ids,raw.state,raw.k['state'],
                raw.active,int(condition['alpha']==1.),samples.shape[1],monitor,error])
            if condition['alpha']==.025:
                for a,b in zip(arrays,saved):np.testing.assert_array_equal(a.numpy(),b)
            wp.launch(raw.control_kernel,n,control_args(raw),block_dim=32)
            wp.launch(p.filtered_measurement,n,[0,raw.k['state'],monitor])
            expected=samples[:,0].astype(np.float64)*condition['alpha']
            np.testing.assert_allclose(raw.k['state'].numpy()[:,8],expected,atol=1e-12,rtol=0)
            if condition['label']=='original_clean':
                # Same original kernel/state/input, without probe, must be exactly identical.
                got=[raw.k['state'].numpy().copy(),raw.data.ctrl.numpy().copy(),raw.diag.numpy().copy()]
                raw.reset();wp.launch(raw.control_kernel,n,control_args(raw),block_dim=32)
                for a,b in zip(got,[raw.k['state'].numpy(),raw.data.ctrl.numpy(),raw.diag.numpy()]):np.testing.assert_array_equal(a,b)
                same_original_outputs=True
                raw.reset()
                wp.launch(p.before_control,n,[0,raw.data.sensordata,raw.ids,raw.state,raw.k['state'],raw.active,0,samples.shape[1],monitor,error])
                wp.launch(raw.control_kernel,n,control_args(raw),block_dim=32)
                wp.launch(p.filtered_measurement,n,[0,raw.k['state'],monitor])
            # CPU test fixture stands in for step's fresh forward sensor evaluation.
            sensor=raw.data.sensordata.numpy();sensor[:,g]=0.;raw.data.sensordata.assign(sensor)
            before=sensor.copy()
            wp.launch(p.fresh_measurement,n,[0,raw.data.sensordata,raw.ids,raw.state,raw.active,
                table,int(condition['noise_std_rad_s']!=0),monitor,error])
            current=raw.data.sensordata.numpy();np.testing.assert_array_equal(current[:,g],samples[:,0])
            np.testing.assert_array_equal(np.delete(current,g,axis=1),np.delete(before,g,axis=1))
            wp.launch(p.rec.old.after,n,after_args(raw),block_dim=32)
            np.testing.assert_array_equal(raw.obs.numpy()[:,5],current[:,g])
            # Next Nom sees the very same already noisy preceding sample; never add again.
            wp.launch(p.before_control,n,[1,raw.data.sensordata,raw.ids,raw.state,raw.k['state'],raw.active,
                int(condition['alpha']==1.),samples.shape[1],monitor,error])
            np.testing.assert_array_equal(raw.data.sensordata.numpy(),current)
            np.testing.assert_array_equal(monitor.numpy()[1,:,3],current[:,g])
            wp.launch(raw.control_kernel,n,control_args(raw),block_dim=32)
            wp.launch(p.filtered_measurement,n,[1,raw.k['state'],monitor])
            sensor=raw.data.sensordata.numpy();sensor[:,g]=.1;raw.data.sensordata.assign(sensor)
            wp.launch(p.fresh_measurement,n,[1,raw.data.sensordata,raw.ids,raw.state,raw.active,
                table,int(condition['noise_std_rad_s']!=0),monitor,error])
            for w in range(n):p.check_log(monitor.numpy()[:2,w],samples[w],condition['alpha'])
            # Capacity and inactivity are explicit, never silently clamp an out-of-range sample.
            state=raw.state.numpy();state[:,0]=samples.shape[1];raw.state.assign(state)
            wp.launch(p.fresh_measurement,n,[2,raw.data.sensordata,raw.ids,raw.state,raw.active,table,1,monitor,error])
            assert np.all(error.numpy()==1)
            raw.reset();active=raw.active.numpy();active[0]=0;raw.active.assign(active)
            wp.launch(p.before_control,n,[0,raw.data.sensordata,raw.ids,raw.state,raw.k['state'],raw.active,1,samples.shape[1],monitor,error])
            assert monitor.numpy()[0,0,0]==0
            raw.reset();assert not monitor.numpy().any() and not error.numpy().any()
            assert not raw.state.numpy()[:,0].any() and not raw.data.time.numpy().any()
            assert not raw._gyro_chunks[0] and not raw._gyro_frozen
            gc.collect();assert raw._gyro_buffers[0] is table
            owners.append(dict(**raw._gyro_topology,original_noop_checked=condition['label']=='original_clean'))
        finally:raw.close()
    assert same_original_outputs
    p.rec.atomic_json(p.OUT/'unit.json',dict(verified=True,owners=owners,original_noop_inputs_and_outputs_exact=True,
        both_alpha_original_kernel_output_matches=True,initial_and_delayed_packet_same_noisy_measurement=True,
        no_double_noise_or_other_channel_change=True,measurement_carry_index_checked=True,
        PCG64_case_seed_order_independent_and_repeatable=True,capacity_overflow_and_inactive_checked=True,
        reset_and_buffer_ownership_checked=True,new_physics_evaluations=0,training_updates=0,
        limits='Construct/capture, original-control computations and synthetic fresh sensors/after only; no integration/live120 data. Original packet delay0 for these20 registered cases; no arbitrary-delay noise certificate.'))
    print('PASS4x20 source/no-op/gyro/noise/packet/owners/reset;0physics evaluation',flush=True)


if __name__=='__main__':check()
