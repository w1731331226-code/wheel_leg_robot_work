"""All declared cases compile/reset; original no-op and delayed packet without step."""
import gc
import json
import numpy as np
import warp as wp
import floor_broad_adapter as p
from test_nom_yaw_filter_probe import control_args,after_args


def check():
    proposal=json.loads((p.OUT/'proposal.json').read_text())
    cases=proposal['panels']['regular']+proposal['panels']['controlled_remaining']+proposal['panels']['legacy']
    assert len(cases)==144
    groups=p.batches(cases)
    assert sorted(i for group in groups for i,c in group)==list(range(len(cases)))
    assert all(len(g)<=20 and len({c['scenario']['solver_iterations'] for i,c in g})==1 for g in groups)
    try:p.zero_table(cases,.02)
    except ValueError:pass
    else:raise AssertionError('Broad noise falsely accepted')
    owners=[];empty=0;delay_checks=0
    for group in groups:
        selected=[c for i,c in group]
        for arm in ('original','floor_only'):
            raw=p.instrument(selected,'diff3',arm)
            try:
                raw.reset();contract=raw._broad_contract;empty+=len(contract['no_target_worlds'])
                assert not raw._gyro_buffers[0].numpy().any()
                args=control_args(raw);ref,log=raw._role_buffers;mode=0 if arm=='original' else 1
                arrays=[a for a in args if isinstance(a,wp.array)];saved=[a.numpy().copy() for a in arrays]
                wp.launch(p.roles.prepare,raw.num_envs,[0,mode,args[12],args[0],args[2],args[7],args[6],raw.state,args[5],ref,log])
                for a,b in zip(arrays,saved):np.testing.assert_array_equal(a.numpy(),b)
                altered=list(args);altered[12]=ref
                wp.launch(p.roles.experimental.control_physical_nominal,raw.num_envs,altered,block_dim=32)
                wp.launch(p.roles.finish,raw.num_envs,[0,raw.diag,log])
                output=[raw.k['state'].numpy().copy(),raw.diag.numpy().copy(),raw.data.ctrl.numpy().copy()]
                if arm=='original':
                    for a,b in zip(arrays,saved):a.assign(b)
                    wp.launch(p.roles.original.control_physical_nominal,raw.num_envs,args,block_dim=32)
                    for a,b in zip(output,[raw.k['state'].numpy(),raw.diag.numpy(),raw.data.ctrl.numpy()]):np.testing.assert_array_equal(a,b)
                for w in range(raw.num_envs):p.roles.check_log(log.numpy()[:1,w],mode)
                raw.reset()
                # Original after/history delay path with synthetic gyro sample, no integration.
                h=raw.history.numpy();h[:,:,5]=.123;raw.history.assign(h)
                sensor=raw.data.sensordata.numpy();sensor[:,int(raw.ids.numpy()[10])+2]=.7;raw.data.sensordata.assign(sensor)
                wp.launch(p.rec.old.after,raw.num_envs,after_args(raw),block_dim=32)
                expected=np.where(raw.param.numpy()[:,4]==0,np.float32(.7),np.float32(.123))
                np.testing.assert_array_equal(raw.obs.numpy()[:,5],expected);delay_checks+=raw.num_envs
                raw.reset();assert not raw.state.numpy()[:,0].any() and not raw.data.time.numpy().any()
                assert not raw._role_buffers[0].numpy().any() and not raw._role_buffers[1].numpy().any()
                owners.append(dict(indices=[i for i,c in group],**contract,core=raw._complete_topology,role=raw._role_topology))
            finally:raw.close()
            del raw;gc.collect()
    assert empty and delay_checks==288
    p.rec.atomic_json(p.OUT/'unit.json',dict(verified=True,registered_cases_compiled=144,static_original_and_candidate_worlds=288,
        batch_owners=owners,original_static_noop_inputs_state_diag_control_exact=True,
        original_packet_delays_and41history_checked=True,delayed_packet_synthetic_checks=delay_checks,
        all9terrain_and_named28_cases_checked=True,empty_target_world_checks=empty,
        zero_noise_only_capacity_exceeds_constructor_horizon=True,full_gyro_role_source_inherited=True,
        new_physics_evaluations=0,training_updates=0,
        limits='Constructor/capture/control/syntheticafter only, no actualworld integration. Originalno-op source reuse qualification, not concurrent trajectorybitwise. Mixedterrainfull logs not yet collected.'))
    print('PASS144cases/288static worlds9terrains/delay/flat/named/noop;0physics evaluations',flush=True)


if __name__=='__main__':check()
