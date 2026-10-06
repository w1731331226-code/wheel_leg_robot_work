"""Current-command, branch identity and synthetic stream checks; no integration."""
import gc
import json
import tempfile
from pathlib import Path
import numpy as np
import warp as wp
import phase_support_probe as p
from test_nom_yaw_filter_probe import control_args, after_args


def outputs(raw):
    return [a.numpy().copy() for a in (raw.k['state'], raw.diag, raw.data.ctrl)]


def static_query(raw, arm, command, degrees):
    raw.reset()
    q = raw.data.qpos.numpy(); q[:, 3] = np.cos(np.deg2rad(degrees)/2)
    q[:, 4] = np.sin(np.deg2rad(degrees)/2); raw.data.qpos.assign(q)
    raw.command.fill_(command)
    args = control_args(raw)
    wp.launch(p.roles.original.control_physical_nominal, raw.num_envs, args, block_dim=32)
    state = raw.k['state'].numpy(); state[:, 0] = 2.; state[:, 3:9] = 0.; raw.k['state'].assign(state)
    arrays = [a for a in args if isinstance(a, wp.array)]
    saved = [a.numpy().copy() for a in arrays]
    p.prepare_step(raw, 0, arm)
    for a, b in zip(arrays, saved):
        np.testing.assert_array_equal(a.numpy(), b)
    ref, role = raw._role_buffers
    assert not ref.numpy()[:, 10:15].any()
    altered = list(args); altered[12] = ref
    wp.launch(p.roles.experimental.control_physical_nominal, raw.num_envs, altered, block_dim=32)
    wp.launch(p.roles.finish, raw.num_envs, [0, raw.diag, role])
    actual = outputs(raw)
    for w in range(raw.num_envs):
        p.check_log(raw._phase_buffer.numpy()[:1, w], arm)
        p.roles.check_log(role.numpy()[:1, w], int(arm == 'floor_only'))
    for a, b in zip(arrays, saved):
        a.assign(b)
    if arm == 'original' or command == 0:
        wp.launch(p.roles.original.control_physical_nominal, raw.num_envs, args, block_dim=32)
    else:
        # The pre-existing fixedfloor implementation is the independent expected branch.
        wp.launch(p.roles.prepare, raw.num_envs, [0, 1, args[12], args[0], args[2], args[7],
            args[6], raw.state, args[5], ref, role])
        wp.launch(p.roles.experimental.control_physical_nominal, raw.num_envs, altered, block_dim=32)
    for a, b in zip(actual, outputs(raw)):
        np.testing.assert_array_equal(a, b)
    assert np.isfinite(actual[2]).all()
    assert np.all(abs(actual[2][:, :4]) <= 40+1e-6) and np.all(abs(actual[2][:, 4:]) <= 4.5+1e-6)


def command_boundaries(raw):
    for step, arrived in ((0, False), (2000, False), (2001, False), (4000, False), (4000, True)):
        raw.reset(); state = raw.state.numpy(); state[:, 0] = step
        state[:, 1] = 0. if arrived else -1.; raw.state.assign(state)
        raw.command.fill_(777.)  # The new command must replace this stale sentinel.
        d = raw.data
        wp.launch(p.rec.old.command_step, raw.num_envs, [raw.state, raw.param, raw.command, raw.active,
            d.qpos, d.qvel, d.qacc_warmstart, raw.stopped_q, raw.stopped_v, raw.stopped_w, raw.contact_flags])
        p.prepare_step(raw, 0, 'phase_support')
        expected = raw.param.numpy()[:, 0]*np.clip(step*.0005-1., 0., 1.)
        if arrived:
            expected[:] = 0.
        np.testing.assert_array_equal(raw.command.numpy(), expected)
        np.testing.assert_array_equal(raw._phase_buffer.numpy()[0, :, 2], expected)
        for w in range(raw.num_envs):
            # Set the fixture step to one only for the single-row checker.
            row = raw._phase_buffer.numpy()[:1, w].copy(); row[:, 1] = 1
            p.check_log(row, 'phase_support')


def stream_fixture(case):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp); raw = p.instrument([case], 'diff3', directory=directory)
        try:
            raw.reset()
            geometry = json.loads((directory/'geometry.json').read_text())
            assert geometry['case_ids'] == [case['seed']]
            assert geometry['recording_case_ids'] == raw._broad_contract['recording_ids']
            ref, role = raw._role_buffers; table, gyro, error, _ = raw._gyro_buffers
            trace, _, count, *_ = raw._complete_buffers
            for step, command in enumerate((0., -1., 0.), 1):
                state = raw.state.numpy(); state[0, 0] = step-1; raw.state.assign(state)
                raw.command.fill_(command); p.prepare_step(raw, 0, 'phase_support')
                args = control_args(raw)
                wp.launch(p.roles.noise.before_control, 1, [0, args[2], args[7], raw.state,
                    args[6], args[5], 0, table.shape[1], gyro, error])
                args[12] = ref
                wp.launch(p.roles.experimental.control_physical_nominal, 1, args, block_dim=32)
                wp.launch(p.roles.finish, 1, [0, raw.diag, role])
                wp.launch(p.roles.noise.filtered_measurement, 1, [0, raw.k['state'], gyro])
                wp.launch(p.roles.noise.fresh_measurement, 1, [0, raw.data.sensordata, raw.ids,
                    raw.state, raw.active, table, 0, gyro, error])
                values = np.zeros(trace.shape); values[0, 0, :4] = [1, step, (step-1)*.0005, step*.0005]
                values[0, 0, 43] = command; trace.assign(values)
                count.assign(np.array([[step, step]], dtype=np.float64))
                state[0, 0] = step; raw.state.assign(state)
                if step == 3:
                    raw.done.fill_(5)
                result = raw.step_wait()
                if step == 2:
                    partial = p.preserve_phase(raw, directory)
                    assert list(partial['prefix_rows'].values()) == [2] and not partial['complete_case_ids']
            assert result[3][0]['complete_counts'] == dict(steps=3, contacts=0)
            assert raw._phase_frozen == raw._role_frozen == raw._gyro_frozen == raw._complete_frozen == {0}
            assert not raw._phase_chunks[0]
            file = directory/result[3][0]['phase_trace']['path']
            with np.load(file, allow_pickle=False) as z:
                assert z['columns'].tolist() == p.COL
                p.check_log(z['trace'], 'phase_support')
                np.testing.assert_array_equal(z['trace'][:, 2], [0., -1., 0.])
            raw.done.fill_(5); assert 'phase_trace' not in raw.step_wait()[3][0]
            raw.reset()
            assert not raw._phase_buffer.numpy().any() and not ref.numpy().any() and not role.numpy().any()
            assert not raw._phase_frozen and not raw._phase_chunks[0]
            assert not raw.state.numpy()[:, 0].any() and not raw.data.time.numpy().any()
        finally:
            raw.close()


def check():
    proposal = json.loads((p.OUT/'proposal.json').read_text())
    cases = sum(proposal['panels'].values(), []); assert len(cases) == 164
    groups = p.broad.batches(cases)
    assert sorted(i for group in groups for i, _ in group) == list(range(164))
    defaults = (wp.launch, p.roles.noise.noise_table, p.rec.old.geometry,
                p.roles.environment.control_physical_nominal, p.rec.old.control_physical_nominal)
    owners = []; queries = delayed = 0
    for group in groups:
        selected = [c for _, c in group]
        for arm in ('original', 'phase_support'):
            raw = p.instrument(selected, 'diff3', arm)
            try:
                assert defaults == (wp.launch, p.roles.noise.noise_table, p.rec.old.geometry,
                    p.roles.environment.control_physical_nominal, p.rec.old.control_physical_nominal)
                for degrees in ((0.,) if arm == 'original' else (-5., 0., 5.)):
                    for command in ((0.,) if arm == 'original' else (0., 1., -1., 1e-12, -1e-12)):
                        static_query(raw, arm, command, degrees); queries += raw.num_envs
                if arm == 'phase_support':
                    command_boundaries(raw)
                raw.reset()
                h = raw.history.numpy(); h[:, :, 5] = .123; raw.history.assign(h)
                sensor = raw.data.sensordata.numpy(); sensor[:, int(raw.ids.numpy()[10])+2] = .7
                raw.data.sensordata.assign(sensor)
                wp.launch(p.rec.old.after, raw.num_envs, after_args(raw), block_dim=32)
                expected = np.where(raw.param.numpy()[:, 4] == 0, np.float32(.7), np.float32(.123))
                np.testing.assert_array_equal(raw.obs.numpy()[:, 5], expected); delayed += raw.num_envs
                raw.reset(); active = raw.active.numpy(); active[0] = 0; raw.active.assign(active)
                p.prepare_step(raw, 0, arm); assert not raw._phase_buffer.numpy()[0, 0].any()
                raw.reset(); assert not raw._phase_buffer.numpy().any()
                assert not raw.state.numpy()[:, 0].any() and not raw.data.time.numpy().any()
                owners.append(dict(indices=[i for i, _ in group], phase=raw._phase_topology,
                    broad=raw._broad_contract, role=raw._role_topology, core=raw._complete_topology,
                    gyro=raw._gyro_topology))
            finally:
                raw.close()
            del raw; gc.collect()
            print('CHECKED', arm, [i for i, _ in group], flush=True)
    for case in (proposal['panels']['legacy'][0], proposal['panels']['controlled'][0]):
        stream_fixture(case)
    assert delayed == 328 and queries == 2624
    p.rec.atomic_json(p.OUT/'unit.json', dict(verified=True, registered_cases=164,
        original_and_phase_static_worlds=328, static_output_comparisons=queries,
        branch_comparison_commands=[0., 1., -1., 1e-12, -1e-12], synthetic_root_roll_deg=[-5., 0., 5.],
        batch_owners=owners, current_command_after_command_step_matches_control_pointer=True,
        all_original_noop_and_phase_branch_outputs_exact=True, original_input_arrays_not_mutated_by_prepare=True,
        startup_ramp_and_postarrival_current_command_checked=True, synthetic_delayed_packet_checks=delayed,
        inactive_and_owned_reset_checked=True, two_synthetic_metadata_terminal_partial_fixtures=True,
        nested_four_stream_terminal_frozen_once=True, repeated_done_not_duplicated=True,
        partial_prefix_and_last_buffer_retained=True, globals_restored=True,
        new_physics_evaluations=0, training_updates=0,
        limits='Constructor/capture/static controls and synthetic after/stream fixtures only; no physics integration. '
            'Branch equivalence supports source-qualified reuse, not trajectory bitwise, dynamic safety or performance.'))
    print('PASS164 cases/328 static worlds/2624 output comparisons, current command/streams/reset;0 evaluations', flush=True)


if __name__ == '__main__':
    check()
