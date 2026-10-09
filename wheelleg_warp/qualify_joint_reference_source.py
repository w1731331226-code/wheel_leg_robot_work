"""One real capture and80 static queries; physics graph launches forbidden."""
import json
import time
from pathlib import Path
import mujoco
import numpy as np
import warp as wp
import joint_reference_adapter as adapter
from route_state import RouteState
from execution_history_env import ExecutionHistory
from train_height_comparison import raw_env
from test_nom_yaw_filter_probe import control_args
from native.controller import D
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'


def prepare(raw, slot=0):
    requested, effective, doubles, motor, filtered, seen, base, private, role, phase, park, trace, _ = raw._joint_buffers
    n = raw.num_envs
    args = control_args(raw)
    wp.launch(adapter.parking.phase.roles.prepare, n, [slot, 0, raw.k['reference'], raw.data.qpos,
        raw.data.sensordata, raw.ids, raw.k['state'], raw.state, raw.active, base, role])
    wp.launch(adapter.parking.phase.select_anchor, n, [slot, 1, raw.command, raw.active, base, role, phase])
    wp.launch(adapter.parking.prepare, n, [slot, 1, raw.command, raw.active, raw.state,
        raw.k['state'], requested, seen, effective, park])
    wp.launch(adapter.convert, n, [effective, doubles, motor])
    wp.launch(adapter.reference.prepare, n, [base, doubles, raw.active, filtered, private])
    args[3] = motor; args[12] = private
    return args


def run():
    assert not any((OUT / f).exists() for f in ('source_check_started.json', 'source_check_completion.json', 'source_check_failure.json'))
    p = json.loads((OUT / 'source_check_proposal.json').read_text())
    assert all(sha(ROOT / name) == h for name, h in p['source_sha256'].items())
    fixture = ROOT / p['fixture_path']; assert sha(fixture) == p['fixture_sha256']
    with np.load(fixture, allow_pickle=False) as z:
        data = {name: z[name][p['fixture_rows']].copy() for name in
                ('q', 'v', 'sensor', 'memory_before', 'base_reference')}
    np.testing.assert_array_equal(data['base_reference'][:, 2], [.115, .16, .25, .3, .38])
    records, count, fd_calls = [], 0, []
    raw = env = None
    fd, graph = mujoco.mjd_transitionFD, wp.capture_launch
    def counted(*args, **kwargs):
        fd_calls.append(float(args[2]))
        assert len(fd_calls) <= p['baseline_constructor_FD_budget']
        return fd(*args, **kwargs)
    def reject(*args, **kwargs):
        raise AssertionError('No physical graph launch admitted')
    mujoco.mjd_transitionFD, wp.capture_launch = counted, reject
    atomic_json(OUT / 'source_check_started.json', dict(round=302, runner_sha256=sha(__file__),
        proposal_sha256=sha(OUT / 'source_check_proposal.json'), static_query_budget=80,
        real_constructor_budget=1, physics_graph_launch_budget=0))
    start = time.perf_counter()
    try:
        directory = OUT / 'source_constructor'; directory.mkdir(exist_ok=False)
        raw = adapter.instrument(raw_env, p['cases'], 'virtual6', directory)
        assert raw._joint_topology['captured_controller_calls'] == 80
        assert raw._complete_topology['physics_calls'] == 40
        assert raw._execution_topology['control_steps'] == 40
        route = RouteState(raw); history = ExecutionHistory(route, raw, 'H1')
        env = adapter.JointReferenceActions(history, raw, 'M3')
        obs = env.reset(); assert obs.shape == (5, 481) and env.action_space.shape == (3,)
        np.testing.assert_array_equal(raw.data.time.numpy(), 0)
        np.testing.assert_array_equal(raw._joint_requested.numpy(), 0)
        print('CONSTRUCTED302 real2captures,481reset,FD', len(fd_calls), flush=True)
        ids = raw.ids.numpy(); results = {}
        for phase in p['phases']:
            for arm in p['arms']:
                env.reset(); raw.diag.zero_()
                q, v, sensor, memory = [data[name].copy() for name in ('q', 'v', 'sensor', 'memory_before')]
                command = np.zeros(5); seen = 0
                if phase == 'startup_before_motion':
                    memory[:, 0] = 0
                elif phase == 'asymmetric_moving':
                    sign = np.where(np.arange(5) % 2 == 0, 1., -1.)
                    angle = sign*np.deg2rad(4.)
                    q[:, 3:7] = np.column_stack((np.cos(angle/2), np.sin(angle/2), np.zeros(5), np.zeros(5)))
                    v[:, int(ids[8])] = 60; v[:, int(ids[9])] = 14
                    sensor[:, int(ids[10])] = sign*.2
                    command[:] = .7
                elif phase == 'parking_after_motion':
                    seen = 1; v[:, 0] = .1; memory[:, 7] = .1
                raw.data.qpos.assign(q); raw.data.qvel.assign(v); raw.data.sensordata.assign(sensor)
                raw.k['state'].assign(memory); raw.command.assign(command)
                raw._joint_buffers[5].fill_(seen)
                state = raw.state.numpy(); state[:, 0] = 0; raw.state.assign(state)
                action = np.zeros((5, 3))
                if arm == 'M3':
                    action[:] = [.4, -.3, .5]
                canonical, motor = adapter.decode(action, 'M3', 5)
                if arm == 'U6':
                    canonical, motor = adapter.decode(np.tile([.4, -.3, .2, -.2, .5, -.5], (5, 1)), 'U6', 5)
                raw._joint_requested.assign(canonical); raw.targets.assign(motor)
                args = prepare(raw)
                readonly = {str(i): a.numpy().copy() for i, a in enumerate(args)
                            if isinstance(a, wp.array) and i not in (6, 14, 15)}
                before = raw.k['state'].numpy().copy()
                kernel = adapter.parking.phase.roles.experimental.control_physical_nominal if arm == 'old_zero' else adapter.candidate.control_physical_nominal
                count += 5; wp.launch(kernel, 5, args, block_dim=32)
                wp.launch(adapter.parking.phase.roles.finish, 5, [0, raw.diag, raw._joint_buffers[8]])
                wp.launch(adapter.record, 5, [0, raw.state, raw.command, raw.active, *raw._joint_buffers[6:8],
                    raw._joint_buffers[0], raw._joint_buffers[1], raw._joint_buffers[5],
                    raw.k['state'], raw.diag, raw._joint_buffers[11]])
                result = dict(q=q, v=v, sensor=sensor, command=command, memory_before=before,
                    memory_after=raw.k['state'].numpy().copy(), ctrl=raw.data.ctrl.numpy().copy(),
                    diag=raw.diag.numpy().copy(), trace=raw._joint_buffers[11].numpy()[0].copy(),
                    role=raw._joint_buffers[8].numpy()[0].copy(), phase=raw._joint_buffers[9].numpy()[0].copy())
                file = OUT / f'source_{phase}_{arm}.npz'; np.savez_compressed(file, **result)
                records.append(dict(phase=phase, arm=arm, path=file.name, sha256=sha(file), queries=5))
                for key, value in readonly.items():
                    np.testing.assert_array_equal(args[int(key)].numpy(), value)
                assert all(np.isfinite(result[n]).all() for n in ('ctrl', 'diag', 'memory_after', 'trace'))
                assert not result['diag'][:, 14].any(), 'Static controller mapping failure'
                for w in range(5):
                    adapter.check_log(result['trace'][w:w+1], result['role'][w:w+1], result['phase'][w:w+1], initial_seen=bool(seen))
                results[phase, arm] = result
                np.testing.assert_array_equal(raw.data.time.numpy(), 0)
                print('STATIC', phase, arm, count, flush=True)
            for name in ('ctrl', 'diag', 'memory_after'):
                np.testing.assert_array_equal(results[phase, 'old_zero'][name], results[phase, 'new_zero'][name])
        assert count == 80
        # Real state fields reset through the existing reset_rows kernel plus the exact adapter done-mask clear.
        raw._joint_requested.fill_(.5); raw._joint_buffers[4].fill_(.4); raw._joint_buffers[5].fill_(1)
        mask = np.array([0, 1, 0, 1, 0], np.int32); raw.mask.assign(mask)
        untouched = raw.data.qpos.numpy().copy()
        wp.launch(adapter.environment.reset_rows, 5, raw.reset_args)
        raw._joint_buffers[-1].assign(mask)
        b = raw._joint_buffers
        wp.launch(adapter.clear, 5, [b[-1], b[0], b[1], b[2], b[3], b[4], b[5], b[7]])
        np.testing.assert_array_equal(raw.data.qpos.numpy()[mask == 0], untouched[mask == 0])
        np.testing.assert_array_equal(raw._joint_requested.numpy()[mask == 1], 0)
        np.testing.assert_array_equal(raw._joint_requested.numpy()[mask == 0], .5)
        obs = env.reset(); assert obs.shape == (5, 481)
        np.testing.assert_array_equal(raw._joint_buffers[4].numpy(), 0)
        atomic_json(OUT / 'source_check_completion.json', dict(verified=True, round=302,
            records=records, individual_static_queries=count, constructor_count=1,
            baseline_constructor_transitionFD_calls=fd_calls, topology=raw._joint_topology,
            real_capture_qualified=True, phase_startup_asymmetric_static_qualified=True,
            zero_static_output_identity=True, masked_native_reset_and_adapter_clear_checked=True,
            actual_episode_autoreset_qualified=False, physics_graph_launches=0, new_training_samples=0,
            wall_seconds=time.perf_counter()-start, runner_sha256=sha(__file__),
            formal_PPO_admitted=False, full_task_rollout_admitted=False,
            next='303 independent finite source review and register one paired dynamic mechanism only if sourcepassed andsame-reference analytic controls specified. No newPPO.'))
        print('PASS302 real capture/80static/phase/maskedreset;0graphphysics/PPO', flush=True)
    except BaseException as error:
        atomic_json(OUT / 'source_check_failure.json', dict(error=repr(error), records=records,
            attempted_static_queries=count, baseline_constructor_transitionFD_calls=fd_calls,
            implicit_retry=False, physics_graph_launches=0, formal_PPO_admitted=False))
        raise
    finally:
        mujoco.mjd_transitionFD, wp.capture_launch = fd, graph
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()


if __name__ == '__main__':
    run()
