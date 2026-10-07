"""Bounded collector qualification: synthetic CUDA clocks +480 real steps."""
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import warp as wp
import execution_input_collector as collector
from execution_input_features import features, unit as feature_unit
from native.controller import D
from native.terrain import HeightTerrainScenario
from review_yaw_sector import ROOT, sha

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/execution_input_history_v1'


@wp.kernel
def copy_ctrl(slot: int, ctrl: wp.array2d[float], trace: wp.array3d[float]):
    w = wp.tid()
    for j in range(6): trace[slot, w, j] = ctrl[w, j]


def synthetic():
    n = 4; state = wp.zeros((n, 39), dtype=D); active = wp.ones(n, dtype=int)
    ctrl = wp.zeros((n, 6)); total = wp.zeros((n, 6), dtype=D)
    prefix = wp.zeros((n, collector.RING, 6), dtype=D)
    stamp = wp.full((n, collector.RING), -1, dtype=int); mask = wp.ones(n, dtype=int)
    wp.launch(collector.clear, n, [mask, total, prefix, stamp])
    counts = np.zeros(n, np.int64); previous = counts.copy(); delay = np.array([0, 10, 20, 40])
    history = [[] for _ in range(n)]; checked = 0
    for tick in range(1, 161):
        s = state.numpy(); s[:, 0] = counts; state.assign(s)
        value = np.array([[(counts[w]+1)*.1+w+j*.01 for j in range(6)] for w in range(n)], np.float32)
        ctrl.assign(value); wp.launch(collector.record, n, [state, active, ctrl, total, prefix, stamp])
        for w, a in enumerate(active.numpy()):
            if a: history[w].append(value[w].astype(float)*.0005); counts[w] += 1
        if tick == 13: active.assign(np.array([0, 1, 1, 1], np.int32))
        if tick%40 == 0:
            impulse, elapsed, end = collector.interval(prefix.numpy(), stamp.numpy(), previous, counts, delay)
            expected = np.array([np.sum(history[w][previous[w]:end[w]], axis=0) if end[w]>previous[w]
                                 else np.zeros(6) for w in range(n)])
            np.testing.assert_allclose(impulse, expected, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(elapsed, (end-previous)*.0005)
            previous[:] = end; checked += 1
            if tick == 40:
                assert counts[0] == 13 and elapsed[0] == .0065
                old = prefix.numpy().copy(); mask.assign(np.array([1, 0, 0, 0], np.int32))
                wp.launch(collector.clear, n, [mask, total, prefix, stamp])
                np.testing.assert_array_equal(prefix.numpy()[1:], old[1:])
                counts[0] = previous[0] = 0; history[0] = []; active.fill_(1)
    try: collector.interval(prefix.numpy(), stamp.numpy(), previous-100, counts, delay)
    except ValueError: pass
    else: raise AssertionError('Expired/reversed episode accepted')
    return dict(polls=checked, terminal_steps=13, partial_elapsed_s=.0065,
                ring_wrap_tested=True, partial_reset_preserves_other_worlds=True, physics_steps=0)


def real(cases):
    import light_phase_reference as light
    from train_height_comparison import raw_env
    from route_state import RouteState
    trace = wp.zeros((40, 4, 6)); calls = 0; launch = wp.launch
    def spy(kernel, dim, inputs=None, **kwargs):
        nonlocal calls
        result = launch(kernel, dim, **kwargs) if inputs is None else launch(kernel, dim, inputs, **kwargs)
        if kernel is collector.record:
            launch(copy_ctrl, 4, [calls%40, inputs[2], trace]); calls += 1
        return result
    wp.launch = spy
    try: raw = collector.instrument(lambda rows, mode: light.instrument(raw_env, rows, mode), cases, 'virtual6')
    finally: wp.launch = launch
    assert calls == 40
    env = RouteState(raw); previous = env.reset(); packets = [previous.copy()]; records = []; controls = []
    previous_sensor = np.zeros(4, np.int64); cumulative = np.zeros((1, 4, 6))
    wait = raw.step_wait; snapshots = {}
    def observed_wait():
        buffers = [raw.data.qpos, raw.data.qvel, raw.data.ctrl, raw.state, raw.k['state']]
        before = [b.numpy().copy() for b in buffers]
        value = wait()
        for b, v in zip(buffers, before): np.testing.assert_array_equal(b.numpy(), v)
        snapshots['packet'] = value[0].copy()
        snapshots['reward'] = value[1].copy(); snapshots['done'] = value[2].copy()
        return value
    raw.step_wait = observed_wait
    try:
        for tick in range(3):
            obs, reward, done, _ = env.step(np.zeros((4, 6), np.float32))
            assert not done.any()
            np.testing.assert_array_equal(obs[:, :38], snapshots['packet'])
            np.testing.assert_array_equal(reward, snapshots['reward']); np.testing.assert_array_equal(done, snapshots['done'])
            batch = trace.numpy().copy(); controls.append(batch)
            accumulated = np.cumsum(batch.astype(float)*.0005, axis=0)+cumulative[-1]
            cumulative = np.concatenate([cumulative, accumulated])
            end = np.maximum((tick+1)*40-raw.param.numpy()[:, 4].astype(np.int64), 0)
            expected = cumulative[end, np.arange(4)]-cumulative[previous_sensor, np.arange(4)]
            data = raw._execution_intervals
            np.testing.assert_allclose(data['command_integral'], expected, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(data['sensor_elapsed_s'], (end-previous_sensor)*.0005)
            np.testing.assert_array_equal(obs[:, 20:22], raw.history.numpy()[np.arange(4), end%41, 20:22])
            derived = features(previous, obs, data['command_integral'], data['sensor_elapsed_s'])
            records.append(dict(actor_step=tick+1, integral=data['command_integral'].tolist(),
                                elapsed=data['sensor_elapsed_s'].tolist(), features=derived.tolist()))
            previous_sensor[:] = end; previous = obs.copy(); packets.append(obs.copy())
        assert np.all(raw.state.numpy()[:, 0] == 120)
        collector.preserve(raw, OUT/'partial_buffers.npz')
        saved = np.load(OUT/'partial_buffers.npz', allow_pickle=False)
        for name, b in zip(('total', 'prefix', 'stamp'), raw._execution_buffers):
            np.testing.assert_array_equal(saved[name], b.numpy())
        env.reset(); assert not raw._execution_previous_sensor_steps.any()
        assert not raw._execution_buffers[0].numpy().any() and raw._execution_intervals is None
        np.savez_compressed(OUT/'real_trace.npz', final_ctrl=np.concatenate(controls), packets=np.array(packets))
        return dict(physical_steps=480, actor_steps=3, worlds=4, records=records,
                    final_ctrl_independently_recorded=True, delayed_packet_wheel_and_impulse_aligned=True,
                    same_execution_output_and_control_buffers_unchanged=True,
                    partial_buffers_preserved=True, explicit_reset_clears_history=True,
                    real_terminal_or_curriculum_transition_tested=False, topology=raw._execution_topology)
    finally: env.close()


def run():
    assert not any((OUT/n).exists() for n in ('collector_registration.json', 'round229_collector_unit.json', 'collector_failure.json'))
    cases = [dict(seed=2290000+i, scenario=asdict(HeightTerrainScenario(speed=.7, mass=7., height_l=0., height_r=0.,
             center=1.8, offset=0., mu_l=.8, mu_r=.8, drive_difference=0., delay_ms=delay,
             solver_iterations=100, stand_height_m=height))) for i, (height, delay) in
             enumerate(zip((.115, .16, .30, .38), (0., 5., 10., 20.)))]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve(),
        ROOT/'wheelleg_warp/execution_input_collector.py', ROOT/'wheelleg_warp/execution_input_features.py',
        ROOT/'wheelleg_warp/light_phase_reference.py', ROOT/'wheelleg_warp/native/environment.py']}
    collector_file = OUT/'collector_registration.json'
    collector_file.write_text(json.dumps(dict(cases=cases, maximum_physics_steps=480, training_steps=0,
         synthetic_gpu_no_integrator=True, source_sha256=sources, proposal_sha256=sha(OUT/'proposal.json'),
         scope='Interface qualification only,not controlperformance. Synthetic13stepterminal+partialrowreset;real3stepclock/collector/noop only.'), indent=2)+'\n')
    try:
        feature_unit(); synthetic_result = synthetic(); result = real(cases)
        assert all(sha(ROOT/n) == v for n, v in sources.items())
        (OUT/'round229_collector_unit.json').write_text(json.dumps(dict(verified=True, round=229,
            synthetic=synthetic_result, real=result, source_sha256=sources,
            registration_sha256=sha(collector_file), new_training_steps=0, main_training_admitted=False,
            limits='Only4staticworlds/3actorsteps;synthetic terminal/reset is not real autoreset or curriculum qualification. Tenendpoint policywrapper/RMS/reload not implemented. Finalctrl before gain is not actualtorque. Sensor clock routing reads private delay internally,not exposed as an Actor feature.'), indent=2)+'\n')
        print('PASS229 CUDAclock/ring/partialreset+480realphysics,alignedinput/noop;training0', flush=True)
    except BaseException as e:
        (OUT/'collector_failure.json').write_text(json.dumps(dict(error=repr(e), implicit_retry=False), indent=2)+'\n'); raise


if __name__ == '__main__': run()
