"""Read-only final-command prefix at the delayed sensor clock."""
import numpy as np
import warp as wp
import mujoco_warp as mjw
import native.environment as environment
from native.controller import D
from task_mode_recorder import signature

RING = 82  # 40-step packet interval + at most40 steps sensor delay + endpoints.


@wp.kernel
def record(state: wp.array2d[D], active: wp.array[int], ctrl: wp.array2d[float],
           total: wp.array2d[D], prefix: wp.array3d[D], stamp: wp.array2d[int]):
    w = wp.tid()
    if active[w] == 0: return
    step = int(state[w, 0])+1
    slot = step % RING
    stamp[w, slot] = step
    for j in range(6):
        total[w, j] = total[w, j]+D(ctrl[w, j])*D(.0005)
        prefix[w, slot, j] = total[w, j]


@wp.kernel
def clear(mask: wp.array[int], total: wp.array2d[D], prefix: wp.array3d[D], stamp: wp.array2d[int]):
    w = wp.tid()
    if mask[w] == 0: return
    for j in range(6): total[w, j] = D(0)
    for slot in range(RING):
        stamp[w, slot] = -1
        if slot == 0: stamp[w, slot] = 0
        for j in range(6): prefix[w, slot, j] = D(0)


def interval(prefix, stamps, previous, counts, delay):
    previous, counts, delay = map(np.asarray, (previous, counts, delay))
    n = len(counts)
    if prefix.shape != (n, RING, 6) or stamps.shape != (n, RING):
        raise ValueError('Invalid command-prefix storage')
    if any(v.shape != (n,) or not np.issubdtype(v.dtype, np.integer) for v in (previous, counts, delay)):
        raise ValueError('Integer per-world sensor clocks required')
    end = np.maximum(counts-delay, 0)
    if np.any(delay<0) or np.any(delay>40) or np.any(previous<0) or np.any(end<previous) or np.any(end-previous>40):
        raise ValueError('Invalid interval,delay or missing episode reset')
    world = np.arange(n)
    if not np.array_equal(stamps[world, end%RING], end) or not np.array_equal(stamps[world, previous%RING], previous):
        raise ValueError('Stale or overwritten command prefix')
    impulse = prefix[world, end%RING]-prefix[world, previous%RING]
    if not np.isfinite(impulse).all(): raise ValueError('Nonfinite command impulse')
    return impulse.copy(), (end-previous)*.0005, end.copy()


def instrument(factory, cases, mode, expected_captures=1):
    n = len(cases)
    if n == 0: raise ValueError('Nonempty native worlds required')
    total = wp.zeros((n, 6), dtype=D)
    prefix = wp.zeros((n, RING, 6), dtype=D)
    stamp = wp.full((n, RING), -1, dtype=int)
    mask = wp.ones(n, dtype=int)
    wp.launch(clear, n, [mask, total, prefix, stamp])
    launch, step = wp.launch, mjw.step
    state = active = None; calls = []
    def spy(kernel, dim, inputs=None, **kwargs):
        nonlocal state, active
        if kernel is environment.command_step:
            state, active = inputs[0], inputs[3]
        return launch(kernel, dim, **kwargs) if inputs is None else launch(kernel, dim, inputs, **kwargs)
    def physical(model, data, *args, **kwargs):
        assert state is not None and active is not None
        calls.append((signature(state), signature(active), signature(data.ctrl)))
        launch(record, n, [state, active, data.ctrl, total, prefix, stamp])
        return step(model, data, *args, **kwargs)
    wp.launch, mjw.step = spy, physical
    try: raw = factory(cases, mode)
    finally: wp.launch, mjw.step = launch, step
    assert len(calls) == 40*expected_captures and all(c == (signature(raw.state), signature(raw.active), signature(raw.data.ctrl)) for c in calls)
    raw._execution_buffers = (total, prefix, stamp)  # Strong ownership for captured graph.
    raw._execution_topology = dict(control_steps=40, final_ctrl_before_physics=True,
        original_step_function_restored=True, prefix_ring=RING, reads_only_physical_control=True)
    previous = np.zeros(n, dtype=np.int64)
    reset, wait = raw.reset, raw.step_wait
    def reset_all():
        value = reset()
        mask.fill_(1); wp.launch(clear, n, [mask, total, prefix, stamp])
        previous.fill(0); raw._execution_intervals = None
        return value
    def step_wait():
        # Read BEFORE Native autoreset destroys the terminal count and packet.
        count = raw.state.numpy()[:, 0].astype(np.int64)
        delay = raw.param.numpy()[:, 4].astype(np.int64)
        impulse, elapsed, end = interval(prefix.numpy(), stamp.numpy(), previous, count, delay)
        value = wait(); done = value[2]
        raw._execution_intervals = dict(command_integral=impulse, sensor_elapsed_s=elapsed,
                                        terminal_worlds=done.copy())
        previous[:] = end
        if done.any():
            mask.assign(done.astype(np.int32)); wp.launch(clear, n, [mask, total, prefix, stamp])
            previous[done] = 0
        return value
    raw.reset, raw.step_wait = reset_all, step_wait
    raw._execution_previous_sensor_steps = previous
    raw._execution_intervals = None
    return raw


def preserve(raw, path):
    values = {name: b.numpy() for name, b in zip(('total', 'prefix', 'stamp'), raw._execution_buffers)}
    values['previous_sensor_steps'] = raw._execution_previous_sensor_steps.copy()
    if raw._execution_intervals is not None:
        values.update({name: value.copy() for name, value in raw._execution_intervals.items()})
    np.savez_compressed(path, **values)
