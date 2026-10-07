"""Reuse qualified parking kernels without dense trajectory recording."""
import numpy as np
import warp as wp
import light_phase_reference as light
import parking_withdrawal_probe as gate
from task_mode_recorder import signature


def instrument(factory, cases, mode):
    n = len(cases); seen = wp.zeros(n, dtype=int); effective = wp.zeros((n, 6))
    log = wp.zeros((40, n, 30), dtype=gate.D)
    launch = wp.launch; state = command = None; calls = 0
    def spy(kernel, dim, inputs=None, **kwargs):
        nonlocal state, command, calls
        if kernel is light.environment.command_step: state, command = inputs[0], inputs[2]
        if kernel is light.phase.roles.experimental.control_physical_nominal:
            assert state is not None and command is inputs[4]
            slot = calls%40; calls += 1
            launch(gate.prepare, n, [slot, 1, command, inputs[5], state, inputs[6], inputs[3], seen, effective, log])
            args = list(inputs); args[3] = effective
            assert all(signature(a)==signature(b) for j,(a,b) in enumerate(zip(inputs,args)) if j!=3)
            result = launch(kernel, dim, args, **kwargs)
            launch(gate.finish, n, [slot, inputs[6], inputs[15], log]); return result
        return launch(kernel, dim, **kwargs) if inputs is None else launch(kernel, dim, inputs, **kwargs)
    wp.launch = spy
    try: raw = light.instrument(factory, cases, mode)
    finally: wp.launch = launch
    assert calls == 40
    raw._light_parking_buffers = (seen, effective, log)
    raw._light_parking_stats = dict(valid_samples=0, withdraw_samples=0)
    reset, wait = raw.reset, raw.step_wait
    def reset_all():
        value = reset(); seen.zero_(); effective.zero_(); log.zero_(); return value
    def step_wait():
        value = wait(); x = log.numpy(); x = x[x[:, :, 0]==1]
        active = (x[:, 3]==1)&(x[:, 2]==0)
        np.testing.assert_array_equal(x[:, 4], active)
        np.testing.assert_array_equal(x[:, 11:17], np.where(active[:, None], 0, x[:, 5:11]))
        expected = x[:, 17:23]+np.clip(x[:, 11:17]-x[:, 17:23], -.01, .01)
        assert np.all(np.all(x[:, 23:29]==expected, axis=1)|((x[:, 29]==2)&np.all(x[:, 23:29]==x[:, 17:23], axis=1)))
        raw._light_parking_stats['valid_samples'] += len(x)
        raw._light_parking_stats['withdraw_samples'] += int(active.sum())
        if value[2].any(): wp.launch(gate.clear_rows, n, [raw.mask, seen])
        return value
    raw.reset, raw.step_wait = reset_all, step_wait
    return raw
