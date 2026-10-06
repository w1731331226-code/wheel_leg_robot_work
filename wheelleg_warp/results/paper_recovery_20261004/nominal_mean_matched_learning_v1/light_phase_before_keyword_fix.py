"""Qualified phase anchor, with small buffers and no dense training recorder."""
import numpy as np
import warp as wp
import train_height_comparison as base
import native.environment as environment
import phase_support_probe as phase


def instrument(factory, cases, mode):
    if not cases or len(cases) > 1024:
        raise ValueError('Nonempty declared native world batch required')
    phase.roles.check_control_copy()
    n = len(cases)
    ref = wp.zeros((n, 16), dtype=phase.D)
    role = wp.zeros((40, n, 11), dtype=phase.D)
    monitor = wp.zeros((40, n, 6), dtype=phase.D)
    launch, kernel = wp.launch, environment.control_physical_nominal
    state = command = None; calls = 0; signatures = []
    def spy(k, dim, inputs=None, **kwargs):
        nonlocal state, command, calls
        if k is environment.command_step:
            state, command = inputs[0], inputs[2]
        if k is phase.roles.experimental.control_physical_nominal:
            assert state is not None and command is inputs[4] and inputs[12].shape[1] == 10
            slot = calls % 40; calls += 1
            launch(phase.roles.prepare, dim, [slot, 0, inputs[12], inputs[0], inputs[2], inputs[7],
                inputs[6], state, inputs[5], ref, role])
            launch(phase.select_anchor, dim, [slot, 1, command, inputs[5], ref, role, monitor])
            args = list(inputs); args[12] = ref
            signatures.append(tuple(phase.rec.old.signature(x) for x in args))
            result = launch(k, dim, args, **kwargs)
            launch(phase.roles.finish, dim, [slot, inputs[15], role])
            return result
        return launch(k, dim, **kwargs) if inputs is None else launch(k, dim, inputs, **kwargs)
    wp.launch = spy; environment.control_physical_nominal = phase.roles.experimental.control_physical_nominal
    try:
        raw = factory(cases, mode)
    finally:
        wp.launch, environment.control_physical_nominal = launch, kernel
    assert calls == 40 and len(signatures) == 40
    raw._light_phase_buffers = (ref, role, monitor)
    raw._light_phase_topology = dict(verified=True, worlds=n, mode=mode, control_calls=40,
        actual_reference_shape=list(ref.shape), current_command_owner=True,
        dense_recording_installed=False, external_force_installed=False,
        small_owned_buffer_bytes=(n*16+40*n*17)*8)
    reset = raw.reset
    def reset_all():
        result = reset()
        for buffer in raw._light_phase_buffers: buffer.zero_()
        return result
    raw.reset = reset_all
    return raw


def prepare_step(raw, slot=0):
    ref, role, monitor = raw._light_phase_buffers
    wp.launch(phase.roles.prepare, raw.num_envs, [slot, 0, raw.k['reference'], raw.data.qpos,
        raw.data.sensordata, raw.ids, raw.k['state'], raw.state, raw.active, ref, role])
    wp.launch(phase.select_anchor, raw.num_envs, [slot, 1, raw.command, raw.active, ref, role, monitor])


def curriculum(p, mode, seed, n, milestones=None):
    factory = base.raw_env
    base.raw_env = lambda cases, native: instrument(factory, cases, native)
    try:
        return base.CurriculumEnv(p, mode, seed, n, milestones=milestones)
    finally:
        base.raw_env = factory
