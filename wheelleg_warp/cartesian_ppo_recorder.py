"""Read-only missing evidence around the frozen Cartesian light adapter."""
import numpy as np
import warp as wp

import cartesian_light_runtime as cart
import joint_state_guard as guard
from native.controller import D

COLUMNS = ('valid', 'wheel_left_rad_s', 'wheel_right_rad_s',
           'Nom_invalid_base_flag', 'Nom_projection_error')


@wp.kernel
def record(slot: int, v: wp.array2d[float], active: wp.array[int],
           ids: wp.array[int], diag: wp.array2d[D], out: wp.array3d[D]):
    w = wp.tid()
    for j in range(5):out[slot, w, j] = D(0)
    if active[w] == 0:return
    out[slot, w, 0] = D(1)
    out[slot, w, 1] = D(v[w, ids[8]])
    out[slot, w, 2] = D(v[w, ids[9]])
    out[slot, w, 3] = diag[w, 13]
    out[slot, w, 4] = diag[w, 14]


def signature(value):
    if hasattr(value, 'ptr'):
        return ['array', int(value.ptr), list(value.shape), str(value.dtype)]
    return ['scalar', type(value).__name__, repr(value)]


def instrument(factory, cases, mode):
    """Same factory/result tuple as cart.instrument; install outside its spy."""
    n = len(cases)
    if mode not in (0, 1) or not 1 <= n <= 100:
        raise ValueError('Cartesian mode and declared1..100 worlds required')
    wp.init();wp.set_device('cuda:0')
    nominal = wp.zeros((40, n, 5), dtype=D)
    launch = wp.launch;last = zero = effective = None;phase = None;slots = []
    def spy(kernel, dim, inputs=None, **kwargs):
        nonlocal last, zero, effective, phase
        if kernel is cart.NOMINAL:
            assert phase in (None, 'guard') and len(slots) < 40 and len(inputs) == 20
            assert dim == n and inputs[0].shape == (n, 17) and inputs[1].shape == (n, 16)
            assert inputs[2].shape == (n, 28) and inputs[3].shape == (n, 6)
            assert inputs[6].shape == (n, 30) and inputs[15].shape == (n, 38)
            if zero is None:zero = inputs[3]
            else:assert inputs[3] is zero
            last = inputs;slot = len(slots)
            slots.append(dict(slot=slot, nominal_args=[signature(x) for x in inputs]))
            value = launch(kernel, dim, inputs, **kwargs)
            launch(record, n, [slot, inputs[1], inputs[5], inputs[7], inputs[15], nominal])
            phase = 'Nom';return value
        if kernel is cart.deliver:
            assert phase == 'Nom' and inputs[0] == len(slots) - 1
            assert all(inputs[i] is last[j] for i, j in ((2,0),(3,1),(4,7),(6,5),(9,6),(11,14),(12,15)))
            assert inputs[5] is not zero
            if effective is None:effective = inputs[5]
            else:assert inputs[5] is effective
            slots[-1]['candidate_args'] = [signature(x) for x in inputs]
            value = launch(kernel, dim, inputs, **kwargs);phase = 'candidate';return value
        if kernel is guard.guard:
            assert phase == 'candidate' and inputs[0] == len(slots) - 1
            assert all(inputs[i] is last[j] for i, j in ((1,0),(2,1),(3,5),(4,7),(5,14),(6,15)))
            slots[-1]['guard_args'] = [signature(x) for x in inputs]
            value = launch(kernel, dim, inputs, **kwargs);phase = 'guard';return value
        return launch(kernel, dim, **kwargs) if inputs is None else launch(kernel, dim, inputs, **kwargs)
    wp.launch = spy
    try:
        result = cart.instrument(factory, cases, mode);raw = result[0]
        assert len(slots) == 40 and phase == 'guard' and all('guard_args' in row for row in slots)
        assert zero is raw._cartesian_buffers['zero'] and effective is raw._light_parking_buffers[1]
        assert all(last[i] is buffer for i, buffer in ((0,raw.data.qpos),(1,raw.data.qvel),
            (2,raw.data.sensordata),(5,raw.active),(6,raw.k['state']),(7,raw.ids),(14,raw.data.ctrl),(15,raw.diag)))
        assert all(row['nominal_args'] == slots[0]['nominal_args'] for row in slots)
    finally:wp.launch = launch
    raw._cartesian_ppo_buffers = dict(nominal=nominal)
    raw._cartesian_ppo_topology = dict(worlds=n, control_slots=40,
        columns=COLUMNS, record_after_actual_Nom_before_candidate=True,
        zero_target=signature(zero), effective_target=signature(effective), slots=slots,
        same_control_owners_verified=True, original_launch_restored=True)
    reset = raw.reset
    def reset_all():
        value = reset();nominal.zero_();return value
    raw.reset = reset_all
    return result


def post_private(raw):
    values = dict(guard_memory=raw._joint_guard_buffers[0],
        parking_seen=raw._light_parking_buffers[0], parking_effective=raw._light_parking_buffers[1],
        phase_reference=raw._light_phase_buffers[0], phase_role=raw._light_phase_buffers[1],
        phase_monitor=raw._light_phase_buffers[2], command_total=raw._execution_buffers[0],
        command_prefix=raw._execution_buffers[1], command_stamp=raw._execution_buffers[2])
    result = {name:buffer.numpy().copy() for name,buffer in values.items()}
    result['previous_sensor_steps'] = raw._execution_previous_sensor_steps.copy()
    return result


def bank_actual(raw):
    return {name:buffer.numpy().copy() for name,buffer in dict(param=raw.param,
        q0=raw.q0,obs0=raw.obs0,native_reference=raw.k['reference']).items()}
