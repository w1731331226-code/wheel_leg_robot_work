"""Current-command protective anchor on the frozen dense recording pipeline."""
import json
from pathlib import Path
import numpy as np
import warp as wp
import floor_broad_adapter as broad

roles = broad.roles
rec = broad.rec
D = rec.D
OUT = rec.ROOT/'wheelleg_warp/results/paper_recovery_20261004/phase_support_qualification_v1'
COL = ['valid', 'step', 'current_command', 'zero_command', 'tracking_mean', 'guard_lower']
ARMS = ('original', 'floor_only', 'phase_support')


@wp.kernel
def select_anchor(slot: int, phase: int, command: wp.array[D], active: wp.array[int],
                  reference: wp.array2d[D], role: wp.array3d[D], monitor: wp.array3d[D]):
    w = wp.tid()
    for j in range(6):
        monitor[slot, w, j] = D(0)
    if active[w] == 0:
        return
    zero = command[w] == D(0)
    if phase:
        anchor = D(.115)
        if zero:
            anchor = wp.min(reference[w, 2], D(.160))
        reference[w, 15] = anchor
        role[slot, w, 5] = anchor
    monitor[slot, w, 0] = D(1)
    monitor[slot, w, 1] = role[slot, w, 1]
    monitor[slot, w, 2] = command[w]
    monitor[slot, w, 3] = D(int(zero))
    monitor[slot, w, 4] = reference[w, 2]
    monitor[slot, w, 5] = role[slot, w, 5]


def check_log(data, arm):
    assert arm in ARMS and data.ndim == 2 and data.shape[1] == len(COL)
    assert np.isfinite(data).all() and np.all(data[:, 0] == 1)
    np.testing.assert_array_equal(data[:, 1], np.arange(1, len(data)+1))
    np.testing.assert_array_equal(data[:, 3], data[:, 2] == 0)
    anchor = np.minimum(data[:, 4], .160)
    if arm == 'floor_only':
        anchor = np.full(len(data), .115)
    elif arm == 'phase_support':
        anchor = np.where(data[:, 2] == 0, anchor, .115)
    np.testing.assert_array_equal(data[:, 5], anchor)


def prepare_step(raw, slot, arm):
    """Same preparation as capture, also usable in non-integrating source checks."""
    assert arm in ARMS
    ref, log = raw._role_buffers
    mode = int(arm == 'floor_only')
    wp.launch(roles.prepare, raw.num_envs, [slot, mode, raw.k['reference'], raw.data.qpos,
        raw.data.sensordata, raw.ids, raw.k['state'], raw.state, raw.active, ref, log])
    wp.launch(select_anchor, raw.num_envs, [slot, int(arm == 'phase_support'), raw.command,
        raw.active, ref, log, raw._phase_buffer])


def instrument(cases, native_mode, arm='phase_support', directory=None):
    if arm not in ARMS:
        raise ValueError('Unknown phase-support arm')
    monitor = wp.zeros((40, len(cases), len(COL)), dtype=D)
    launch = wp.launch
    command = None
    prepared = None
    events = []
    def intercept(kernel, dim, inputs=None, **kwargs):
        nonlocal command, prepared
        if kernel is rec.old.command_step:
            command = inputs[2]
            events.append(('command_step', rec.old.signature(command)))
        if kernel is roles.prepare:
            assert command is not None and events[-1][0] == 'command_step'
            result = launch(kernel, dim, inputs, **kwargs)
            slot, ref, log = inputs[0], inputs[9], inputs[10]
            launch(select_anchor, dim, [slot, int(arm == 'phase_support'), command,
                                       inputs[8], ref, log, monitor])
            prepared = (rec.old.signature(command), rec.old.signature(ref))
            events.append(('prepare', slot, *prepared))
            return result
        if kernel is roles.experimental.control_physical_nominal:
            assert prepared == (rec.old.signature(inputs[4]), rec.old.signature(inputs[12]))
            assert events[-1][0] == 'prepare'
            events.append(('control', *prepared))
        return launch(kernel, dim, **kwargs) if inputs is None else launch(kernel, dim, inputs, **kwargs)
    wp.launch = intercept
    try:
        # Original mode has unchanged tracking and permits dynamic anchor logging.
        # Its inherited checker verifies mean/force invariants; check_log adds the phase rule.
        raw = broad.instrument(cases, native_mode, 'floor_only' if arm == 'floor_only' else 'original', directory)
    finally:
        wp.launch = launch
    assert len(events) == 240 and events[:120] == events[120:]
    assert all(events[i][0] == ('command_step', 'prepare', 'control')[i % 3] for i in range(240))
    raw._phase_buffer = monitor
    raw._phase_chunks = [[] for _ in cases]
    raw._phase_frozen = set()
    raw._phase_topology = dict(verified=True, arm=arm, identical_command_prepare_control_events=True,
        current_command_pointer_matches_control=True, calls_per_capture=40,
        inherited_role_mode=int(arm == 'floor_only'), columns=COL, model_or_default_source_changed=False)
    raw._role_topology.update(arm=arm, phase_anchor=arm == 'phase_support',
        inherited_checker='Mean/force invariants; phase stream also checks actual anchor against current command')
    raw._broad_contract.update(arm=arm, base_role_arm='floor_only' if arm == 'floor_only' else 'original')
    if directory is not None:
        for name, value in [('phase_contract.json', raw._phase_topology),
                            ('role_contract.json', raw._role_topology), ('broad_contract.json', raw._broad_contract)]:
            rec.atomic_json(directory/name, value)
    reset, wait = raw.reset, raw.step_wait
    def reset_all():
        result = reset(); monitor.zero_(); raw._phase_frozen.clear()
        for chunk in raw._phase_chunks:
            chunk.clear()
        return result
    def step_wait():
        result = wait()
        frames = raw._complete_buffers[0].numpy()
        phases = monitor.numpy(); role = raw._role_buffers[1].numpy()
        for w in range(raw.num_envs):
            if w in raw._phase_frozen:
                continue
            keep = frames[:, w, 0] == 1
            selected = phases[keep, w].copy()
            np.testing.assert_array_equal(selected[:, :2], frames[keep, w, :2])
            np.testing.assert_array_equal(selected[:, 2], frames[keep, w, 43])
            np.testing.assert_array_equal(selected[:, 4], role[keep, w, 3])
            np.testing.assert_array_equal(selected[:, 5], role[keep, w, 5])
            if len(selected):
                raw._phase_chunks[w].append(selected)
            if result[2][w]:
                data = np.concatenate(raw._phase_chunks[w]); check_log(data, arm)
                assert len(data) == result[3][w]['physical_steps']
                if directory is not None:
                    file = directory/f'case_phase_{raw._broad_contract["recording_ids"][w]}.npz'
                    assert not file.exists()
                    np.savez_compressed(file, trace=data, columns=np.array(COL))
                    result[3][w]['phase_trace'] = dict(path=file.name, sha256=rec.sha(file), rows=len(data))
                raw._phase_frozen.add(w); raw._phase_chunks[w].clear()
        return result
    raw.reset, raw.step_wait = reset_all, step_wait
    return raw


def preserve_phase(raw, directory):
    """Supplement the inherited full/gyro/role interruption saver without replay."""
    file = directory/'interrupted_phase_last_buffer.npz'
    np.savez_compressed(file, trace=raw._phase_buffer.numpy(), columns=np.array(COL))
    prefixes = {}
    for w, chunk in enumerate(raw._phase_chunks):
        if chunk:
            tag = raw._broad_contract['recording_ids'][w]
            data = np.concatenate(chunk)
            np.savez_compressed(directory/f'interrupted_phase_prefix_{tag}.npz', trace=data, columns=np.array(COL))
            prefixes[str(tag)] = len(data)
    return dict(last_buffer_sha256=rec.sha(file), prefix_rows=prefixes,
                complete_case_ids=[raw._broad_contract['cases'][w] for w in sorted(raw._phase_frozen)])


def freeze():
    from test_phase_support_probe import check
    assert not (OUT/'source_contract.json').exists()
    p = json.loads((OUT/'proposal.json').read_text())
    parent = rec.ROOT/p['parent_source_contract']
    assert rec.sha(parent) == p['parent_source_contract_sha256']
    sources = dict(json.loads(parent.read_text())['source_sha256'])
    assert all(rec.sha(rec.ROOT/n) == s for n, s in sources.items())
    roles.check_control_copy()
    check()
    for name in ('phase_support_probe.py', 'test_phase_support_probe.py'):
        sources['wheelleg_warp/'+name] = rec.sha(rec.ROOT/'wheelleg_warp'/name)
    rec.atomic_json(OUT/'source_contract.json', dict(verified=True,
        proposal_sha256=rec.sha(OUT/'proposal.json'), parent_source_contract_sha256=rec.sha(parent),
        source_sha256=sources, unit_sha256=rec.sha(OUT/'unit.json'), new_evaluations=0, training_updates=0,
        status='Phase source qualified; runner/admission/reuse bindings still required before328 queue.'))
    print('PHASE SOURCE QUALIFIED; 0 physics evaluations/learning; 328 queue not run', flush=True)


if __name__ == '__main__':
    freeze()
