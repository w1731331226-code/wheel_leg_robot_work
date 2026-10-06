"""World-Z root-body torque after control, before integration; no policy feature."""
import json
from pathlib import Path
import numpy as np
import warp as wp
import mujoco_warp as mjw
import phase_support_probe as phase
from native.terrain import HeightTerrainScenario, model

rec = phase.rec
D = rec.D
OUT = rec.ROOT/'wheelleg_warp/results/paper_recovery_20261004/equal_exposure_recovery_v1'
COL = ['valid', 'step', 'pre_s', 'profile', 'body', 'desired_tau_z_Nm']
COL += ['submitted_'+s for s in ('fx_N', 'fy_N', 'fz_N', 'tx_Nm', 'ty_Nm', 'tz_Nm')]


@wp.kernel
def set_wrench(slot: int, root: int, state: wp.array2d[D], active: wp.array[int],
               indices: wp.array[int], desired: wp.array2d[D], samples: wp.array2d[float],
               applied: wp.array2d[wp.spatial_vector], monitor: wp.array3d[D]):
    w = wp.tid()
    for b in range(applied.shape[1]):
        applied[w, b] = wp.spatial_vector()
    for j in range(12):
        monitor[slot, w, j] = D(0)
    if active[w] == 0:
        return
    index = int(state[w, 0])-5000
    p = indices[w]
    raw = D(0); value = float(0)
    if p >= 0 and index >= 0 and index < 200:
        raw = desired[p, index]; value = samples[p, index]
    applied[w, root] = wp.spatial_vector(0., 0., 0., 0., 0., value)
    monitor[slot, w, 0] = D(1); monitor[slot, w, 1] = state[w, 0]+D(1)
    monitor[slot, w, 2] = state[w, 0]*D(.0005); monitor[slot, w, 3] = D(p)
    monitor[slot, w, 4] = D(root); monitor[slot, w, 5] = raw
    for j in range(6):
        monitor[slot, w, 6+j] = D(applied[w, root][j])


@wp.kernel
def clear_rows(mask: wp.array[int], applied: wp.array2d[wp.spatial_vector]):
    w, b = wp.tid()
    if mask[w]:
        applied[w, b] = wp.spatial_vector()


def profiles():
    p = json.loads((OUT/'proposal.json').read_text())
    file = rec.ROOT/p['stimulus']['profile_file']
    assert rec.sha(file) == p['stimulus']['profile_sha256']
    with np.load(file, allow_pickle=False) as z:
        values = z['yaw_moment_Nm']
    assert values.shape == (8, 200) and np.isfinite(values).all()
    return values


def check_samples(data, profile, root, values):
    assert data.ndim == 2 and data.shape[1] == len(COL) and np.isfinite(data).all()
    np.testing.assert_array_equal(data[:, 0], 1)
    np.testing.assert_array_equal(data[:, 3], profile)
    np.testing.assert_array_equal(data[:, 4], root)
    np.testing.assert_allclose(data[:, 2], (data[:, 1]-1)*.0005, rtol=0, atol=1e-12)
    desired = np.zeros(len(data))
    k = data[:, 1].astype(int)-1-5000
    if profile >= 0:
        selected = (k >= 0) & (k < 200); desired[selected] = values[profile, k[selected]]
    np.testing.assert_array_equal(data[:, 5], desired)
    np.testing.assert_array_equal(data[:, 6:11], 0)
    np.testing.assert_array_equal(data[:, 11], desired.astype(np.float32).astype(np.float64))


def instrument(cases, mode, directory=None):
    registered = {c['seed']: c for c in json.loads((OUT/'proposal.json').read_text())['cases']}
    if not cases or any(c != registered.get(c['seed']) for c in cases):
        raise ValueError('Use exactly the registered operational cases')
    indices = np.array([-1 if c['profile'] == 'zero' else c['profile'] for c in cases], np.int32)
    values = profiles(); desired = wp.array(values, dtype=D); submitted = wp.array(values.astype(np.float32))
    tags = wp.array(indices); monitor = wp.zeros((40, len(cases), len(COL)), dtype=D)
    cpu = model(HeightTerrainScenario(**cases[0]['scenario']))
    root = int(cpu.body_rootid[cpu.geom_bodyid[cpu.geom('wheel_collide_L').id]])
    assert root > 0 and root == cpu.body_rootid[cpu.geom_bodyid[cpu.geom('wheel_collide_R').id]]
    launch, step = wp.launch, mjw.step
    state = active = slot = None; ready = False; events = []
    def spy_launch(kernel, dim, inputs=None, **kwargs):
        nonlocal state, active, slot, ready
        if kernel is rec.old.command_step:
            state, active = inputs[0], inputs[3]
        result = launch(kernel, dim, **kwargs) if inputs is None else launch(kernel, dim, inputs, **kwargs)
        if kernel is phase.roles.finish:
            slot = inputs[0]; ready = True
        return result
    def spy_step(m, d, **kwargs):
        nonlocal ready
        if ready:
            assert state is not None and active is not None
            assert d.xfrc_applied.dtype == wp.spatial_vector and d.xfrc_applied.shape == (len(cases), cpu.nbody)
            launch(set_wrench, len(cases), [slot, root, state, active, tags, desired, submitted, d.xfrc_applied, monitor])
            events.append((slot, rec.old.signature(state), rec.old.signature(active), rec.old.signature(d.xfrc_applied)))
            ready = False
        return step(m, d, **kwargs)
    wp.launch, mjw.step = spy_launch, spy_step
    try:
        raw = phase.instrument(cases, mode, directory=directory)
    finally:
        wp.launch, mjw.step = launch, step
    assert len(events) == 80 and events[:40] == events[40:]
    assert [e[0] for e in events[:40]] == list(range(40))
    assert raw.cpu.nbody == cpu.nbody and int(raw.cpu.body_rootid[raw.cpu.geom_bodyid[raw.wheel_geom_ids[0]]]) == root
    raw._force_buffers = (tags, desired, submitted, monitor)
    raw._force_chunks = [[] for _ in cases]; raw._force_frozen = set()
    raw._force_root = root
    raw._force_topology = dict(verified=True, root_body=root, modes=mode, profiles=indices.tolist(),
        calls_per_capture=40, identical_force_slots_and_input_owners=True, after_control_before_step=True,
        columns=COL, source_profile_sha256=rec.sha(OUT/'profiles.npz'),
        force_order='fx,fy,fz,tx,ty,tz at body COM, global Cartesian frame; pure torque tz',
        actor_features_added=0, nominal_controller_inputs_added=0)
    if directory is not None:
        rec.atomic_json(directory/'force_contract.json', raw._force_topology)
    mask = wp.zeros(len(cases), dtype=int)
    reset, wait = raw.reset, raw.step_wait
    def reset_all():
        result = reset(); raw.data.xfrc_applied.zero_(); monitor.zero_(); mask.zero_()
        raw._force_frozen.clear()
        for chunk in raw._force_chunks:
            chunk.clear()
        return result
    def step_wait():
        result = wait(); frames = raw._complete_buffers[0].numpy(); logs = monitor.numpy()
        for w in range(raw.num_envs):
            if w in raw._force_frozen: continue
            keep = frames[:, w, 0] == 1; selected = logs[keep, w].copy()
            np.testing.assert_array_equal(selected[:, :2], frames[keep, w, :2])
            np.testing.assert_allclose(selected[:, 2], frames[keep, w, 2], atol=1e-12, rtol=0)
            if len(selected): raw._force_chunks[w].append(selected)
            if result[2][w]:
                data = np.concatenate(raw._force_chunks[w])
                np.testing.assert_array_equal(data[:, 1], np.arange(1, len(data)+1))
                check_samples(data, int(indices[w]), root, values)
                assert len(data) == result[3][w]['physical_steps']
                delivery = int(np.count_nonzero((data[:, 1]-1 >= 5000) & (data[:, 1]-1 < 5200))) if indices[w] >= 0 else 0
                result[3][w]['force_delivery'] = dict(profile=int(indices[w]), samples=delivery,
                    expected_samples=200 if indices[w] >= 0 else 0, complete=delivery == (200 if indices[w] >= 0 else 0))
                if directory is not None:
                    file = directory/f'case_force_{cases[w]["seed"]}.npz'; assert not file.exists()
                    np.savez_compressed(file, trace=data, columns=np.array(COL))
                    result[3][w]['force_trace'] = dict(path=file.name, sha256=rec.sha(file), rows=len(data))
                raw._force_frozen.add(w); raw._force_chunks[w].clear()
        if result[2].any():
            mask.assign(np.asarray(result[2], np.int32))
            wp.launch(clear_rows, raw.data.xfrc_applied.shape, [mask, raw.data.xfrc_applied])
        return result
    raw.reset, raw.step_wait = reset_all, step_wait
    return raw


def preserve_force(raw, directory):
    np.savez_compressed(directory/'interrupted_force_last_buffer.npz', trace=raw._force_buffers[3].numpy(),
                        applied=raw.data.xfrc_applied.numpy(), columns=np.array(COL))
    for w, chunks in enumerate(raw._force_chunks):
        if chunks:
            np.savez_compressed(directory/f'interrupted_force_prefix_{raw._broad_contract["cases"][w]}.npz',
                                trace=np.concatenate(chunks), columns=np.array(COL))
    return dict(complete_case_ids=[raw._broad_contract['cases'][w] for w in sorted(raw._force_frozen)],
                prefix_rows=[sum(len(x) for x in chunks) for chunks in raw._force_chunks])


def freeze():
    from test_equal_exposure_probe import check
    assert not (OUT/'source_contract.json').exists()
    p = json.loads((OUT/'proposal.json').read_text())
    parent = phase.OUT/'source_contract.json'
    sources = dict(json.loads(parent.read_text())['source_sha256'])
    assert all(rec.sha(rec.ROOT/n) == s for n, s in sources.items())
    check()
    for name in ('equal_exposure_probe.py', 'test_equal_exposure_probe.py'):
        sources['wheelleg_warp/'+name] = rec.sha(rec.ROOT/'wheelleg_warp'/name)
    rec.atomic_json(OUT/'source_contract.json', dict(verified=True, source_sha256=sources,
        proposal_sha256=rec.sha(OUT/'proposal.json'), profile_sha256=rec.sha(OUT/'profiles.npz'),
        parent_source_sha256=rec.sha(parent), unit_sha256=rec.sha(OUT/'unit.json'),
        new_evaluations=0, training_updates=0, status='Force source qualified;225 runner not frozen/run.'))
    print('SOURCE QUALIFIED worldZ force/zero/native6/39/RMS/streams;0 task evaluations or training', flush=True)


if __name__ == '__main__':
    freeze()
