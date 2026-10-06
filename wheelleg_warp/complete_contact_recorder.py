"""Dense readonly first-episode state/contact recording on the frozen Native graph."""
import gc
import inspect
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

import task_mode_recorder as old
from task_mode_recorder import D, ROOT, sha, atomic_json

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/complete_contact_witness_v2'
CONTACT_COL = ['valid', 'world', 'geom0', 'geom1', 'distance', 'dim']
CONTACT_COL += [f'point_{a}' for a in 'xyz']
CONTACT_COL += [f'frame_{i}{j}' for i in range(3) for j in range(3)]
CONTACT_COL += [f'friction_{i}' for i in range(5)]
CONTACT_COL += [f'local_{s}_{a}' for s in ('force', 'torque') for a in 'xyz']
assert len(CONTACT_COL) == 29


@wp.kernel
def state_copy(slot: int, q: wp.array2d[float], v: wp.array2d[float],
               effort: wp.array2d[float], trace: wp.array3d[D], out: wp.array3d[D]):
    w = wp.tid()
    if trace[slot, w, 0] == D(0):
        return
    for i in range(q.shape[1]):
        out[slot, w, i] = D(q[w, i])
    for i in range(v.shape[1]):
        out[slot, w, q.shape[1] + i] = D(v[w, i])
    for i in range(effort.shape[1]):
        out[slot, w, q.shape[1] + v.shape[1] + i] = D(effort[w, i])


@wp.kernel
def contact_copy(slot: int, ncon: wp.array[int], world: wp.array[int], geom: wp.array[wp.vec2i],
                 dist: wp.array[float], pos: wp.array[wp.vec3], frame: wp.array[wp.mat33],
                 friction: wp.array[vec5], dim: wp.array[int], address: wp.array2d[int],
                 force: wp.array2d[float], njmax: int, cone: int,
                 trace: wp.array3d[D], out: wp.array3d[D]):
    j = wp.tid()
    out[slot, j, 0] = D(0)
    if ncon[0] > out.shape[1] or j >= ncon[0]:
        return
    w = world[j]
    if w < 0 or w >= trace.shape[1]:
        return
    if trace[slot, w, 0] == D(0):
        return
    out[slot, j, 0] = D(1)
    out[slot, j, 1] = D(w)
    out[slot, j, 2] = D(geom[j][0])
    out[slot, j, 3] = D(geom[j][1])
    out[slot, j, 4] = D(dist[j])
    out[slot, j, 5] = D(dim[j])
    f = contact_force_fn(cone, frame, friction, dim, address, force, njmax, ncon, w, j, False)
    for i in range(3):
        out[slot, j, 6 + i] = D(pos[j][i])
        for k in range(3):
            out[slot, j, 9 + 3*i + k] = D(frame[j][i, k])
    for i in range(5):
        out[slot, j, 18 + i] = D(friction[j][i])
    for i in range(6):
        out[slot, j, 23 + i] = D(f[i])


@wp.kernel
def contact_counts(slot: int, ncon: wp.array[int], world: wp.array[int], capacity: int,
                   overflow: wp.array[int], root_body: int, com: wp.array2d[wp.vec3],
                   trace: wp.array3d[D], meta: wp.array3d[D]):
    w = wp.tid()
    if trace[slot, w, 0] == D(0):
        return
    meta[slot, w, 0] = D(ncon[0])
    meta[slot, w, 1] = D(0)
    meta[slot, w, 2] = D(overflow[w])
    for i in range(3):
        meta[slot, w, 3+i] = D(com[w, root_body][i])
    if ncon[0] > capacity:
        meta[slot, w, 2] = D(-1)
        return
    # ponytail: serial per-world count for this 20-world diagnostic; index if scaled.
    for j in range(ncon[0]):
        if world[j] == w:
            meta[slot, w, 1] += D(1)


def validate(data):
    """One dense first episode, with every raw contact accounted for."""
    trace, pre, post, meta, contacts, steps = data
    n = len(trace)
    assert n and np.array_equal(trace[:, 1], np.arange(1, n+1))
    np.testing.assert_allclose(trace[:, 2], np.arange(n)*.0005, atol=1e-9, rtol=0)
    np.testing.assert_allclose(trace[:, 3], np.arange(1, n+1)*.0005, atol=1e-9, rtol=0)
    assert pre.shape == post.shape and len(pre) == len(meta) == n
    assert meta.shape[1] == 6 and not meta[:, 2].any()
    assert len(contacts) == len(steps) and contacts.shape[1] == len(CONTACT_COL)
    assert np.all((steps >= 1) & (steps <= n))
    assert np.array_equal(np.bincount(steps, minlength=n+1)[1:], meta[:, 1])
    for array in (trace, pre, post, meta, contacts):
        assert np.isfinite(array).all()


def instrument(factory, cases, mode, directory=None):
    original, rebuilt = [], []
    current = original
    launch, step = wp.launch, mjw.step
    watched = {old.begin, old.command_step, old.control_physical_nominal, old.reduce_contacts,
               old.collect_physical, old.after, old.update_shared_reference}

    def spy(kernel, dim, inputs=None, **kwargs):
        if kernel in watched:
            current.append((kernel.key, repr(dim), tuple(old.signature(x) for x in inputs), repr(sorted(kwargs.items()))))
        return launch(kernel, dim, **kwargs) if inputs is None else launch(kernel, dim, inputs, **kwargs)

    def spy_step(model, data, **kwargs):
        current.append(('mjw.step', id(model), id(data), repr(sorted(kwargs.items()))))
        return step(model, data, **kwargs)

    wp.launch, mjw.step = spy, spy_step
    try:
        raw = factory(cases, mode)
    finally:
        wp.launch, mjw.step = launch, step
    n = raw.num_envs
    assert raw.data.ctrl.shape[1] == raw.data.actuator_force.shape[1]
    _, boxes = old.geometry(raw)
    trace = wp.zeros((40, n, old.WIDTH), dtype=D)
    mask = wp.ones(n, dtype=int)
    count = wp.zeros((n, 2), dtype=D)
    # Keep every step through the existing post kernel; no geometry or force censoring.
    window = wp.array(np.tile([-1e30, 1e30], (n, 1)), dtype=D)
    width = raw.cpu.nq + raw.cpu.nv + raw.cpu.nu
    pre = wp.zeros((40, n, width), dtype=D)
    post = wp.zeros((40, n, width), dtype=D)
    meta = wp.zeros((40, n, 6), dtype=D)
    contacts = wp.zeros((40, raw.data.naconmax, len(CONTACT_COL)), dtype=D)
    raw._complete_buffers = trace, mask, count, window, pre, post, meta, contacts
    root = int(raw.cpu.body_rootid[raw.cpu.geom_bodyid[raw.wheel_geom_ids[0]]])
    assert root > 0 and raw.cpu.body_rootid[raw.cpu.geom_bodyid[raw.wheel_geom_ids[1]]] == root
    protected = [raw.data.qpos, raw.data.qvel, raw.data.ctrl, raw.diag, raw.targets, raw.param,
                 raw.state, raw.k['state'], raw.k['reference'], raw.nominal_correction, raw.active]
    saved = [a.numpy().copy() for a in protected]
    current = rebuilt
    wp.launch, mjw.step = spy, spy_step
    try:
        with wp.ScopedCapture() as capture:
            wp.launch(old.begin, n, [raw.reward])
            for slot in range(40):
                if raw.shared_reference_enabled and slot % 10 == 0:
                    wp.launch(old.update_shared_reference, n, raw.shared_reference_args)
                wp.launch(old.command_step, n, [raw.state,raw.param,raw.command,raw.active,raw.data.qpos,raw.data.qvel,raw.data.qacc_warmstart,raw.stopped_q,raw.stopped_v,raw.stopped_w,raw.contact_flags])
                wp.launch(raw.control_kernel, n, [raw.data.qpos,raw.data.qvel,raw.data.sensordata,raw.targets,raw.command,raw.active,raw.k['state'],raw.ids,raw.k['heights'],raw.k['gains'],raw.k['feed'],raw.k['angles'],raw.k['reference'],raw.k['yaw'],raw.data.ctrl,raw.diag,int(raw.project_clipped_base),int(raw.grouped_residual)]+raw.control_extra, block_dim=32)
                launch(old.pre, n, [slot,raw.data.qpos,raw.data.qvel,raw.ids,raw.wheel_offsets,raw.active,mask,raw.state,raw.k['state'],raw.k['reference'],raw.param,raw.targets,raw.diag,raw.data.ctrl,raw.command,raw.control_extra[0],raw.nominal_correction,count,trace])
                launch(state_copy, n, [slot,raw.data.qpos,raw.data.qvel,raw.data.ctrl,trace,pre])
                mjw.step(raw.model, raw.data)
                launch(old.post, n, [slot,raw.data.qpos,raw.data.qvel,raw.ids,raw.wheel_offsets,raw.physical_args[8],raw.data.actuator_force,window,raw.param,count,trace])
                launch(state_copy, n, [slot,raw.data.qpos,raw.data.qvel,raw.data.actuator_force,trace,post])
                c = raw.data.contact
                args = [slot,raw.data.nacon,c.worldid,c.geom,c.dist,c.pos,c.frame,c.friction,c.dim,c.efc_address,raw.data.efc.force,raw.data.njmax,raw.model.opt.cone]
                launch(contact_copy, raw.data.naconmax, [*args,trace,contacts])
                launch(contact_counts, n, [slot,raw.data.nacon,c.worldid,raw.data.naconmax,raw.data.overflow,root,raw.data.subtree_com,trace,meta])
                launch(old.contact_record, n, [*args,raw.ids,raw.model.geom_bodyid,raw.model.body_rootid,raw.data.cvel,raw.data.subtree_com,trace])
                wp.launch(old.reduce_contacts, raw.data.naconmax, [raw.data.nacon,c.worldid,c.geom,raw.ids,raw.contact_flags])
                if raw.height_safety == 'physical_v1':
                    wp.launch(old.collect_physical, n, raw.physical_args)
                wp.launch(old.after, n, [raw.data.qpos,raw.data.qvel,raw.data.sensordata,raw.data.qacc_warmstart,raw.data.time,raw.contact_flags,raw.ids,raw.param,raw.command,raw.state,raw.k['state'],raw.diag,raw.residual,raw.active,raw.done,raw.reward,raw.obs,raw.history,raw.stopped_q,raw.stopped_v,raw.stopped_w,raw.wheel_offsets], block_dim=32)
        raw.graph = capture.graph
    finally:
        wp.launch, mjw.step = launch, step
    assert original == rebuilt and sum(x[0] == 'mjw.step' for x in original) == 40
    assert sum(x[0] == old.update_shared_reference.key for x in original) == 4
    for a, b in zip(protected, saved):
        np.testing.assert_array_equal(a.numpy(), b)
    gc.collect()
    assert raw._complete_buffers[0] is trace
    wait, reset = raw.step_wait, raw.reset
    chunks = [[] for _ in cases]
    frozen = set()
    # The runner can preserve consumed partial episodes on interruption without rerunning.
    raw._complete_chunks, raw._complete_frozen = chunks, frozen

    def reset_all():
        result = reset()
        for a in raw._complete_buffers:
            if a is not window:
                a.zero_()
        mask.fill_(1)
        frozen.clear()
        for x in chunks:
            x.clear()
        return result

    def step_wait():
        result = wait()
        frames, ps, qs, ms, cs = [a.numpy() for a in (trace, pre, post, meta, contacts)]
        tracking = mask.numpy()
        for w in range(n):
            if w in frozen:
                continue
            keep = frames[:, w, 0] == 1
            take = (cs[:, :, 0] == 1) & (cs[:, :, 1] == w)
            slots = np.nonzero(take)[0]
            assert not ms[keep, w, 2].any()
            assert np.array_equal(np.bincount(slots, minlength=40)[keep], ms[keep, w, 1])
            if keep.any():
                chunks[w].append((frames[keep,w].copy(), ps[keep,w].copy(), qs[keep,w].copy(),
                                  ms[keep,w].copy(), cs[take].copy(), frames[slots,w,1].astype(np.int64)))
            if result[2][w]:
                data = tuple(np.concatenate([x[i] for x in chunks[w]]) for i in range(6))
                validate(data)
                assert len(data[0]) == result[3][w]['physical_steps']
                result[3][w]['complete_counts'] = dict(steps=len(data[0]), contacts=len(data[4]))
                if directory is not None:
                    file = directory / f'case_{cases[w]["seed"]}.npz'
                    assert not file.exists()
                    np.savez_compressed(file, trace=data[0], pre=data[1], post=data[2], meta=data[3],
                                        contacts=data[4], contact_steps=data[5], columns=np.array(old.COL),
                                        contact_columns=np.array(CONTACT_COL), state_sizes=np.array([raw.cpu.nq,raw.cpu.nv,raw.cpu.nu]))
                    result[3][w]['complete_trace'] = dict(path=file.name, sha256=sha(file), **result[3][w]['complete_counts'])
                tracking[w] = 0
                frozen.add(w)
                chunks[w].clear()
        mask.assign(tracking)
        return result

    raw.reset, raw.step_wait = reset_all, step_wait
    raw._complete_topology = dict(verified=True, original_calls=len(original), physics_calls=40,
        shared_reference_calls=4, input_pointer_and_scalar_identity=True, protected_arrays_unchanged=True,
        buffer_owner=True, no_window_censoring=True, contact_capacity=raw.data.naconmax, state_sizes=[raw.cpu.nq,raw.cpu.nv,raw.cpu.nu])
    if directory is not None:
        assert not (directory / 'geometry.json').exists()
        atomic_json(directory / 'geometry.json', dict(boxes=boxes, case_ids=[x['seed'] for x in cases],
            root_body=root, wheel_geom_ids=raw.wheel_geom_ids.tolist(), geom_body_ids=raw.cpu.geom_bodyid.tolist(),
            geom_names=[mujoco.mj_id2name(raw.cpu,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(raw.cpu.ngeom)],
            timing='State pre/post bracketing integration. Contact/frame/local force and subtree COM are solver pre-integration quantities retained by mjw.step; frame rows map local force to world by local @ frame. Force acts on geom1, opposite on geom0.'))
    return raw


def evaluate(cases, arm, classical, directory):
    factory = old.evaluator.raw_env
    old.evaluator.raw_env = lambda cases, mode: instrument(factory, cases, mode, directory)
    try:
        return old.evaluator.evaluate(cases, arm, classical=classical)
    finally:
        old.evaluator.raw_env = factory


def freeze():
    from test_complete_contact_recorder import check
    assert not (OUT / 'source_contract.json').exists()
    proposal = json.loads((OUT / 'proposal.json').read_text())
    assert sha(ROOT / proposal['parent_proposal']) == proposal['parent_proposal_sha256']
    parent = old.OUT / 'source_contract.json'
    sources = dict(json.loads(parent.read_text())['source_sha256'])
    assert all(sha(ROOT / name) == value for name, value in sources.items())
    check()
    for name in ('complete_contact_recorder.py','test_complete_contact_recorder.py'):
        sources['wheelleg_warp/' + name] = sha(ROOT / 'wheelleg_warp' / name)
    atomic_json(OUT / 'source_contract.json', dict(verified=True, proposal_sha256=sha(OUT / 'proposal.json'),
        parent_source_contract_sha256=sha(parent), source_sha256=sources, unit_sha256=sha(OUT / 'unit.json'),
        contact_api_sha256=sha(Path(inspect.getfile(contact_force_fn.func))), columns=old.COL,
        contact_columns=CONTACT_COL, new_evaluations=0, training_updates=0,
        status='Recorder source admitted; execution requires separately frozen runner and40 episode accounting.'))
    print('SOURCE ADMITTED; 0 evaluation episodes/learning updates', flush=True)


if __name__ == '__main__':
    freeze()
