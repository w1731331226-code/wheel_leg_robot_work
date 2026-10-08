"""One frozen two-construction GPU check; never trains or retries."""
import dataclasses
import enum
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
import execution_history_evaluation as evaluation
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/baseline_repeatability_v1'
SOLVER = ('qacc_warmstart', 'qacc', 'qfrc_constraint', 'solver_niter')


def flatten(value, name, arrays, metadata):
    if isinstance(value, wp.array):
        arrays[name] = value.numpy().copy()
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            flatten(getattr(value, field.name), name + '.' + field.name, arrays, metadata)
    elif isinstance(value, (tuple, list)):
        metadata[name + '.length'] = len(value)
        for i, item in enumerate(value):
            flatten(item, name + '.' + str(i), arrays, metadata)
    elif isinstance(value, enum.Enum):
        metadata[name] = dict(type=type(value).__name__, value=value.value)
    elif value is None or isinstance(value, (bool, int, float, str)):
        metadata[name] = value
    else:
        raise TypeError(f'Unrecorded simulator field {name}: {type(value)}')


def raw_arrays(value, name, arrays):
    if isinstance(value, wp.array):
        arrays[name] = value.numpy().copy()
    elif isinstance(value, np.ndarray):
        arrays[name] = value.copy()
    elif isinstance(value, dict):
        for key, item in value.items():
            raw_arrays(item, name + '.' + str(key), arrays)
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            raw_arrays(item, name + '.' + str(i), arrays)


def snapshot(raw, route, history, directory):
    arrays, metadata = {}, {}
    flatten(raw.model, 'model', arrays, metadata)
    flatten(raw.data, 'data', arrays, metadata)
    for label, obj in (('raw', raw), ('route', route), ('history', history)):
        raw_arrays(vars(obj), label, arrays)
    assert 'data.qacc_warmstart' in arrays and 'data.contact.geom' in arrays
    np.savez_compressed(directory / 'initial.npz', **arrays)
    atomic_json(directory / 'initial_metadata.json', metadata)
    mujoco.mj_saveModel(raw.cpu, str(directory / 'cpu_model.mjb'), None)
    return arrays, metadata


def differences(left, right):
    assert set(left) == set(right), 'State schema differs'
    return [name for name in left if left[name].shape != right[name].shape or
            left[name].dtype != right[name].dtype or left[name].tobytes() != right[name].tobytes()]


def construct(cases, directory):
    original = mjw.step
    calls, buffers = [], {}
    def copy(slot, data, prefix):
        for name in SOLVER:
            source = getattr(data, name)
            key = prefix + name
            if key not in buffers:
                buffers[key] = wp.zeros((40, *source.shape), dtype=source.dtype, device=source.device)
            wp.copy(buffers[key][slot], source)
    def record(model, data, *args, **kwargs):
        slot = len(calls) % 40
        calls.append((id(model), id(data)))
        copy(slot, data, 'pre_')
        result = original(model, data, *args, **kwargs)
        copy(slot, data, 'post_')
        return result
    mjw.step = record
    try:
        result = evaluation.make_env(cases, 'B0', None, directory)
    finally:
        mjw.step = original
    raw = result[0]
    assert calls == [(id(raw.model), id(raw.data))] * 80
    assert raw._execution_topology['control_steps'] == 40
    raw._repeatability_buffers = buffers
    return result, dict(captured_integrations=80, captures=2, integrations_per_capture=40,
                       same_model_data_owner=True, original_step_restored=mjw.step is original)


def run():
    assert not any((OUT / name).exists() for name in ('started.json', 'completion.json', 'failure.json'))
    p = json.loads((OUT / 'proposal.json').read_text())
    assert p['total_world_steps'] == 1600 and p['independent_constructions'] == 2
    assert len(p['cases']) == 20 and p['cases'][13]['seed'] == 6300013
    assert all(sha(ROOT / name) == digest for name, digest in p['sources'].items())
    # Bitwise comparison must detect a one-ULP difference without fitting tolerance.
    a = np.zeros((1, 2)); b = a.copy(); b[0, 1] = np.nextafter(0., 1.)
    assert differences({'a': a}, {'a': b}) == ['a'] and not differences({'a': a}, {'a': a})
    runtime = dict(python=platform.python_version(), packages={name: importlib.metadata.version(name)
        for name in ('mujoco', 'mujoco-warp', 'warp-lang', 'numpy', 'torch')},
        GPU=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip())
    atomic_json(OUT / 'started.json', dict(proposal_sha256=sha(OUT / 'proposal.json'),
        runner_sha256=sha(__file__), runtime=runtime, budget_world_steps=1600))
    envs, snapshots, reports = [], [], []
    fd_calls, integrated = [], 0
    original_fd = mujoco.mjd_transitionFD
    def fd(*args, **kwargs):
        fd_calls.append(1)
        return original_fd(*args, **kwargs)
    mujoco.mjd_transitionFD = fd
    start = time.perf_counter()
    try:
        for i in range(2):
            directory = OUT / f'construction_{i}'
            directory.mkdir(exist_ok=False)
            result, report = construct(p['cases'], directory)
            envs.append(result)
            raw, route, history, _, env = result
            env.reset()
            snapshots.append(snapshot(raw, route, history, directory))
            reports.append(report)
            assert not raw.state.numpy()[:, 0].any() and not raw.data.time.numpy().any()
            print('CONSTRUCTED', i, 'full initial fields', len(snapshots[-1][0]), flush=True)
        mismatch = differences(snapshots[0][0], snapshots[1][0])
        metadata_equal = snapshots[0][1] == snapshots[1][1]
        cpu_equal = sha(OUT / 'construction_0/cpu_model.mjb') == sha(OUT / 'construction_1/cpu_model.mjb')
        admitted = not mismatch and metadata_equal and cpu_equal
        atomic_json(OUT / 'source_admission.json', dict(verified=True, reports=reports,
            full_initial_array_fields=len(snapshots[0][0]), initial_array_differences=mismatch,
            initial_metadata_equal=metadata_equal, cpu_model_binary_equal=cpu_equal,
            physical_prefix_admitted=admitted, baseline_constructor_transitionFD_calls=len(fd_calls),
            physical_world_steps=0, proposal_sha256=sha(OUT / 'proposal.json')))
        if not admitted:
            atomic_json(OUT / 'completion.json', dict(verified=True, round=296,
                outcome='inconclusive_full_initial_state_not_equal', physical_world_steps=0,
                initial_array_differences=mismatch, initial_metadata_equal=metadata_equal,
                cpu_model_binary_equal=cpu_equal, implicit_retry=False, formal_PPO_admitted=False,
                conclusion='Recorded initial qvctrl equality alone is insufficient. Full initial simulator/controller fields differ before integration; preserve and review which fields are active rather than zeroing or excluding them to force equality.'))
            print('STOP296 initial qualification differs', mismatch, flush=True)
            return
        for i, result in enumerate(envs):
            raw, _, _, _, env = result
            output = env.step(np.zeros((20, 6), np.float32))
            integrated += 800
            assert not output[2].any()
            np.testing.assert_array_equal(raw.state.numpy()[:, 0], 40)
            np.testing.assert_allclose(raw.data.time.numpy(), .02, rtol=0, atol=1e-8)
            trace, _, _, _, pre, post, meta, contacts = raw._complete_buffers
            np.testing.assert_array_equal(trace.numpy()[:, :, 0], 1)
            values = dict(trace=trace.numpy(), pre=pre.numpy(), post=post.numpy(), meta=meta.numpy(), contacts=contacts.numpy())
            values.update({name: value.numpy() for name, value in raw._repeatability_buffers.items()})
            np.savez_compressed(OUT / f'construction_{i}/prefix.npz', **values)
            print('INTEGRATED', i, 'world_steps', integrated, flush=True)
        assert integrated == 1600
        with np.load(OUT / 'construction_0/prefix.npz') as x, np.load(OUT / 'construction_1/prefix.npz') as y:
            changed = differences(dict(x), dict(y))
        atomic_json(OUT / 'completion.json', dict(verified=True, round=296, outcome='two_prefixes_complete',
            physical_world_steps=integrated, prefix_differences=changed, identical_recorded_prefixes=not changed,
            baseline_constructor_transitionFD_calls=len(fd_calls), wall_seconds=time.perf_counter() - start,
            new_training_samples=0, new_optimizer=0, source_admission_sha256=sha(OUT / 'source_admission.json'),
            runner_sha256=sha(__file__), formal_PPO_admitted=False,
            limits='Two current20ms prefixes only, not historical cause, full task stability or method contribution. 297 independent finite review before changing paired experimental design.'))
        print('DONE296 two prefixes', changed, flush=True)
    except BaseException as error:
        for i, result in enumerate(envs):
            raw = result[0]
            np.savez_compressed(OUT / f'construction_{i}/partial_solver.npz',
                **{name: value.numpy() for name, value in raw._repeatability_buffers.items()})
        atomic_json(OUT / 'failure.json', dict(error=repr(error), physical_world_steps=integrated,
            baseline_constructor_transitionFD_calls=len(fd_calls), implicit_retry=False))
        raise
    finally:
        mujoco.mjd_transitionFD = original_fd
        for result in envs:
            result[-1].close()


if __name__ == '__main__':
    run()
