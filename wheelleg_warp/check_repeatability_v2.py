"""Compare two fresh GPU terrain banks under identical zero M3 actions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'wheelleg_ppo/tools'))
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from training_contract import TASK_CONTRACT_VERSION
PROTOCOL = ROOT / 'wheelleg_warp/results/contract_v2_baseline_checked_20260923/protocol.json'
PHYSICS_SOURCES = ('wheelleg_warp/training_contract.py', 'wheelleg_warp/native/terrain.py',
                   'wheelleg_warp/native/terrain_env.py', 'wheelleg_warp/native/models.py',
                   'wheelleg_warp/native/environment.py', 'wheelleg_warp/native/controller.py',
                   'wheelleg_ppo/tools/ppo_env.py', 'wheelleg_ppo/tools/wheelleg_sim.py',
                   'wheelleg_ppo/tools/hardware_profile.py', 'wheelleg_ppo/tools/rm_controller.py',
                   'wheelleg_ppo/tools/model_lqr.py', 'wheelleg_ppo/tools/state_estimation.py',
                   'wheelleg_ppo/xml/wheelleg.xml')
def load_cases():
    raw = PROTOCOL.read_bytes()
    protocol = json.loads(raw)
    rows = protocol['panels']
    if protocol['task_contract_version'] != TASK_CONTRACT_VERSION or protocol['backend'] != 'MuJoCo Warp GPU':
        raise ValueError('Task contract or backend changed')
    if len(rows) != 160 or protocol['world_orders'][0] != list(range(160)):
        raise ValueError('Frozen 160-world order changed')
    if len({r['scenario']['terrain_seed'] for r in rows}) != 160: raise ValueError('Duplicate frozen terrain seed')
    for source in PHYSICS_SOURCES:
        expected = protocol['source_sha256'][source]
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: ' + source)
    return [TerrainScenario(**r['scenario']) for r in rows], hashlib.sha256(raw).hexdigest()
def initial_arrays(env):
    arrays = {name: getattr(env.data, name).numpy() for name in
              ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'time', 'sensordata', 'geom_xpos', 'geom_xmat')}
    arrays.update({name: getattr(env, name).numpy() for name in
                   ('history', 'state', 'targets', 'obs', 'residual', 'active', 'done')})
    arrays['control_history'] = env.k['state'].numpy()
    arrays['model_meaninertia'] = env.model.stat.meaninertia.numpy()
    arrays['model_geom_pos'] = env.model.geom_pos.numpy()
    return arrays
def delta(a, b):
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(np.max(difference)) if np.isfinite(difference).all() else None
def contacts(env):
    count = int(env.data.nacon.numpy()[0])
    worlds = env.data.contact.worldid.numpy()[:count]
    geoms = env.data.contact.geom.numpy()[:count]
    pairs = [[] for _ in range(env.num_envs)]
    for world, geom in zip(worlds, geoms):
        if 0 <= world < env.num_envs:
            pairs[int(world)].append((int(geom[0]), int(geom[1])))
    return pairs
def contact_difference(a, b, cases, step):
    for world, (left, right) in enumerate(zip(a, b)):
        if left != right:
            canonical = lambda pairs: Counter(tuple(sorted(pair)) for pair in pairs)
            return dict(policy_step=step, world=world, seed=cases[world].terrain_seed,
                        counts=[len(left), len(right)], order_equal=False,
                        pair_multisets_equal=canonical(left) == canonical(right),
                        ordered_pairs=[left, right])
    return None
def run(output, max_steps):
    if not 1 <= max_steps <= 500:
        raise ValueError('max_steps must be 1..500')
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    cases, digest = load_cases()
    a = TerrainEnv(len(cases), scenario=cases)
    b = None
    try:
        b = TerrainEnv(len(cases), scenario=cases)
        a.reset(); b.reset()
        left, right = initial_arrays(a), initial_arrays(b)
        initial = {name: delta(left[name], right[name]) for name in left
                   if not np.array_equal(left[name], right[name])}
        sensor_a, sensor_b = left['sensordata'], right['sensordata']
        sensor_differences = np.argwhere(sensor_a != sensor_b)
        gyro = int(a.ids.numpy()[10])
        if gyro != int(b.ids.numpy()[10]): raise ValueError('Controller gyro address changed')
        gyro_a, gyro_b = sensor_a[:, gyro:gyro + 3], sensor_b[:, gyro:gyro + 3]
        gyro_changed = np.flatnonzero(np.any(gyro_a != gyro_b, axis=1))
        gyro_worlds = list(dict.fromkeys([int(w) for w in gyro_changed] +
                                      [int(w) for w in sensor_differences[:, 0]]))[:10] or [0]
        critical = {name: value for name, value in initial.items() if name != 'sensordata'}
        if len(gyro_changed):
            critical['controller_gyro'] = delta(gyro_a, gyro_b)
        result = dict(protocol_sha256=digest, task_contract_version=TASK_CONTRACT_VERSION,
                      cases=len(cases), action='zero M3 residual', max_policy_steps=max_steps,
                      scope='first episode only; step_wait is not called, so completed worlds stay inactive',
                      initial_mismatch=initial, initial_critical_mismatch=critical,
                      initial_sensor_difference=dict(total_elements=len(sensor_differences),
                          examples=[dict(world=int(w), sensor_index=int(j),
                                         values=[float(sensor_a[w, j]), float(sensor_b[w, j])],
                                         abs_delta=float(abs(sensor_a[w, j] - sensor_b[w, j])))
                                    for w, j in sensor_differences[:10]],
                          controller_gyro_start=gyro,
                          controller_gyro_triplets=[dict(world=w, values=[gyro_a[w].tolist(), gyro_b[w].tolist()])
                                                    for w in gyro_worlds]),
                      first_contact_difference=None,
                      first_state_difference=None, policy_steps_run=0)
        if not critical:
            action = np.zeros((len(cases), 3), dtype=np.float32)
            for step in range(1, max_steps + 1):
                a.step_async(action); b.step_async(action)
                result['policy_steps_run'] = step
                if result['first_contact_difference'] is None:
                    result['first_contact_difference'] = contact_difference(contacts(a), contacts(b), cases, step)
                q0, q1 = a.data.qpos.numpy(), b.data.qpos.numpy()
                v0, v1 = a.data.qvel.numpy(), b.data.qvel.numpy()
                d0, d1 = a.done.numpy(), b.done.numpy()
                mismatch = np.any(q0 != q1, axis=1) | np.any(v0 != v1, axis=1) | (d0 != d1)
                if mismatch.any():
                    world = int(np.flatnonzero(mismatch)[0])
                    result['first_state_difference'] = dict(policy_step=step, world=world,
                        seed=cases[world].terrain_seed, qpos_max_abs=delta(q0[world], q1[world]),
                        qvel_max_abs=delta(v0[world], v1[world]), done=[int(d0[world]), int(d1[world])],
                        qpos_delta=(q0[world]-q1[world]).tolist(),
                        qvel_delta=(v0[world]-v1[world]).tolist(),
                        worlds_different=int(np.count_nonzero(mismatch)))
                    break
        with output.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(output, 'steps:', result['policy_steps_run'], 'diverged:', bool(result['first_state_difference']))
    finally:
        a.close()
        if b is not None:
            b.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=500)
    args = parser.parse_args()
    run(args.output, args.max_steps)
