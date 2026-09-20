"""独立 MJWarp 基线；原 CPU 控制/模型保持不变，不是 GPU PPO 训练器。"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'wheelleg_ppo/tools'))
import mujoco
import mujoco.rollout
import mujoco_warp as mjw
import numpy as np
import warp as wp
from ppo_env import Scenario, build_model, Residual
import wheelleg_sim as sim


@wp.kernel
def replay_control(ctrl: wp.array2d[float], trace: wp.array2d[float], tick: wp.array[int]):
    world, actuator = wp.tid()
    ctrl[world, actuator] = trace[tick[0], actuator]


@wp.kernel
def advance(tick: wp.array[int]):
    tick[0] = tick[0] + 1


class WarpPhysics:
    """物理在 GPU；校对时保留原 Python 控制器，逐步回读完整 MjData。"""
    def __init__(self, model, data, worlds=1):
        self.model = mjw.put_model(model)
        self.data = mjw.put_data(model, data, nworld=worlds, nconmax=64, njmax=256)
        with wp.ScopedCapture() as capture:
            mjw.step(self.model, self.data)
        self.graph = capture.graph

    def step(self, model, data):
        self.data.ctrl.assign(np.asarray(data.ctrl[None, :], dtype=np.float32))
        wp.capture_launch(self.graph)
        mjw.get_data_into(data, model, self.data)
        # MJWarp 3.12只回填contact.geom；MuJoCo 3.12的旧字段不再共享存储。
        # 原控制器wheel_contact使用geom1/geom2，必须在唯一回读入口同步。
        data.contact.geom1[:] = data.contact.geom[:, 0]
        data.contact.geom2[:] = data.contact.geom[:, 1]
        if np.any(self.data.overflow.numpy()):
            raise RuntimeError('GPU 接触或约束容量溢出')


def initial(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe('stand').id)
    mujoco.mj_forward(model, data)
    return data


def closed_loop(case, seconds):
    scenario = Scenario(height_l=.02, mu_l=.6, mu_r=1.) if case == 'asymmetric' else Scenario()
    model = build_model(scenario)
    cpu, gpu = initial(model), initial(model)
    state = sim.make_state(model, True, True)
    states = [state, copy.deepcopy(state)]
    residuals = [Residual(), Residual()]
    physics = WarpPhysics(model, gpu)
    steps = round(seconds / model.opt.timestep)
    peak_error = np.zeros(3)
    peak_angles = np.zeros((2, 3))
    touched = [set(), set()]
    wall = np.zeros(2)
    trace = []
    for i in range(steps):
        for k, data in enumerate((cpu, gpu)):
            start = time.perf_counter()
            states[k].cmd_vel = (scenario.speed * min(1., max(0., (i * model.opt.timestep - 1.)))
                                 if case == 'asymmetric' else 0.)
            # 已见开发场景固定时刻刹车；不使用研究保留集或按结果选场景。
            if i * model.opt.timestep >= 4.5:
                states[k].cmd_vel = 0.
            states[k].cmd_jump = case == 'jump' and 2. < i * model.opt.timestep < 2.05
            sim.control(model, data, states[k], residuals[k])
            if k == 0:
                trace.append(data.ctrl.copy())
                mujoco.mj_step(model, data)
            else:
                physics.step(model, data)
            wall[k] += time.perf_counter() - start
            if not np.isfinite(np.r_[data.qpos, data.qvel, data.ctrl]).all():
                raise FloatingPointError('非有限状态')
            peak_angles[k] = np.maximum(peak_angles[k], np.abs(sim.euler(data)))
            for contact in data.contact:
                for geom in (contact.geom1, contact.geom2):
                    name = model.geom(geom).name
                    if name.startswith('bump_'):
                        touched[k].add(name)
        peak_error = np.maximum(peak_error, [np.max(abs(cpu.qpos - gpu.qpos)),
                                            np.max(abs(cpu.qvel - gpu.qvel)),
                                            np.max(abs(cpu.ctrl - gpu.ctrl))])
    result = dict(case=case, scenario=asdict(scenario), seconds=steps * model.opt.timestep,
                  max_abs_error=dict(zip(('qpos', 'qvel', 'ctrl'), peak_error.tolist())),
                  peak_angles_rad=peak_angles.tolist(), final_xyz=[cpu.qpos[:3].tolist(), gpu.qpos[:3].tolist()],
                  contacts=[sorted(t) for t in touched], cpu_wall_s=wall[0], gpu_with_host_controller_wall_s=wall[1])
    # 工程对齐阈值，在首次运行前固定；不是论文成功率或逐位相等承诺。
    result['aligned'] = bool(peak_error[0] <= .02 and peak_error[1] <= .5 and peak_error[2] <= 2.
                             and touched[0] == touched[1])
    return result, model, np.asarray(trace, dtype=np.float32)


def benchmark(model, trace, worlds, steps):
    """同模型/初态/CPU控制轨迹，比较 GPU 与8线程原生 CPU rollout 的物理吞吐。"""
    trace = trace[:steps]
    steps = len(trace)
    seed = initial(model)
    physics = WarpPhysics(model, seed, worlds)
    controls = wp.array(trace, dtype=wp.float32)
    tick = wp.zeros(1, dtype=wp.int32)
    with wp.ScopedCapture() as capture:
        wp.launch(replay_control, (worlds, model.nu), [physics.data.ctrl, controls, tick])
        mjw.step(physics.model, physics.data)
        wp.launch(advance, 1, [tick])
    # 编译/分配不计入稳态吞吐。预热使用单独 data，测量从同一初态开始。
    wp.capture_launch(capture.graph)
    wp.synchronize()
    physics = WarpPhysics(model, seed, worlds)
    tick.zero_()
    with wp.ScopedCapture() as capture:
        wp.launch(replay_control, (worlds, model.nu), [physics.data.ctrl, controls, tick])
        mjw.step(physics.model, physics.data)
        wp.launch(advance, 1, [tick])
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    gpu_time = time.perf_counter() - start
    if np.any(physics.data.overflow.numpy()):
        raise RuntimeError('吞吐测试容量溢出')
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, seed, state, state_spec)
    states = np.repeat(state[None, :], worlds, axis=0)
    ctrl = np.broadcast_to(trace, (worlds, steps, model.nu)).astype(float).copy()
    data = [mujoco.MjData(model) for _ in range(8)]
    mujoco.rollout.rollout(model, data, states[:8], ctrl[:8])
    start = time.perf_counter()
    cpu_states, _ = mujoco.rollout.rollout(model, data, states, ctrl)
    cpu_time = time.perf_counter() - start
    qpos = physics.data.qpos.numpy()
    qvel = physics.data.qvel.numpy()
    if not np.isfinite(np.r_[qpos.ravel(), qvel.ravel()]).all():
        raise FloatingPointError('批量GPU状态无效')
    err = float(np.max(abs(qpos - cpu_states[:, -1, 1:1 + model.nq])))
    return dict(worlds=worlds, steps=steps, cpu_threads=8, gpu_seconds=gpu_time, cpu_seconds=cpu_time,
                gpu_steps_per_second=worlds * steps / gpu_time, cpu_steps_per_second=worlds * steps / cpu_time,
                speedup=cpu_time / gpu_time, final_qpos_max_error=err, aligned=err <= .02,
                scope='原生物理开环回放；不包含闭环控制/PPO，不代表端到端训练加速')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=6.)
    parser.add_argument('--worlds', type=int, default=256)
    parser.add_argument('--replay-steps', type=int, default=200)
    parser.add_argument('--cases', nargs='+', choices=['stand', 'asymmetric', 'jump'], default=['stand', 'asymmetric', 'jump'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < .001 or args.worlds < 8 or args.replay_steps < 1:
        parser.error('seconds必须有限且>=.001，worlds>=8，replay-steps>=1')
    reference = json.loads(Path(__file__).with_name('CPU_REFERENCE.json').read_text())
    changed = [p for p, h in reference['source_sha256'].items()
               if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    if changed:
        raise RuntimeError(f'原CPU参考源码已变化，需显式建立新配对基线：{changed}')
    args.output.mkdir(parents=True, exist_ok=False)
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError('必须使用CUDA，不允许静默回退CPU')
    wp.set_device('cuda:0')
    paths = list((ROOT / 'wheelleg_ppo/tools').glob('*.py')) + list((ROOT / 'wheelleg_ppo/xml').glob('*.xml'))
    paths += [Path(__file__), Path(__file__).with_name('requirements.txt')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(cpu_reference_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  versions=dict(mujoco=mujoco.__version__, mujoco_warp=mjw.__version__, warp=wp.__version__),
                  frozen_cpu_reference=reference['git_commit'],
                  device=str(wp.get_device()), source_sha256=hashes, closed_loop=[], benchmark=None,
                  tolerances=dict(qpos=.02, qvel=.5, ctrl=2.), status='running')
    def save():
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    save()
    try:
        for case in args.cases:
            result, model, trace = closed_loop(case, args.seconds)
            report['closed_loop'].append(result)
            save()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if report['benchmark'] is None:
                report['benchmark'] = benchmark(model, trace, args.worlds, args.replay_steps)
                save()
                print(json.dumps(report['benchmark'], ensure_ascii=False), flush=True)
        report['status'] = 'passed' if all(r['aligned'] for r in report['closed_loop']) and report['benchmark']['aligned'] else 'alignment_failed'
    except Exception as exc:
        report.update(status='error', error=repr(exc))
        raise
    finally:
        save()
    if report['status'] != 'passed':
        raise SystemExit('CPU/GPU校对超出固定阈值，保留失败结果；不得替换原CPU基线')


if __name__ == '__main__':
    main()
