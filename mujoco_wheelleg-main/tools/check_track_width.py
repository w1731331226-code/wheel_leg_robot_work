"""硬件名义模型的轮距敏感性检查；不修改 XML 或正式控制参数。

同质量/车架惯量对比，只回答加宽的几何与侧撞收益；新增横梁重量、
安装空间和横坡跟随能力仍须 CAD 与地形试验验证。
"""
import math
from pathlib import Path

import mujoco
import numpy as np

import wheelleg_sim as sim


XML = str(Path(__file__).resolve().parents[1] / 'xml' / 'wheelleg_dm8009.xml')


def model_at_width(width):
    if not math.isfinite(width) or not 0.108 <= width <= 0.30:
        raise ValueError('轮距敏感性范围为 0.108～0.30 m')
    model, data = sim.load_model(XML, True, width)
    # ponytail: 固定车架质量和本体惯量的敏感性对照；定型时用加宽后 CAD 属性替换。
    # 两条闭环链同时平移，子连杆和轮组件随动，保留真实六执行器。
    left = data.xpos[model.body('wheelL').id]
    right = data.xpos[model.body('wheelR').id]
    assert abs(abs(left[1] - right[1]) - width) < 1e-9
    assert model.nu == 6
    assert abs(sum(model.body_mass) - sim.hw.DESIGN_MASS) < 1e-9
    return model, data


def run(width, speed, impulse):
    model, data = model_at_width(width)
    state = sim.St(True)
    chassis = model.body('chassis').id
    peak_roll = peak_pitch = peak_yaw = 0.0
    com_height = None
    last_unsettled = 3.05
    for step in range(round(5.6 / model.opt.timestep)):
        t = step * model.opt.timestep
        state.cmd_vel = speed if t > 1.0 else 0.0
        data.xfrc_applied[chassis] = 0.0
        if 3.0 <= t < 3.05:
            data.xfrc_applied[chassis, 1] = impulse / 0.05
        if com_height is None and t >= 3.0:
            com_height = float(data.subtree_com[chassis, 2])
        sim.control(model, data, state)
        assert np.isfinite(data.ctrl).all()
        assert (data.ctrl >= model.actuator_ctrlrange[:, 0] - 1e-9).all()
        assert (data.ctrl <= model.actuator_ctrlrange[:, 1] + 1e-9).all()
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        if t >= 3.0:
            roll, pitch, yaw = map(abs, sim.euler(data))
            peak_roll, peak_pitch, peak_yaw = (
                max(peak_roll, roll), max(peak_pitch, pitch), max(peak_yaw, yaw))
            if t >= 3.05 and roll > math.radians(3):
                last_unsettled = t
    peaks = tuple(map(math.degrees, (peak_roll, peak_pitch, peak_yaw)))
    recovery = max(0.0, last_unsettled - 3.05)
    within = max(peaks) <= 15.0 and recovery <= 2.0
    print(f'width={width:.3f}m v={speed:+.1f} impulse={impulse:+.1f}Ns '
          f'r/p/y={peaks[0]:.2f}/{peaks[1]:.2f}/{peaks[2]:.2f}deg '
          f'roll_recovery={recovery:.2f}s envelope={"PASS" if within else "FAIL"}',
          flush=True)
    return com_height


if __name__ == '__main__':
    for width in (0.108, 0.150, 0.180, 0.210):
        for speed in (0.0, 1.0):
            for impulse in (-1.0, 1.0):
                height = run(width, speed, impulse)
        slope = math.radians(8)
        print(f'geometry width={width:.3f}m COM={height:.4f}m '
              f'rigid_tilt_limit={math.degrees(math.atan(width / (2 * height))):.1f}deg '
              f'8deg_margin={width / 2 - height * math.tan(slope):.4f}m '
              f'8deg_leg_difference={width * math.tan(slope):.4f}m', flush=True)
    print('检查完成；envelope 仅为姿态/roll恢复门，不代表复杂地面或实机验收。')
