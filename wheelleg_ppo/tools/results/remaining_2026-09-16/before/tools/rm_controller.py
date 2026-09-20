"""RoboMaster 思路的本机六状态 LQR + VMC 名义仿真候选。

只保留六个 LQR 反馈状态；腿长/腿速由独立 VMC 支撑环处理。
从本机模型求增益，不复制开源机器人质量、符号或增益常数。
"""
import math
from functools import lru_cache

import mujoco
import numpy as np

import model_lqr as ml
import wheelleg_sim as sim
from state_estimation import leg_kinematics


def design_controller(model):
    """显式离线设计；随机物理模型不得隐式调用此函数。"""
    heights = np.array([sim.L_SQUAT_MIN, sim.L_PREP, sim.L_STAND, sim.L_MAX])
    table = []
    inputs = ml.sagittal_basis(model)[1]
    for height in heights:
        ref, a, b, _ = ml.design(model, height)
        gain, report = ml.reduced_design(model, ref, a, b, 6)
        if report['full_closed_radius'] >= 1:
            raise RuntimeError('六状态候选在完整本机模型不稳定')
        c, g, _ = ml.vmc_coordinates(ref)
        # 取出两路六状态反馈，第三路固定为独立的腿长 PD。
        k6 = (np.linalg.solve(g, gain)[:2] @ np.linalg.pinv(c[:6]))
        equilibrium_input = np.linalg.solve(g, np.linalg.pinv(inputs) @ ref.ctrl)
        fk = sim.fk_joints(ref.qpos[7], ref.qpos[10])
        k6.setflags(write=False)
        equilibrium_input.setflags(write=False)
        table.append((k6, equilibrium_input, fk['phi5'] + math.pi / 2))
    heights.setflags(write=False)
    return heights, tuple(table), float(sum(model.body_mass) / sim.hw.BASELINE_MASS), float(model.opt.timestep)


@lru_cache(maxsize=1)
def nominal_design():
    """每进程一次：固定名义模型的增益、前馈、平衡角和质量缩放。"""
    model, _ = sim.load_model(ml.XML, True)
    return design_controller(model)


class SixStateController:
    def __init__(self, model, design=None):
        self.heights, self.table, self.mass_scale, timestep = nominal_design() if design is None else design
        if not math.isclose(model.opt.timestep, timestep, rel_tol=0, abs_tol=1e-12):
            raise ValueError('物理步长与冻结控制器采样周期不一致')
        self.joints = np.array([[model.joint(n).id for n in names] for names in
                                [('alphaL', 'betaL'), ('alphaR', 'betaR')]])
        self.qadr, self.vadr = model.jnt_qposadr[self.joints], model.jnt_dofadr[self.joints]
        self.hip_actuators = np.array([[model.actuator(n).id for n in names] for names in
                                      [('motor_alphaL', 'motor_betaL'), ('motor_alphaR', 'motor_betaR')]])
        self.wheel_actuators = [model.actuator(n).id for n in ('motor_wheelL', 'motor_wheelR')]
        self.stop_xy = None

    def apply(self, model, data, state):
        """在原状态机完成计算后替换地面共模，保留其 roll/yaw 差模和跳跃控制。"""
        if state.jp not in ('DRIVE', 'LAND'):
            self.stop_xy = None
            return
        q, dq = data.qpos[self.qadr], data.qvel[self.vadr]
        legs = [leg_kinematics(q[i], dq[i]) for i in range(2)]
        lengths = np.array([np.linalg.norm(leg[0]) for leg in legs])
        rates = np.array([leg[3][:, 0] @ dq[i] for i, leg in enumerate(legs)])
        angle_rates = np.array([leg[3][:, 1] @ dq[i] for i, leg in enumerate(legs)])
        length = float(np.mean(lengths))
        index = int(np.clip(np.searchsorted(self.heights, length) - 1, 0, len(self.heights) - 2))
        ratio = float(np.clip((length - self.heights[index]) /
                              (self.heights[index + 1] - self.heights[index]), 0, 1))
        gain, feedforward, theta_eq = [(1 - ratio) * a + ratio * b
                                      for a, b in zip(self.table[index], self.table[index + 1])]
        roll, pitch, yaw = sim.euler(data)
        pitch_rate = float(data.sensor('body_gyro').data[1])
        theta = np.mean([math.atan2(leg[0][1], leg[0][0]) for leg in legs]) + math.pi / 2 - pitch
        velocity = sim.forward_component(data.qvel, yaw)  # 名义真值；影子估计待实测回标。
        if abs(state.cmd_vel) > 0.01 or abs(velocity) > 0.03:
            self.stop_xy = None
        elif self.stop_xy is None:
            self.stop_xy = data.qpos[:2].copy()
        position_error = (0.0 if self.stop_xy is None else
                          sim.forward_component(data.qpos[:2] - self.stop_xy, yaw))
        state6 = np.array([theta - theta_eq, np.mean(angle_rates) - pitch_rate,
                           position_error, velocity - state.cmd_vel, pitch, pitch_rate])
        wheel, hub = feedforward[:2] - gain @ state6
        offset = np.clip(sim.KP_R * roll + sim.KD_R * state.rw_f, -0.035, 0.035)
        force = feedforward[2] + self.mass_scale * (
            sim.KP_LEG * (state.L_cur + np.array([offset, -offset]) - lengths)
            - sim.KD_LEG * rates)
        # 使用当前关节构型执行 VMC；15 状态模型只在离线建模中出现。
        virtual = np.array([np.linalg.solve(leg[3], data.ctrl[self.hip_actuators[i]])
                            for i, leg in enumerate(legs)])
        # 分腿径向阻抗抑制落地后的相对伸缩；平均腿速反馈不提供差动阻尼。
        virtual[:, 0] = force
        virtual[:, 1] += hub - virtual[:, 1].mean()
        for i, leg in enumerate(legs):
            data.ctrl[self.hip_actuators[i]] = leg[3] @ virtual[i]
        # 旧 DRIVE 输出还混有低速轮阻尼及质量缩放；不能把它们当作纯航向差矩继承。
        data.ctrl[self.wheel_actuators] = [wheel + state.yaw_torque, wheel - state.yaw_torque]
        data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0],
                               model.actuator_ctrlrange[:, 1])
