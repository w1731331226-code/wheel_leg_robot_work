"""IMU/主动编码器/实际髋力矩的影子估计器；不接入正式控制。

初始要求水平静止，位置与航向取局部零点；平地滚动观测不适用于未知坡面。
髋力矩须来自实际电流反馈（仿真适配器使用 actuator_force），不能传命令力矩。
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation

import wheelleg_sim as sim


def leg_kinematics(q, dq):
    """只用两个主动编码器重建轮心位置/速度、A 链被动关节速度及极坐标雅可比。"""
    qa, qb = q
    phi1, phi4 = sim.PHI1_STAND - qa, sim.PHI4_STAND - qb
    b = sim.L1 * np.array([math.cos(phi1), math.sin(phi1)])
    d = np.array([sim.L5, 0]) + sim.L4 * np.array([math.cos(phi4), math.sin(phi4)])
    fk = sim.fk_joints(qa, qb)
    c = np.array([fk['xC'], fk['zC']])
    cb, cd = c - b, c - d
    db = sim.L1 * np.array([math.sin(phi1), -math.cos(phi1)])
    dd = sim.L4 * np.array([math.sin(phi4), -math.cos(phi4)])
    jac = np.linalg.solve(np.array([cb, cd]), np.diag([cb @ db, cd @ dd]))
    dc = jac @ dq
    shank = dc - db * dq[0]
    # 几何平面角的正向与 MuJoCo 的 +y 转轴相反。
    parent_rate = -(cb[0] * shank[1] - cb[1] * shank[0]) / (cb @ cb)
    leg = c - np.array([sim.L5 / 2, 0])
    polar = np.array([leg / fk['leg_len'], np.array([-leg[1], leg[0]]) / fk['leg_len']**2])
    return leg, dc, parent_rate, (polar @ jac).T


class Estimator:
    def __init__(self, mass, radius, track_width, wheel_mass=0.0, *,
                 hip_inertia=0.0, hip_damping=0.0, max_leg_rate=1.0):
        if not all(math.isfinite(v) and v > 0 for v in (mass, radius, track_width)):
            raise ValueError('质量、轮半径和轮距必须为有限正数')
        if not math.isfinite(wheel_mass) or not 0 <= wheel_mass < mass / 2:
            raise ValueError('单轮总成质量必须有限、非负且小于整机质量的一半')
        if (not all(math.isfinite(v) and v >= 0 for v in (hip_inertia, hip_damping))
                or not math.isfinite(max_leg_rate) or max_leg_rate <= 0):
            raise ValueError('转子惯量/阻尼必须有限非负，滚动观测的腿速上限必须有限正数')
        self.mass, self.radius, self.track_width = mass, radius, track_width
        self.wheel_mass = wheel_mass
        self.hip_inertia, self.hip_damping = hip_inertia, hip_damping
        self.max_leg_rate = max_leg_rate
        self.rotation = Rotation.identity()
        self.velocity = np.zeros(3)  # IMU 安装点的世界速度
        self.position = 0.0         # 基座原点的局部纵向里程计
        self.support = np.zeros(2)
        self.contact = np.zeros(2, dtype=bool)
        self.previous_vx = 0.0
        self.previous_relative_velocity = None
        self.previous_joint_velocity = None
        self.previous_acceleration = None

    def update(self, gyro, accel, q, dq, wheel_speed, hip_torque, dt):
        arrays = [np.asarray(v, dtype=float) for v in (gyro, accel, q, dq, wheel_speed, hip_torque)]
        shapes = [(3,), (3,), (2, 2), (2, 2), (2,), (2, 2)]
        if (not math.isfinite(dt) or not 0 < dt <= 0.02
                or any(v.shape != shape or not np.isfinite(v).all()
                       for v, shape in zip(arrays, shapes))):
            raise ValueError('无效传感器数据或采样间隔')
        gyro, accel, q, dq, wheel_speed, hip_torque = arrays
        self.rotation = self.rotation * Rotation.from_rotvec(gyro * dt)
        # 只有近静止且比力接近重力时，用加速度缓慢校正倾斜；航向不可由重力校正。
        if (max(abs(wheel_speed)) < 0.1 and np.max(abs(dq)) < 0.1
                and abs(np.linalg.norm(accel) - 9.81) < 0.3):
            up = self.rotation.apply(accel / np.linalg.norm(accel))
            correction = np.cross(up, [0, 0, 1])
            self.rotation = Rotation.from_rotvec(correction * dt / 0.5) * self.rotation
        acceleration = self.rotation.apply(accel) + np.array([0, 0, -9.81])
        # 当前测量属于 t，不能将 a(t)*dt 当作已经发生的 t→t+dt 运动。
        if self.previous_acceleration is not None:
            self.velocity += 0.5 * (self.previous_acceleration + acceleration) * dt
        self.previous_acceleration = acceleration
        joint_acceleration = (np.zeros_like(dq) if self.previous_joint_velocity is None
                              else (dq - self.previous_joint_velocity) / dt)
        self.previous_joint_velocity = dq.copy()
        candidates = []
        rolling = []
        relative_velocities = []
        for side, sign in enumerate((-1, 1)):  # XML 中 L 的 y 为负，不能按名称猜左右符号。
            leg, dc, parent_rate, jac = leg_kinematics(q[side], dq[side])
            p = np.array([leg[0], sign * self.track_width / 2, leg[1]])
            dp = np.array([dc[0], 0, dc[1]])
            relative = self.rotation.apply(np.cross(gyro, p) + dp)
            relative_velocities.append(relative)
            axle_acc = acceleration.copy()
            if self.previous_relative_velocity is not None:
                axle_acc += (relative - self.previous_relative_velocity[side]) / dt
            transmitted_torque = (hip_torque[side] - self.hip_inertia * joint_acceleration[side]
                                  - self.hip_damping * dq[side])
            radial, hub = np.linalg.solve(jac, transmitted_torque)
            length = np.linalg.norm(leg)
            leg_force = radial * leg / length + hub * np.array([-leg[1], leg[0]]) / length**2
            # 与开源支持力式一致：虚拟腿力投影 + 轮总成竖直惯性/重力。
            # ponytail: 尚省略连杆惯性/摩擦；动态残差不过门时补齐模型，不用命令力假装接触。
            force = (-self.rotation.apply([leg_force[0], 0, leg_force[1]])[2]
                     + self.wheel_mass * (axle_acc[2] + 9.81))
            self.support[side] += (force - self.support[side]) * (1 - math.exp(-dt / 0.01))
            threshold = (0.08 if self.contact[side] else 0.15) * self.mass * 9.81 / 2
            # 卸载立即禁止轮速更新，避免低通滤波把已离地的支撑力拖入腾空期。
            self.contact[side] = force > threshold and self.support[side] > threshold
            omega = self.rotation.apply(gyro + [0, parent_rate + wheel_speed[side], 0])
            axle_v = self.radius * np.cross(omega, [0, 0, 1])
            imu_v = axle_v - relative
            rolling.append(imu_v[0])
            # 快速摆腿时闭环柔性/角加速度误差使滚动约束不可靠；支撑存在不代表轮速可用。
            if (self.contact[side] and max(abs(dq[side])) <= self.max_leg_rate
                    and abs(imu_v[0] - self.velocity[0]) < 0.2
                    and abs(acceleration[0]) < 3.0):
                candidates.append(imu_v[0])
        self.previous_relative_velocity = np.array(relative_velocities)
        if candidates:
            self.velocity[0] += (np.mean(candidates) - self.velocity[0]) * (1 - math.exp(-dt / 0.03))
        # IMU 位于基座原点前 0.03m；控制里的 vx 对应基座原点。
        vx = self.velocity[0] - self.rotation.apply(np.cross(gyro, [0.03, 0, 0]))[0]
        self.position += 0.5 * (self.previous_vx + vx) * dt
        self.previous_vx = vx
        return dict(vx=float(vx), x=float(self.position), contact=self.contact.copy(),
                    support=self.support.copy(), rolling=np.array(rolling),
                    wheel_update=bool(candidates), quaternion=self.rotation.as_quat())
