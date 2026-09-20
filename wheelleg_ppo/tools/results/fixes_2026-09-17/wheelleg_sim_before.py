#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轮腿机器人整合仿真 (单文件): MuJoCo 可视化 + 键盘控制
========================================================================
键位 (按住生效):
  W / S        前进 1 m/s / 后退 1 m/s
  A / D        左转 / 右转
  ↑ / ↓        升高 / 降低身高 (名义工作下限 0.160m)
  7            跳跃 (小键盘7也可, 每按一次跳一次)
  ESC          退出
功能:
  1) 站立平衡 (位置保持环 + 轮速阻尼: 初始站立与停车后轮子均不抖, 漂移≈0)
  2) 1 m/s 行驶 (速度摆)
  3) 松开按键急停 (六状态协同制动, 腿前伸且平台保持水平)
  4) 身高调节 (范围 0.160~0.380m, 限速轨迹)
  5) 跳跃 (静止/带初速, COM 实际上升 0.15m, 全阶段平台恒平)
无头测试: python3 wheelleg_sim.py [--hardware] --test <stand|stop|jump|jumpfwd|jumpback>
========================================================================
"""
import numpy as np
import mujoco
import mujoco.viewer   # viewer 模块需显式导入
import math
import time
import os
import sys
import argparse

import hardware_profile as hw

# =========================================================================
# 五连杆 FK / IK（自包含）
# =========================================================================
L1 = L4 = .150
L2 = L3 = .270
L5 = .150
PHI1_STAND = -2.3873937057493047
PHI4_STAND = -0.7541989478404885


def fk_joints(q_alpha, q_beta):
    """关节角 -> 几何量 (phi = phi_stand - q)"""
    phi1 = PHI1_STAND - q_alpha
    phi4 = PHI4_STAND - q_beta
    xB, zB = L1 * math.cos(phi1), L1 * math.sin(phi1)
    xD, zD = L5 + L4 * math.cos(phi4), L4 * math.sin(phi4)
    BD = math.hypot(xD - xB, zD - zB)
    A0 = 2 * L2 * (xD - xB)
    B0 = 2 * L2 * (zD - zB)
    C0 = L2 * L2 + BD * BD - L3 * L3
    phi2 = 2 * math.atan2(B0 - math.sqrt(max(0.0, A0 * A0 + B0 * B0 - C0 * C0)), A0 + C0)
    xC, zC = xB + L2 * math.cos(phi2), zB + L2 * math.sin(phi2)
    phi5 = math.atan2(zC, xC - L5 / 2)
    leg_len = math.hypot(xC - L5 / 2, zC)
    return dict(xC=xC, zC=zC, phi5=phi5, leg_len=leg_len)


def ik(leg_len, xC_rel=0.0):
    """逆解: 轮心在髋中垂线 (站立支)"""
    xC, zC = L5 / 2 + xC_rel, -leg_len
    AC = math.hypot(xC, zC)
    CE = math.hypot(xC - L5, zC)
    psi_AC = math.atan2(zC, xC)
    psi_CE = math.atan2(zC, xC - L5)

    def clamp(v):
        return max(-1.0, min(1.0, v))

    g1 = math.acos(clamp((L1 * L1 + AC * AC - L2 * L2) / (2 * L1 * AC)))
    g4 = math.acos(clamp((L4 * L4 + CE * CE - L3 * L3) / (2 * L4 * CE)))
    return PHI1_STAND - (psi_AC - g1), PHI4_STAND - (psi_CE + g4)


# =========================================================================
# 控制参数 (DM8009/MF9025当前Python路径；历史C++控制器未迁移)
# =========================================================================
KP_LEG, KD_LEG = 500.0, 25.0
KP_R, KD_R = 0.30, 0.12
KP_T, KD_T = 4.0, 0.5
KP_P, KD_P = 6.0, 0.5
KP_HIP_LEVEL, KD_HIP_LEVEL = 4.0, 0.5
KD_W = 0.12                     # 轮速阻尼 (抑制粘滑微滑)
KD_WY = 0.02
KP_WZ = 0.05
# 急停 Standby: 停车后与初始站立保持同一套位置环，仅锁定停车点并清空积分
STOP_STANDBY_DELAY = 0.1
STOP_SPEED_EPS = 0.03
KP_Q, KD_Q = 0.8, 0.08
KP_Q_JUMP = 4.0
KD_Q_SQUAT = 0.10              # 1.0 会让四髋在 SQUAT 约 97% 时间撞 ±3 N·m 限幅
K_THA, KI_A = -0.3, 0.08
KP_P2 = 0.5
V_MAX, K_VMAX = 1.0, 1.0
JUMP_READY_HOLD = 0.25
JUMP_REQUEST_TIMEOUT = 3.00
JUMP_SPEED_TOL = 0.10
JUMP_SYNC_Q_TOL = 0.001
JUMP_SYNC_QD_TOL = 0.01
SQUAT_ABORT_ROLL = math.radians(3.0)
SQUAT_ABORT_YAW = math.radians(2.0)
TW_LIM, TW_DRIVE_RECOVERY_LIM, TH_LIM, F_LIM = 0.3, 0.23, 0.8, 40.0
L_STAND = 0.300
L_PREP = 0.250
L_SQUAT = 0.180                 # 跳跃下蹲深度 (非奇异, 有效蹬地)
L_SQUAT_MIN = 0.160             # 名义工作下限，保留折叠奇异位形余量
L_MAX = 0.380
L_JUMP = 0.400
T_SQUAT, T_PUSH, T_LAND = 0.20, 0.6, 0.5
PREP_LEG_RATE = 0.035           # 只用于待起跳预蹲；普通身高调节仍为 0.025 m/s
# 跳跃验收参数
JUMP_HEIGHT = 0.18
PLATFORM_PITCH_LIMIT = math.radians(5.0)
JUMP_YAW_LIMIT = math.radians(5.0)
LAND_IMPEDANCE_RATIO = 0.7                         # 落地阻抗刚度 = 70% 常规刚度
JUMP_HIP_TORQUE_LIM = 13.0                         # 蹬地内部上限，最终仍受 MJCF 限幅
JUMP_SPEED_TORQUE_REDUCTION = 2.0
JUMP_FORCE_K = 1575.0                              # 完整下蹲后的竖直蹬地力增益
JUMP_FORCE_SPEED_GAIN = 35.0                       # 仅按速度幅值补偿，前后严格镜像
KP_P_JUMP, KD_P_JUMP = 8.0, 0.8                   # 蹲跳/蹬地平台零俯仰反馈
KP_P_LAND, KD_P_LAND = 7.0, 0.3                   # 落地低阻尼，避免反弹放大
TW_SQUAT_LIM, TW_JUMP_LIM = 0.76, 1.00
TW_LAND_LIM = 1.00
LAND_SPEED_K = 2.50
KP_P_AIR, KD_P_AIR = 12.0, 2.0                    # 空中轮子反作用姿态环
TW_AIR_BASE = 1.00                                 # 腾空时允许轮电机全量反作用姿态力矩
AIR_HIP_TORQUE_LIM = 13.0
KP_HIP_SQUAT, KD_HIP_SQUAT = 12.0, 3.0             # 下蹲时髋部平台交叉补偿
KP_HIP_JUMP, KD_HIP_JUMP = 4.0, 1.0                # 蹬地时的小幅髋部姿态阻尼
SQUAT_HIP_TORQUE_LIM = 4.0
SQUAT_ROLL_FORCE_LIM = 4.0
SQUAT_ROLL_SPEED_MAX = 0.2
KP_ROLL_SQUAT, KD_ROLL_SQUAT = 150.0, 60.0
LAND_HIP_TORQUE_LIM = 3.0
# 平台恒平急停: 平台俯仰参考始终为 0，制动所需的质心偏移由腿前伸完成。
STOP_LEAN_BASE = 0.10                              # 低速时最小腿前伸角 rad
STOP_LEAN_GAIN = 0.22                              # 腿前伸角 / 速度 (rad/(m/s))
STOP_LEAN_MAX = 0.32                               # 腿前伸上限 rad
STOP_LEAN_RAMP = 0.10                              # 急停腿前伸渐入时间 s

# 矢状面六状态反馈。状态顺序:
#   [theta-theta_ref, theta_dot, x-x_ref, vx-vx_ref, pitch, pitch_dot]
# 输出 u = -K6 @ state = [T_wheel, T_hub]。位置/速度外环会按行驶、急停、
# 静止保持阶段选择参考值；平台 pitch_ref 恒为 0。
# 两个输出共 12 个系数，当前为站立腿长的仿真工作点增益。
K6_GAIN = np.array([
    [0.0, 0.0, -2.4, 1.20, -KP_P, -KD_P],
    [KP_T, KD_T, 0.0, 0.0, 0.0, 0.0],
])
K6_STOP_HIP_SCALE = 1.5                            # 急停时腿前伸/平台交叉反馈增强
K6_STOP_PITCH_SCALE = 10.0                         # 急停优先保持平台水平，腿前伸承担制动
STOP_HIP_TORQUE_LIM = 2.0                          # 腿前伸期髋电机力矩上限
K6_STOP_GAIN = K6_GAIN.copy()
K6_STOP_GAIN[0, 2] = 0.0                           # 高速制动时不拉回松键位置
K6_STOP_GAIN[0, 3] *= -1.0                         # 正轮矩产生 -x 加速度，速度反馈符号相反
K6_STOP_GAIN[0, 4:] *= K6_STOP_PITCH_SCALE
K6_STOP_GAIN[1] = np.array([KP_T, KD_T, 0.0, 0.0, -KP_T * KP_P2, 0.0]) * K6_STOP_HIP_SCALE


# =========================================================================
# 控制器状态 (跨帧)
# =========================================================================
class St:
    def __init__(self, hardware=False):
        self.hardware = hardware
        self.lqr6 = None
        self.yaw_torque = 0.0
        self.motor_peak_t = {}
        self.boot_t = 0.0
        # 键盘命令
        self.cmd_vel = 0.0        # W/S 按住 ±1.0
        self.cmd_turn = 0.0       # A/D 按住 ±1.5 rad/s
        self.leg_ref = 0.0        # ↑/↓ 身高增量
        self.cmd_jump = False     # 7 键单次触发脉冲，control() 读取后立即清零
        self.cmd_jump_prepare = False  # 只进入 PREP；cmd_jump 再提交起跳
        self.leg_dir = 0           # ↑/↓ 按住方向 (持续调身高)
        # 分开保存键的按住状态，避免 GLFW REPEAT 事件被误当成松键，
        # 也使同时按下相反方向时命令自然抵消。
        self.key_w = self.key_s = False
        self.key_a = self.key_d = False
        self.key_up = self.key_down = False
        self.ever_driven = False   # 是否按过行驶键 (停车后强锁轮)
        self.att_on = False        # 停车姿态死区迟滞状态
        self.yaw_target = 0.0     # 起跳时朝向 (跳跃期间回正)
        # 滤波
        self.L_prev = L_STAND
        self.th_prev = 0.0
        self.legd_f = 0.0
        self.thd_f = 0.0
        self.rw_f = 0.0
        self.pw_f = 0.0
        self.vf_f = 0.0
        self.yw_f = 0.0
        self.yaw_f = 0.0
        self.vel_int = 0.0       # 速度环/轮速积分项 (急停 standby 做积分分离)
        self.th_cmd = 0.0
        # 急停 Standby / 积分分离
        self.stop_standby = False
        self.low_speed_t = 0.0
        # 高度轨迹
        self.L_cur = L_STAND
        # 位置保持环
        self.pos_int = 0.0
        self.t_stop = -1.0
        self.xy_stop = np.zeros(2)
        # 跳跃状态机
        self.jp = 'DRIVE'
        self.jp_pending = False
        self.jp_preparing = False
        self.jp_committed = False  # PREP 已收到起跳提交，可在安全门通过后进 SQUAT
        self.jp_req_t = 0.0
        self.jp_ready_t = 0.0
        self.jp_cmd_keep = 0.0
        self.jp_L_keep = L_STAND
        self.jp_t0 = 0.0
        self.jp_v_keep = 0.0
        self.jp_was_air = False
        self.jp_takeoff_z = 0.0
        self.jp_peak_z = 0.0
        self.jp_squat_yaw = 0.0
        # 跳跃新增状态
        self.jp_qa_lock = 0.0
        self.jp_qb_lock = 0.0
        # 轮速 (供打印)
        self.ws1 = 0.0
        self.ws2 = 0.0


def euler(d):
    w, x, y, z = d.qpos[3:7]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def forward_component(vector, yaw):
    """世界系平面向量在当前车头方向的投影；不把横向速度送入纵向 LQR。"""
    return math.cos(yaw) * vector[0] + math.sin(yaw) * vector[1]


def wheel_contact(m, d):
    for i in range(d.ncon):
        c = d.contact[i]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if (n1 and 'wheel' in n1) or (n2 and 'wheel' in n2):
            return True
    return False


def vmc(qa, qb, F_leg, T_hub, vert):
    """VMC: F_leg(沿腿/竖直) + T_hub(轮毂力矩) -> 髋力矩 (数值雅可比)"""
    eps = 1e-6

    def c_at(a, b):
        r = fk_joints(a, b)
        return r['xC'], r['zC']

    xC, zC = c_at(qa, qb)
    r = fk_joints(qa, qb)
    L = r['leg_len']
    ux = (xC - L5 / 2) / L
    uz = zC / L
    x1, z1 = c_at(qa + eps, qb)
    x2, z2 = c_at(qa - eps, qb)
    x3, z3 = c_at(qa, qb + eps)
    x4, z4 = c_at(qa, qb - eps)
    J00 = (x1 - x2) / (2 * eps)
    J10 = (z1 - z2) / (2 * eps)
    J01 = (x3 - x4) / (2 * eps)
    J11 = (z3 - z4) / (2 * eps)
    Fx = 0.0 if vert else F_leg * ux       # 竖直推力: 无水平分量 (防起跳前倾注入)
    Fz = -F_leg if vert else F_leg * uz
    ta = J00 * Fx + J10 * Fz
    tb = J01 * Fx + J11 * Fz

    def th_at(a, b):
        return fk_joints(a, b)['phi5']

    gth_a = (th_at(qa + eps, qb) - th_at(qa - eps, qb)) / (2 * eps)
    gth_b = (th_at(qa, qb + eps) - th_at(qa, qb - eps)) / (2 * eps)
    return ta + T_hub * gth_a, tb + T_hub * gth_b


def lqr6_output(state, gain=K6_GAIN):
    """六状态、双输出状态反馈: [T_wheel, T_hub] = -K6*x。"""
    out = -gain @ state
    return float(out[0]), float(out[1])


# =========================================================================
# 主控制 (每步)
# =========================================================================
def control(m, d, st, residual=None):
    dt = m.opt.timestep
    prior_peak_times = st.motor_peak_t.copy()
    st.boot_t += dt
    mass_scale = hw.DESIGN_MASS / hw.BASELINE_MASS if st.hardware else 1.0
    if st.lqr6 is not None:
        mass_scale = st.lqr6.mass_scale
    grav = 4.0 * mass_scale * min(1.0, st.boot_t / 0.15)

    roll, pitch, yaw = euler(d)
    qaL, qbL = d.qpos[7], d.qpos[10]
    qaR, qbR = d.qpos[12], d.qpos[15]
    qdaL = d.qvel[m.jnt_dofadr[m.joint('alphaL').id]]
    qdbL = d.qvel[m.jnt_dofadr[m.joint('betaL').id]]
    qdaR = d.qvel[m.jnt_dofadr[m.joint('alphaR').id]]
    qdbR = d.qvel[m.jnt_dofadr[m.joint('betaR').id]]

    rL = fk_joints(qaL, qbL)
    rR = fk_joints(qaR, qbR)
    L = (rL['leg_len'] + rR['leg_len']) / 2
    th = (rL['phi5'] + rR['phi5']) / 2 + math.pi / 2 - pitch

    # ---------- 滤波 ----------
    legd = (L - st.L_prev) / dt
    st.legd_f += (legd - st.legd_f) * (dt / 0.01)
    st.L_prev = L
    thd = (th - st.th_prev) / dt
    st.thd_f += (thd - st.thd_f) * (dt / 0.01)
    st.th_prev = th
    gyro_adr = int(m.sensor('body_gyro').adr[0])
    rw = float(d.sensordata[gyro_adr])
    st.rw_f += (rw - st.rw_f) * (dt / 0.01)
    pw = float(d.sensordata[gyro_adr + 1])
    st.pw_f += (pw - st.pw_f) * (dt / 0.01)
    vx = forward_component(d.qvel, yaw)
    st.vf_f += (vx - st.vf_f) * (dt / 0.02)
    yw = float(d.sensordata[gyro_adr + 2])
    st.yw_f += (yw - st.yw_f) * (dt / 0.02)

    vel_cmd_raw = st.cmd_vel
    static_post_jump = (st.jp_takeoff_z > 0.0 and abs(st.jp_v_keep) < 0.05
                        and abs(vel_cmd_raw) < 0.01 and abs(st.cmd_turn) < 0.01)
    if st.jp == 'DRIVE' and abs(st.cmd_turn) > 0.01:
        st.yaw_target = yaw       # 仅在主动转向时更新航向，跳跃不得逐次锁入漂移
    yaw_error = math.atan2(math.sin(yaw - st.yaw_target), math.cos(yaw - st.yaw_target))
    if abs(vel_cmd_raw) >= 0.01:
        st.ever_driven = True

    # ---------- 急停计时 / 停车点冻结 ----------
    if abs(vel_cmd_raw) >= 0.01:
        st.t_stop = -1.0
    elif st.t_stop < 0.0:
        st.t_stop = st.boot_t
        st.xy_stop = d.qpos[:2].copy()
    # 停车后 (低速) 冻结参考点: 位置环不回位 -> 车轮不因回位而微转 (停车不抖)
    # 注意: 进入 standby 后不再跟随当前位置，否则位置环失效会造成缓慢溜车。
    if (st.jp == 'DRIVE' and abs(vel_cmd_raw) < 0.01 and abs(st.vf_f) < 0.05
            and st.low_speed_t < STOP_STANDBY_DELAY
            and not st.jp_pending
            and not (st.jp_takeoff_z > 0.0 and abs(st.jp_v_keep) < 0.05
                     and abs(st.cmd_turn) < 0.01)):
        st.xy_stop = d.qpos[:2].copy()
        st.yaw_target = yaw      # 停车时锁定朝向 (行驶直行轻回正, 防自然漂移)

    # ---------- 急停 Standby 判定 / 积分分离计时 ----------
    if abs(vel_cmd_raw) < 0.01:
        st.low_speed_t += dt
    else:
        st.low_speed_t = 0.0
        st.stop_standby = False
    # 锁轮逻辑仅在“急停 Standby”且不在跳跃/起跳相位时生效
    was_standby = st.stop_standby
    standby_base = (abs(vel_cmd_raw) < 0.01
                    and st.low_speed_t >= STOP_STANDBY_DELAY
                    and st.ever_driven and st.jp == 'DRIVE')
    st.stop_standby = standby_base and (was_standby or abs(st.vf_f) < STOP_SPEED_EPS)
    if st.stop_standby and not was_standby:
        # 进入 standby 的瞬间锁定当前位置，与初始站立一样由位置环保持，
        # 并清空速度环/位置环积分，避免停车前的积分残留造成漂移。
        st.xy_stop = d.qpos[:2].copy()
        st.pos_int = 0.0
        st.vel_int = 0.0
    # ---------- 跳跃状态机 ----------
    wz = d.qpos[2] + (rL['zC'] + rR['zC']) / 2
    contact = wheel_contact(m, d)
    airborne = (not contact) and wz > 0.03
    jp_el = st.boot_t - st.jp_t0
    req_now = st.cmd_jump
    prepare_now = st.cmd_jump_prepare
    st.cmd_jump = False            # 消费本次 7 键脉冲，下一次按 7 可再次触发
    st.cmd_jump_prepare = False
    if st.jp == 'DRIVE' and (prepare_now or req_now):
        if not st.jp_pending:
            st.jp_pending = True
            st.jp_preparing = False
            st.jp_req_t = st.boot_t
            st.jp_ready_t = 0.0
            st.jp_cmd_keep = vel_cmd_raw
            st.jp_L_keep = min(L_MAX, max(L_SQUAT_MIN, L_STAND + st.leg_ref))
        if req_now and not st.jp_committed:
            st.jp_committed = True
            st.jp_req_t = st.boot_t
            st.jp_ready_t = 0.0
    request_dynamic = abs(st.jp_cmd_keep) >= 0.5
    prep_L = min(st.jp_L_keep, max(L_SQUAT + 0.020, L_PREP))
    request_cancelled = (request_dynamic
                         and (abs(vel_cmd_raw) < 0.5
                              or vel_cmd_raw * st.jp_cmd_keep <= 0.0))
    # 高腿位补足相对默认站姿的额外预蹲行程，保留原有稳定等待余量。
    request_timeout = (JUMP_REQUEST_TIMEOUT
                       + max(0.0, st.jp_L_keep - L_STAND) / PREP_LEG_RATE)
    if st.jp_pending and (request_cancelled or st.boot_t - st.jp_req_t > request_timeout):
        st.jp_pending = False
        st.jp_preparing = False
        st.jp_committed = False
        st.jp_ready_t = 0.0
    dynamic_jump = abs(vel_cmd_raw) >= 0.5
    balance_ready = (abs(roll) < math.radians(0.5 if dynamic_jump else 0.2)
                     and abs(st.rw_f) < (0.05 if dynamic_jump else 0.01)
                     and abs(pitch) < math.radians(4.5 if dynamic_jump else 4.0)
                     and abs(st.pw_f) < (0.60 if dynamic_jump else 0.10)
                     and abs(st.yw_f) < (0.02 if dynamic_jump else 0.005)
                     and abs(yaw_error) < math.radians(0.5)
                     and abs(st.ws1 - st.ws2) < (0.5 if dynamic_jump else 0.2)
                     and (not request_dynamic
                          or abs(st.vf_f - st.jp_cmd_keep) < JUMP_SPEED_TOL)
                     and abs(qaL - qaR) < (0.005 if dynamic_jump else JUMP_SYNC_Q_TOL)
                     and abs(qbL - qbR) < (0.005 if dynamic_jump else JUMP_SYNC_Q_TOL)
                     and abs(qdaL - qdaR) < (0.05 if dynamic_jump else JUMP_SYNC_QD_TOL)
                     and abs(qdbL - qdbR) < (0.05 if dynamic_jump else JUMP_SYNC_QD_TOL))
    jump_ready = (balance_ready and abs(L - prep_L) < 0.003
                  and abs(st.legd_f) < (0.05 if dynamic_jump else 0.02))
    ready_hold = 0.05 if dynamic_jump else JUMP_READY_HOLD
    if st.jp_pending:
        if balance_ready and st.jp_ready_t == 0.0:
            st.jp_ready_t = st.boot_t
        elif not balance_ready:
            st.jp_ready_t = 0.0
        elif not st.jp_preparing and st.boot_t - st.jp_ready_t > ready_hold:
            st.jp_preparing = True
    if (st.jp == 'DRIVE' and st.jp_pending and st.jp_preparing and st.jp_committed
            and jump_ready and st.jp_ready_t > 0.0
            and st.boot_t - st.jp_ready_t > ready_hold):
        st.jp_pending = False
        st.jp_preparing = False
        st.jp, st.jp_t0 = 'SQUAT', st.boot_t
        st.jp_v_keep = st.vf_f
        st.jp_was_air = False
        st.jp_squat_yaw = yaw
        # 关键禁忌: 进入起跳相位立即解除 standby 锁轮并清空积分，切换为力矩前馈，
        # 防止锁轮逻辑阻碍蹬地爆发力。
        st.stop_standby = False
        st.low_speed_t = 0.0
        st.vel_int = 0.0
        st.pos_int = 0.0
    elif st.jp == 'SQUAT':
        squat_yaw_error = math.atan2(math.sin(yaw - st.jp_squat_yaw),
                                     math.cos(yaw - st.jp_squat_yaw))
        squat_unstable = (abs(roll) > SQUAT_ABORT_ROLL
                          or abs(squat_yaw_error) > SQUAT_ABORT_YAW)
        squat_ready = abs(st.rw_f) < 0.25 and abs(st.yw_f) < 0.15
        if squat_unstable or (jp_el > 0.5 and not squat_ready):
            st.jp, st.jp_t0 = 'DRIVE', st.boot_t
            st.jp_pending = True
            st.jp_preparing = False
            st.jp_req_t = st.boot_t
            st.jp_ready_t = 0.0
        elif jp_el > T_SQUAT and L <= L_SQUAT + 0.001 and squat_ready:
            st.jp, st.jp_t0 = 'JUMP', st.boot_t
        elif jp_el > 1.0:
            st.jp, st.jp_t0 = 'DRIVE', st.boot_t
            st.jp_pending = True
            st.jp_preparing = False
            st.jp_req_t = st.boot_t
            st.jp_ready_t = 0.0
    elif st.jp == 'JUMP':
        vz = d.qvel[2]
        # 轮子真实离地后立即结束蹬地，避免空中继续伸腿造成反复触地。
        if airborne and vz > 0.20 and jp_el > 0.02:
            st.jp, st.jp_t0 = 'FLY', st.boot_t
            st.jp_qa_lock = qaL
            st.jp_qb_lock = qbL
            com_z = float(d.subtree_com[m.body('chassis').id, 2])
            st.jp_takeoff_z = com_z
            st.jp_peak_z = com_z
        elif jp_el > T_PUSH:
            # 蹬地超时保护：未正常离地也不允许无限制施力。
            st.jp, st.jp_t0 = 'FLY', st.boot_t
            st.jp_qa_lock = qaL
            st.jp_qb_lock = qbL
            com_z = float(d.subtree_com[m.body('chassis').id, 2])
            st.jp_takeoff_z = com_z
            st.jp_peak_z = com_z
    elif st.jp == 'FLY':
        st.jp_peak_z = max(st.jp_peak_z, float(d.subtree_com[m.body('chassis').id, 2]))
        if airborne:
            st.jp_was_air = True
        if jp_el > 1.0 and not st.jp_was_air:
            st.jp, st.jp_t0 = 'DRIVE', st.boot_t
            st.jp_committed = False
        elif st.jp_was_air and not airborne and jp_el > 0.05:
            st.jp, st.jp_t0 = 'LAND', st.boot_t
            st.xy_stop = d.qpos[:2].copy()
            st.pos_int = 0.0
            st.vel_int = 0.0
            st.low_speed_t = 0.0
            st.t_stop = st.boot_t
    elif st.jp == 'LAND':
        land_settled = (abs(st.L_cur - st.jp_L_keep) < 0.008
                        and abs(pitch) < 0.05 and abs(st.pw_f) < 0.5)
        if (jp_el > T_LAND and land_settled) or jp_el > 2.5:
            st.jp, st.jp_t0 = 'DRIVE', st.boot_t
            st.jp_committed = False
            # 以完整落地过程结束位置为新停车点，不拉回首次触地点导致平台长期倾斜。
            st.xy_stop = d.qpos[:2].copy()
            st.pos_int = 0.0
            st.vel_int = 0.0
    jump_active = st.jp != 'DRIVE'
    recovery_elapsed = st.boot_t - st.jp_t0
    drive_recovery = (st.jp == 'DRIVE' and st.jp_takeoff_z > 0.0
                      and recovery_elapsed < 3.0)

    # ---------- 高度目标 ----------
    if jump_active:
        if st.jp == 'SQUAT':
            L_target = L_SQUAT
        elif st.jp == 'JUMP':
            L_target = L_JUMP
        elif st.jp == 'FLY':
            L_target = 0.118
        else:
            L_target = st.jp_L_keep
    elif st.jp_pending:
        L_target = prep_L if st.jp_preparing else st.jp_L_keep
    elif drive_recovery:
        L_target = st.jp_L_keep
    else:
        L_target = min(L_MAX, max(L_SQUAT_MIN, L_STAND + st.leg_ref))
    leg_rate = 0.35 if jump_active else (PREP_LEG_RATE if st.jp_pending else 0.025)
    if st.L_cur < L_target:
        st.L_cur = min(L_target, st.L_cur + leg_rate * dt)
    else:
        st.L_cur = max(L_target, st.L_cur - leg_rate * dt)

    # ---------- 腿长控制 F ----------
    dL = max(-0.035, min(0.035, KP_R * roll + KD_R * st.rw_f))
    hgt = L * math.cos(th)
    F_L = F_R = 0.0
    if st.jp == 'SQUAT':
        roll_force = (mass_scale * max(-SQUAT_ROLL_FORCE_LIM, min(
            SQUAT_ROLL_FORCE_LIM, KP_ROLL_SQUAT * roll + KD_ROLL_SQUAT * st.rw_f))
                      if abs(st.jp_v_keep) < SQUAT_ROLL_SPEED_MAX else 0.0)
        squat_force = grav
        if st.lqr6 is not None:
            # 下蹲沿用本机支撑前馈；旧质量缩放常量会使腿长停在起跳门之外。
            support = np.interp(L, st.lqr6.heights, [row[1][2] for row in st.lqr6.table])
            squat_force = support + mass_scale * (
                KP_LEG * (st.L_cur - L) - KD_LEG * st.legd_f)
            squat_force = max(-F_LIM * mass_scale, min(F_LIM * mass_scale, squat_force))
        F_L, F_R = squat_force + roll_force, squat_force - roll_force
    elif st.jp == 'JUMP':
        # 持续竖直推力 (轮子刚离地几毫米就停推会形成"顶起-落下"极限环, 无法起飞)
        jump_force_k = JUMP_FORCE_K + JUMP_FORCE_SPEED_GAIN * min(1.0, abs(st.jp_v_keep))
        jump_scale = mass_scale * (hw.JUMP_FORCE_SCALE if st.hardware else 1.0)
        F_L = F_R = jump_scale * max(-200.0, min(200.0, jump_force_k * (L_JUMP - hgt)))
    elif st.jp == 'FLY':
        F_L = F_R = grav
    elif st.jp == 'LAND' or (drive_recovery and not st.jp_pending):
        legd_comp = max(-st.legd_f, 0.0) * math.cos(th)
        if st.jp == 'LAND' and jp_el < 0.08:
            F_L = F_R = grav + mass_scale * 50.0 * legd_comp
        else:
            ramp = 1.0 if drive_recovery else min(1.0, (jp_el - 0.08) / 0.25)
            # 落地阻抗控制：刚度降为常规 KP_LEG 的 70%
            kp_land = KP_LEG * LAND_IMPEDANCE_RATIO * ramp
            land_limit = 80.0 * mass_scale
            F_L = max(-land_limit, min(land_limit, mass_scale * (
                kp_land * (st.jp_L_keep + dL - hgt) - 40.0 * st.legd_f * math.cos(th)) + grav))
            F_R = max(-land_limit, min(land_limit, mass_scale * (
                kp_land * (st.jp_L_keep - dL - hgt) - 40.0 * st.legd_f * math.cos(th)) + grav))
    else:
        force_limit = F_LIM * mass_scale
        F_L = max(-force_limit, min(force_limit, mass_scale * (
            KP_LEG * (st.L_cur + dL - hgt) - KD_LEG * st.legd_f * math.cos(th)) + grav))
        F_R = max(-force_limit, min(force_limit, mass_scale * (
            KP_LEG * (st.L_cur - dL - hgt) - KD_LEG * st.legd_f * math.cos(th)) + grav))

    if st.jp == 'FLY':
        qa_ref, qb_ref = st.jp_qa_lock, st.jp_qb_lock
    elif st.jp == 'JUMP':
        qa_ref, qb_ref = ik(L_JUMP)   # 固定站姿角 (kp_q 直接拉腿伸直, 防追 L_cur 稀释推力)
    else:
        qa_ref, qb_ref = ik(st.L_cur)

    # ---------- 速度摆 th_cmd ----------
    vel_cmd = vel_cmd_raw
    if st.jp == 'FLY':
        vel_cmd = 0.0
    ve = st.vf_f - vel_cmd
    if abs(vel_cmd) < 0.01:
        # 积分分离: 目标速度 < 0.01 m/s 时不再累积速度环积分。
        # 进入 standby 后积分直接冻结为 0，防止急停冲击造成的积分饱和/溜车。
        st.vel_int = 0.0
    elif abs(ve) > 0.08:
        st.vel_int = max(-0.3, min(0.3, st.vel_int + ve * dt))
    vmax_term = (math.copysign(K_VMAX * (abs(st.vf_f) - V_MAX), st.vf_f)
                 if abs(st.vf_f) > V_MAX else 0.0)
    k_tha_eff = -0.6 if abs(vel_cmd) < 0.01 else K_THA
    # 起步/跳跃恢复后 0.5s 斜坡 (防恢复期速度摆突变)
    ramp_t = min(1.0, st.boot_t / 0.5)
    if st.jp == 'DRIVE' and st.boot_t - st.jp_t0 < 0.5:
        ramp_t = min(ramp_t, (st.boot_t - st.jp_t0) / 0.5)
    # 外环生成腿角参考；保留平台对腿角的低带宽交叉补偿。
    th_cmd_base = (k_tha_eff * (vel_cmd - st.vf_f) + KI_A * st.vel_int + KP_P2 * pitch) * ramp_t
    st.th_cmd = max(-0.20, min(0.20, th_cmd_base + vmax_term))
    if st.boot_t < 1.0:
        st.th_cmd = 0.0
    if st.jp in ('SQUAT', 'JUMP'):
        st.th_cmd = 0.0
    # 平地急停: 腿前伸/后伸承担制动姿态，上平台俯仰参考仍为 0。
    # 正向行驶时 theta_ref > 0 (腿向前)，倒车时符号反转。
    stop_phase = st.jp in ('DRIVE', 'LAND')
    stop_ramp = min(1.0, max(0.0, st.boot_t - st.t_stop) / STOP_LEAN_RAMP)
    if stop_phase and abs(vel_cmd) < 0.01 and abs(st.vf_f) > STOP_SPEED_EPS:
        lean_mag = min(STOP_LEAN_MAX, STOP_LEAN_BASE + STOP_LEAN_GAIN * abs(st.vf_f))
        st.th_cmd = math.copysign(lean_mag * stop_ramp, st.vf_f)

    # 六状态误差。行驶/急停中位置通道暂不拉回，停稳后才接入车头方向的位置误差。
    pos_rel = forward_component(d.qpos[:2] - st.xy_stop, yaw)
    pos_state = pos_rel if (abs(vel_cmd) < 0.01 and abs(st.vf_f) <= 0.1) else 0.0
    vel_ref_state = vel_cmd if abs(vel_cmd) >= 0.01 else 0.0
    state6 = np.array([th - st.th_cmd, st.thd_f, pos_state,
                       st.vf_f - vel_ref_state, pitch, st.pw_f])
    stop_braking = (stop_phase and abs(vel_cmd) < 0.01
                    and abs(st.vf_f) > STOP_SPEED_EPS)
    T_w_stop, T_hub_stop = lqr6_output(state6, K6_STOP_GAIN) if stop_braking else (0.0, 0.0)

    # ---------- T_hub ----------
    if st.jp == 'SQUAT':
        T_hub = (KP_T * (0.0 - th) - KD_T * st.thd_f
                 - KP_HIP_SQUAT * pitch - KD_HIP_SQUAT * st.pw_f)
    elif st.jp == 'JUMP':
        # 蹬地大推力中只混入小幅髋姿态阻尼；下方力矩向量整体缩放保持竖直方向。
        T_hub = -KP_HIP_JUMP * pitch - KD_HIP_JUMP * st.pw_f
    elif st.jp == 'FLY':
        T_hub = 0.0
    elif st.jp == 'LAND':
        T_hub = T_hub_stop if stop_braking else KP_T * (st.th_cmd - th) - KD_T * st.thd_f
    else:
        if stop_braking:
            # 急停的 th_cmd 被腿前伸参考覆盖，在六状态矩阵内直接接入 pitch 交叉项。
            T_hub = T_hub_stop
        else:
            # 普通行驶/跳跃恢复保留已验证的标量运算路径，避免接触相对浮点微小差异敏感。
            kd_t_eff = 1.2 if (st.boot_t - st.jp_t0 < 0.6) else KD_T
            T_hub = KP_T * (st.th_cmd - th) - kd_t_eff * st.thd_f
    if stop_braking or st.jp == 'LAND' or drive_recovery:
        T_hub -= KP_HIP_LEVEL * pitch + KD_HIP_LEVEL * st.pw_f

    # ---------- VMC + 髋回中 ----------
    if st.jp in ('SQUAT', 'JUMP'):
        kp_q_eff = KP_Q_JUMP
        kd_q_eff = KD_Q_SQUAT if st.jp == 'SQUAT' else 0.5
    elif st.jp == 'FLY':
        kp_q_eff = 0.5 * mass_scale
        kd_q_eff = hw.AIR_HIP_DAMPING * mass_scale if st.hardware else 0.1
    elif st.jp == 'LAND':
        kp_q_eff, kd_q_eff = KP_Q, KD_Q
    else:
        kp_q_eff, kd_q_eff = KP_Q, KD_Q
    if st.lqr6 is not None and st.jp in ('DRIVE', 'LAND'):
        # 六状态只接管共模；保留的左右摆腿差模也必须适配整机质量。
        kp_q_eff *= mass_scale
        kd_q_eff *= mass_scale
    if stop_braking:
        # 高速制动由 T_hub 摆腿；接近停稳后再平滑恢复关节回中。
        kp_q_eff *= min(1.0, max(0.0, (0.3 - abs(st.vf_f)) / 0.2))
    vert = st.jp in ('SQUAT', 'JUMP')
    if st.jp == 'JUMP':
        # 蹬地解耦：VMC 生成世界系竖直推力，共模 T_hub 只负责平台水平。
        # 四个髋力矩达到上限时必须按同一比例整体缩放，不能单轴削顶破坏竖直力方向。
        taL, tbL = vmc(qaL, qbL, F_L, T_hub, True)
        taR, tbR = vmc(qaR, qbR, F_R, T_hub, True)
    else:
        taL, tbL = vmc(qaL, qbL, F_L, T_hub, vert)
        taR, tbR = vmc(qaR, qbR, F_R, T_hub, vert)
        taL += kp_q_eff * (qa_ref - qaL) - kd_q_eff * qdaL
        tbL += kp_q_eff * (qb_ref - qbL) - kd_q_eff * qdbL
        taR += kp_q_eff * (qa_ref - qaR) - kd_q_eff * qdaR
        tbR += kp_q_eff * (qb_ref - qbR) - kd_q_eff * qdbR
    if st.jp == 'JUMP':
        raw_peak = max(abs(taL), abs(tbL), abs(taR), abs(tbR), 1e-9)
        push_limit = ((hw.HIP_PEAK_TORQUE if st.hardware else JUMP_HIP_TORQUE_LIM)
                      - JUMP_SPEED_TORQUE_REDUCTION * min(1.0, abs(st.jp_v_keep)))
        push_scale = min(1.0, push_limit / raw_peak)
        taL, tbL = taL * push_scale, tbL * push_scale
        taR, tbR = taR * push_scale, tbR * push_scale
    th_lim = JUMP_HIP_TORQUE_LIM if st.jp == 'JUMP' else (
        SQUAT_HIP_TORQUE_LIM if st.jp == 'SQUAT' else (
        AIR_HIP_TORQUE_LIM if st.jp == 'FLY' else (
            LAND_HIP_TORQUE_LIM if st.jp == 'LAND' else (
                LAND_HIP_TORQUE_LIM if drive_recovery else (
                    3.0 if jump_active else (STOP_HIP_TORQUE_LIM if stop_braking else TH_LIM))))))
    if st.hardware:
        th_lim = min(hw.HIP_PEAK_TORQUE, th_lim * mass_scale)

    # ---------- T_w (平台恒平 + 急停 + 位置环) ----------
    if st.jp == 'FLY':
        # 空中仅用反作用轮控制平台，避免髋环与关节锁姿重复注入角动量。
        T_w = max(-TW_AIR_BASE, min(TW_AIR_BASE,
                                    KP_P_AIR * pitch + KD_P_AIR * st.pw_f))
        st.pos_int = 0.0   # 腾空相屏蔽位置环积分，防止落地冲击积分饱和
    elif st.jp in ('SQUAT', 'JUMP'):
        tw_jump_lim = TW_SQUAT_LIM if st.jp == 'SQUAT' else TW_JUMP_LIM
        T_w = max(-tw_jump_lim, min(tw_jump_lim, KP_P_JUMP * pitch + KD_P_JUMP * st.pw_f))
    elif st.jp == 'LAND':
        if stop_braking and jp_el > 0.12:
            T_w = max(-TW_LAND_LIM, min(TW_LAND_LIM, T_w_stop))
        else:
            # 触地最初 120 ms 先消化反作用轮角动量；此时叠加速度制动会主动打滑。
            land_speed_error = 0.0 if stop_braking else st.vf_f - vel_cmd
            T_w = max(-TW_LAND_LIM, min(TW_LAND_LIM,
                                        KP_P_LAND * pitch + KD_P_LAND * st.pw_f
                                        + LAND_SPEED_K * land_speed_error))
    elif drive_recovery and abs(vel_cmd) >= 0.01:
        T_w = max(-TW_LAND_LIM, min(TW_LAND_LIM,
                                    KP_P_LAND * pitch + KD_P_LAND * st.pw_f
                                    + LAND_SPEED_K * (st.vf_f - vel_cmd)))
    elif abs(vel_cmd) < 0.01:
        if abs(st.vf_f) > STOP_SPEED_EPS:
            # 急停六状态输出: vx 通道提供制动力矩，theta 通道由髋部输出执行。
            # pitch/pitch_dot 权重同比提高，但 pitch_ref 仍严格为 0。
            stop_wheel_limit = 0.6 if drive_recovery else 0.4
            T_w = max(-stop_wheel_limit, min(stop_wheel_limit, T_w_stop))
            st.pos_int = 0.0
        else:
            # 低速: 位置保持环 (回位 + 稳定; 停车后收敛静止)
            if st.stop_standby:
                st.pos_int = 0.0
                v_t = -0.10 * pos_rel
                pref = max(-0.02, min(0.02, 0.12 * (v_t - st.vf_f)))
            else:
                st.pos_int = max(-0.5, min(0.5, st.pos_int + pos_rel * dt))
                v_t = -0.5 * pos_rel - 0.15 * st.pos_int
                pref = max(-0.12, min(0.12, 0.8 * (v_t - st.vf_f)))
            T_w = max(-TW_LIM, min(TW_LIM, KP_P * (pitch - pref) + KD_P * st.pw_f))
    else:
        st.pos_int = 0.0
        # 落速时减少轮上平衡制动，让腿摆在真实轮矩余量内恢复同向速度。
        wheel_limit = (TW_DRIVE_RECOVERY_LIM
                       if vel_cmd * (st.vf_f - vel_cmd) < -0.05 else TW_LIM)
        T_w = max(-wheel_limit, min(wheel_limit, KP_P * pitch + KD_P * st.pw_f))

    # ---------- yaw ----------
    turn_cmd = st.cmd_turn
    # PREP 纠偏死区须小于 0.5° 起跳门，避免失配时在门边反复打断稳定保持。
    yaw_deadband = math.radians(0.25 if st.lqr6 is not None and st.jp_pending else 0.5)
    if st.jp == 'FLY':
        T_y = 0.0
    elif abs(turn_cmd) > 0.01:
        T_y = max(-0.03, min(0.03, KP_WZ * (turn_cmd - st.yw_f)))
    elif abs(yaw_error) < yaw_deadband and abs(st.yw_f) < 0.05:
        T_y = 0.0
    else:
        # 松开转向: 轻回正到锁定朝向 (防行驶自然漂移; 停车时锁定=保持转向后朝向)
        yaw_limit = 0.04
        if st.lqr6 is not None:
            # 硬件候选需要更大的航向制动余量；小误差增益不随质量放大，避免停车粘滑自激。
            yaw_limit *= mass_scale * hw.WHEEL_RADIUS / 0.025
        # 六状态硬件模型需要更强的角速度阻尼来抑制单轮越障瞬态；位置增益保持不变。
        yaw_damping = 0.60 if st.lqr6 is not None else 0.15
        T_y = max(-yaw_limit, min(yaw_limit, -0.4 * yaw_error - yaw_damping * st.yw_f))
    if st.jp_pending and abs(turn_cmd) < 0.01:
        T_y = max(-0.08, min(0.08, T_y - KD_WY * (st.ws1 - st.ws2)))
    st.yaw_torque = T_y
    # ---------- 轮速阻尼 (起步/停车增强, 锁轮防抖) ----------
    kd_w_eff = KD_W
    if st.boot_t < 2.0:
        kd_w_eff *= 1.0 + 2.0 * max(0.0, 1.0 - st.boot_t / 2.0)
    def wdamp(ws):
        w = abs(ws)
        if w < 0.5:
            return -kd_w_eff * ws
        if w < 2.5:
            return -kd_w_eff * ws * (2.5 - w) / 2.0
        if w < 5.0 and abs(vel_cmd_raw) < 0.01:
            return -kd_w_eff * ws * (5.0 - w) / 2.5   # 停车段阻尼区加宽 (防冲起来后无阻尼)
        return 0.0

    ws1 = d.qvel[m.jnt_dofadr[m.joint('wheel1').id]]
    ws2 = d.qvel[m.jnt_dofadr[m.joint('wheel2').id]]
    st.ws1, st.ws2 = ws1, ws2
    # 腾空时轮子是平台姿态反作用轮，轮速阻尼不得抵消姿态力矩。
    if st.jp == 'FLY':
        T_damp1 = T_damp2 = 0.0
    else:
        T_damp1, T_damp2 = wdamp(ws1), wdamp(ws2)

    # ---------- 输出 ----------
    for name, torque, speed in (('motor_alphaL', taL, qdaL), ('motor_betaL', tbL, qdbL),
                                ('motor_alphaR', taR, qdaR), ('motor_betaR', tbR, qdbR)):
        aid = m.actuator(name).id
        low, high = m.actuator_ctrlrange[aid]
        torque = max(low, min(high, max(-th_lim, min(th_lim, torque))))
        if st.hardware:
            torque, _ = hw.torque_limit(
                torque, speed, True, prior_peak_times.get(name, 0.0), dt)
        d.ctrl[aid] = torque
    wheel_left, wheel_right = T_w + T_y + T_damp1, T_w - T_y + T_damp2
    if st.hardware and st.jp == 'DRIVE' and abs(vel_cmd_raw) >= 0.01:
        drive_scale = mass_scale * hw.WHEEL_RADIUS / 0.025
        wheel_left, wheel_right = wheel_left * drive_scale, wheel_right * drive_scale
    if st.jp in ('SQUAT', 'JUMP') and abs(st.cmd_turn) < 0.01:
        wheel_left = wheel_right = 0.5 * (wheel_left + wheel_right)
    elif (static_post_jump and abs(yaw_error) < math.radians(0.5)
          and abs(st.yw_f) < 0.05):
        # 粘滑区内的微小差速会自激 yaw；超出死区后仍保留物理差速回正。
        wheel_left = wheel_right = 0.5 * (wheel_left + wheel_right)
    for name, torque, speed in (('motor_wheelL', wheel_left, ws1),
                                ('motor_wheelR', wheel_right, ws2)):
        aid = m.actuator(name).id
        low, high = m.actuator_ctrlrange[aid]
        torque = max(low, min(high, torque))
        if st.hardware:
            torque, _ = hw.torque_limit(
                torque, speed, False, prior_peak_times.get(name, 0.0), dt)
        d.ctrl[aid] = torque


    if st.lqr6 is not None:
        st.lqr6.apply(m, d, st)

    # 此处是唯一最终输出口。中间限幅保留旧基线的虚拟量计算，但不提交峰值计时。
    if not np.isfinite(d.ctrl).all():
        raise FloatingPointError('基控制器输出非有限值')
    low, high = m.actuator_ctrlrange.T.copy()
    if st.hardware:
        for aid in range(m.nu):
            name = m.actuator(aid).name
            speed = d.qvel[m.jnt_dofadr[m.actuator_trnid[aid, 0]]]
            allowed, _ = hw.torque_limit(float('inf'), speed, 'wheel' not in name,
                                        prior_peak_times.get(name, 0.0), dt)
            low[aid], high[aid] = max(low[aid], -allowed), min(high[aid], allowed)
    base_infeasible = bool(np.any((d.ctrl < low-1e-9) | (d.ctrl > high+1e-9)))
    d.ctrl[:] = np.clip(d.ctrl, low, high)
    if residual is not None:
        residual.apply(m, d, st, low, high, base_infeasible=base_infeasible)
    if not np.isfinite(d.ctrl).all():
        raise FloatingPointError('残差输出非有限值')
    d.ctrl[:] = np.clip(d.ctrl, low, high)
    if st.hardware:
        for aid in range(m.nu):
            name = m.actuator(aid).name
            speed = d.qvel[m.jnt_dofadr[m.actuator_trnid[aid, 0]]]
            d.ctrl[aid], st.motor_peak_t[name] = hw.torque_limit(
                d.ctrl[aid], speed, 'wheel' not in name, prior_peak_times.get(name, 0.0), dt)
    if residual is not None:
        residual.final_clipped = bool(np.any(abs(d.ctrl-residual.base-residual.executed) > 1e-12))


# =========================================================================
# 键盘按住状态 -> 运动命令
# =========================================================================
def sync_held_commands(st):
    """W/S/A/D/方向键只要按住就持续生效，松开立即停止。"""
    st.cmd_vel = float(st.key_w) - float(st.key_s)
    st.cmd_turn = 1.5 * (float(st.key_a) - float(st.key_d))
    st.leg_dir = int(st.key_up) - int(st.key_down)


# =========================================================================
# 无头测试
# =========================================================================
def load_model(xml, hardware=False, track_width=None):
    if track_width is not None and not hardware:
        raise ValueError('轮距选项需要 --hardware')
    if hardware:
        spec = mujoco.MjSpec.from_file(xml)
        hw.configure_spec(spec, track_width)
        m = spec.compile()
    else:
        m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe('stand').id)
    mujoco.mj_forward(m, d)
    return m, d


def make_state(model, hardware=False, six_state=False, *, design=None):
    if six_state and not hardware:
        raise ValueError('六状态候选需硬件名义模型')
    state = St(hardware)
    if six_state:
        from rm_controller import SixStateController
        state.lqr6 = SixStateController(model, design=design)
    return state


def run_headless(mode, hardware=False, track_width=None, six_state=False):
    here = os.path.dirname(os.path.abspath(__file__))
    xml = os.path.join(here, '..', 'xml', 'wheelleg.xml')
    m, d = load_model(xml, hardware, track_width)
    dt = m.opt.timestep
    st = make_state(m, hardware, six_state)
    st.cmd_jump = False

    def qpitch():
        return euler(d)[1]

    fell = False
    invalid = nonwheel_contact = False
    wheel_geoms = {m.geom(name).id for name in ('wheel_collide_L', 'wheel_collide_R')}
    t_end = {'stand': 20, 'stop': 8, 'jump': 12, 'jumpfwd': 18, 'jumpback': 18}[mode]
    n = int(t_end / dt)
    x_release = None
    slide = 0.0
    stop_pitch_max = 0.0
    stop_roll_max = 0.0
    stop_leg_lean_max = 0.0
    stand_pitch_max = 0.0
    stand_roll_max = 0.0
    jump_pitch_max = 0.0
    jump_roll_max = 0.0
    jump_yaw_max = 0.0
    for i in range(n):
        t = i * dt
        if mode == 'stop':
            st.cmd_vel = 1.0 if t < 3.0 else 0.0
            if t >= 3.0 and x_release is None:
                x_release = float(d.qpos[0])  # 松键瞬间锁定，不能再更新 100ms。
        elif mode in ('jump', 'jumpfwd', 'jumpback'):
            jump_speed = 1.0 if mode == 'jumpfwd' else (-1.0 if mode == 'jumpback' else 0.0)
            st.cmd_vel = jump_speed if (mode != 'jump' and t > 1.0) else 0.0
            st.cmd_jump = (2.0 < t < 2.05) if mode == 'jump' else (4.0 < t < 4.05)
        control(m, d, st)
        invalid |= (not np.isfinite(d.ctrl).all()
                    or np.any(d.ctrl < m.actuator_ctrlrange[:, 0] - 1e-9)
                    or np.any(d.ctrl > m.actuator_ctrlrange[:, 1] + 1e-9))
        mujoco.mj_step(m, d)
        invalid |= not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
        nonwheel_contact |= any(c.geom1 not in wheel_geoms and c.geom2 not in wheel_geoms
                                for c in d.contact)
        if mode == 'stand':
            stand_pitch_max = max(stand_pitch_max, abs(qpitch()))
            stand_roll_max = max(stand_roll_max, abs(euler(d)[0]))
        if mode == 'stop' and t >= 3.0:
            # 包含制动初段和首次低速后的再滑行，避免瞬时过零少计停车距离。
            slide = max(slide, abs(float(d.qpos[0]) - x_release))
            p_now = qpitch()
            qa_l, qb_l = d.qpos[7], d.qpos[10]
            qa_r, qb_r = d.qpos[12], d.qpos[15]
            th_now = ((fk_joints(qa_l, qb_l)['phi5'] + fk_joints(qa_r, qb_r)['phi5']) / 2
                      + math.pi / 2 - p_now)
            stop_pitch_max = max(stop_pitch_max, abs(p_now))
            stop_roll_max = max(stop_roll_max, abs(euler(d)[0]))
            stop_leg_lean_max = max(stop_leg_lean_max, abs(th_now))
        jump_monitor = (mode in ('jump', 'jumpfwd', 'jumpback')
                        and (st.jp_pending or st.jp != 'DRIVE'
                             or (st.jp_takeoff_z > 0.0 and st.boot_t - st.jp_t0 < 5.0)))
        if jump_monitor:
            pitch_now = abs(qpitch())
            if pitch_now > jump_pitch_max:
                jump_pitch_max = pitch_now
            jump_roll_max = max(jump_roll_max, abs(euler(d)[0]))
        if jump_monitor:
            yaw_now = euler(d)[2]
            jump_yaw_max = max(jump_yaw_max, abs(math.atan2(
                math.sin(yaw_now - st.yaw_target), math.cos(yaw_now - st.yaw_target))))
        if mode == 'jump' and i % 500 == 0 and t > 1.9:
            print('jump t=%5.2f z=%.4f pitch=%+.3f vx=%.2f' % (t, d.qpos[2], qpitch(), d.qvel[0]))
        if mode == 'jumpfwd' and i % 500 == 0 and t > 3.9:
            print('jf t=%5.2f z=%.4f pitch=%+.3f vx=%.2f bx=%.2f' % (t, d.qpos[2], qpitch(), d.qvel[0], d.qpos[0]))
        if mode == 'jumpback' and i % 500 == 0 and t > 3.9:
            print('jb t=%5.2f z=%.4f roll=%+.3f pitch=%+.3f vx=%.2f bx=%.2f' % (
                t, d.qpos[2], euler(d)[0], qpitch(), d.qvel[0], d.qpos[0]))
        if d.qpos[2] < 0.02 or abs(qpitch()) > 0.7 or abs(euler(d)[0]) > 0.7:
            print('!!! 摔倒 t=%.2f mode=%s' % (t, mode))
            fell = True
            break
    if mode == 'stand':
        test_ok = (not fell and abs(d.qpos[0]) <= 0.02
                   and stand_pitch_max <= PLATFORM_PITCH_LIMIT
                   and stand_roll_max <= PLATFORM_PITCH_LIMIT
                   and max(abs(st.ws1), abs(st.ws2)) <= 0.1)
        print('站立20s: %s (漂移=%.4fm, 最大俯仰=%.2f°, 轮速=%.3frad/s)' % (
            '通过' if test_ok else '失败', abs(d.qpos[0]), math.degrees(stand_pitch_max),
            max(abs(st.ws1), abs(st.ws2))))
    elif mode == 'stop':
        test_ok = (not fell and slide <= 0.60 and stop_pitch_max <= PLATFORM_PITCH_LIMIT
                   and stop_roll_max <= PLATFORM_PITCH_LIMIT
                   and max(abs(st.ws1), abs(st.ws2)) <= 0.05)
        print('急停: %s 滑行=%.3fm 最大平台俯仰=%.2f° 最大腿前伸=%.2f° 停车后轮速=%.4f rad/s' % (
            '通过' if test_ok else '失败', slide, math.degrees(stop_pitch_max),
            math.degrees(stop_leg_lean_max), abs(st.ws1)))
    elif mode == 'jump':
        jump_h = max(0.0, st.jp_peak_z - st.jp_takeoff_z)
        jump_ok = (not fell and jump_h >= JUMP_HEIGHT
                   and jump_pitch_max <= PLATFORM_PITCH_LIMIT
                   and jump_roll_max <= PLATFORM_PITCH_LIMIT
                   and jump_yaw_max <= JUMP_YAW_LIMIT)
        test_ok = jump_ok
        print('静止跳: %s (COM跳高=%.3fm, 最大平台俯仰=%.2f°, 最大偏航=%.2f°, 落点 bx=%.2f)' % (
            '通过' if jump_ok else '失败', jump_h, math.degrees(jump_pitch_max),
            math.degrees(jump_yaw_max), d.qpos[0]))
    elif mode == 'jumpfwd':
        jump_h = max(0.0, st.jp_peak_z - st.jp_takeoff_z)
        jump_ok = (not fell and jump_h >= JUMP_HEIGHT
                   and st.jp_v_keep >= 0.90
                   and jump_pitch_max <= PLATFORM_PITCH_LIMIT
                   and jump_roll_max <= PLATFORM_PITCH_LIMIT
                   and jump_yaw_max <= JUMP_YAW_LIMIT)
        test_ok = jump_ok
        print('前进跳: %s (起跳速度=%.2fm/s, COM跳高=%.3fm, 最大平台俯仰=%.2f°, 最大偏航=%.2f°, bx=%.2f)' % (
            '通过' if jump_ok else '失败', st.jp_v_keep, jump_h, math.degrees(jump_pitch_max),
            math.degrees(jump_yaw_max), d.qpos[0]))
    else:
        jump_h = max(0.0, st.jp_peak_z - st.jp_takeoff_z)
        jump_ok = (not fell and jump_h >= JUMP_HEIGHT
                   and st.jp_v_keep <= -0.90
                   and jump_pitch_max <= PLATFORM_PITCH_LIMIT
                   and jump_roll_max <= PLATFORM_PITCH_LIMIT
                   and jump_yaw_max <= JUMP_YAW_LIMIT)
        test_ok = jump_ok
        print('后退跳: %s (起跳速度=%.2fm/s, COM跳高=%.3fm, 最大横滚=%.2f°, 最大平台俯仰=%.2f°, 最大偏航=%.2f°, bx=%.2f)' % (
            '通过' if jump_ok else '失败', st.jp_v_keep, jump_h, math.degrees(jump_roll_max),
            math.degrees(jump_pitch_max), math.degrees(jump_yaw_max), d.qpos[0]))
    print(f'输出/状态有效={not invalid} 非轮碰撞={nonwheel_contact} '
          f'roll峰值={math.degrees(max(stand_roll_max, stop_roll_max, jump_roll_max)):.3f}°')
    return test_ok and not invalid and not nonwheel_contact

# =========================================================================
# 主入口
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true', default=True)
    parser.add_argument('--track-width', type=float, help='名义轮心距，m；默认300mm内嵌布局')
    parser.add_argument('--six-state', action='store_true', default=True, help='默认启用本机六状态 LQR/VMC')
    parser.add_argument('--size', metavar='WxH', help='窗口尺寸，例如 1280x720')
    parser.add_argument('--fullscreen', action='store_true', help='全屏窗口')
    parser.add_argument('--test', choices=('stand', 'stop', 'jump', 'jumpfwd', 'jumpback'))
    args = parser.parse_args()
    hardware = args.hardware
    window_size = None
    if args.size is not None:
        try:
            window_size = tuple(map(int, args.size.lower().split('x')))
        except ValueError:
            parser.error('--size 必须为正整数 WxH')
        if len(window_size) != 2 or min(window_size) <= 0:
            parser.error('--size 必须为正整数 WxH')
    if (args.track_width is not None or args.six_state) and not hardware:
        parser.error('--track-width/--six-state 需要 --hardware')
    if args.track_width is not None and (not math.isfinite(args.track_width)
                                        or not 0.30 <= args.track_width <= 0.60):
        parser.error('--track-width 必须为 0.30～0.60 m 内的有限数')
    if args.test:
        sys.exit(0 if run_headless(args.test, hardware, args.track_width, args.six_state) else 1)

    here = os.path.dirname(os.path.abspath(__file__))
    xml = os.path.join(here, '..', 'xml', 'wheelleg.xml')
    m, d = load_model(xml, hardware, args.track_width)
    st = make_state(m, hardware, args.six_state)
    st.cmd_jump = False

    # ---------- 自定义可视化窗口 (默认匹配屏幕 0.9 倍, --size WxH 或 --fullscreen 可调) ----------
    import glfw
    glfw.init()
    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    WIN_W, WIN_H = int(mode.size.width * 0.75), int(mode.size.height * 0.75)
    fullscreen = False
    if args.fullscreen:
        WIN_W, WIN_H, fullscreen = mode.size.width, mode.size.height, True
    if window_size is not None:
        WIN_W, WIN_H = window_size
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(WIN_W, WIN_H,
                                'wheelleg sim (W/S 按住前后, A/D 按住转向, 7 跳跃, ↑/↓ 身高, ESC 退出)',
                                monitor if fullscreen else None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    mjr = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scene = mujoco.MjvScene(m, 2000)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED.value
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'demo')  # trackcom 跟随

    def key_cb(window, key, scancode, action, mods):
        # REPEAT 仅表示键仍被按住，不能当成 RELEASE；按住状态只在
        # PRESS/RELEASE 时更新。
        if action == glfw.REPEAT:
            return
        pressed = action == glfw.PRESS
        if key == glfw.KEY_W:
            st.key_w = pressed
        elif key == glfw.KEY_S:
            st.key_s = pressed
        elif key == glfw.KEY_A:
            st.key_a = pressed
        elif key == glfw.KEY_D:
            st.key_d = pressed
        elif key == glfw.KEY_UP:
            st.key_up = pressed
        elif key == glfw.KEY_DOWN:
            st.key_down = pressed
        elif key in (glfw.KEY_7, glfw.KEY_KP_7) and action == glfw.PRESS:
            st.cmd_jump = True   # 只在按下边沿产生一次跳跃脉冲
        elif key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        sync_held_commands(st)

    def focus_cb(window, focused):
        # 切出窗口时 GLFW 可能收不到原按键的 RELEASE，强制清零防止命令卡住。
        if not focused:
            st.key_w = st.key_s = False
            st.key_a = st.key_d = False
            st.key_up = st.key_down = False
            sync_held_commands(st)

    glfw.set_key_callback(window, key_cb)
    glfw.set_window_focus_callback(window, focus_cb)
    print(__doc__)
    auto_demo = '--demo' in sys.argv
    if auto_demo:
        print('\n=== 自动演示模式 === 站立 -> 跳跃 -> 行驶 1m/s -> 松开急停 -> 停车 (12s 后自动退出)')
    dt = m.opt.timestep
    # 每渲染帧仿真步数: 0.5ms/步, N_SUB=100 -> 50ms 仿真/帧 -> 60fps 下约 3x 实时
    # (原来每帧 1 步 -> 33 倍慢于实时; --speed N 可调, 如 --speed 300 快 9x)
    n_sub = 100
    if '--speed' in sys.argv:
        n_sub = max(1, int(sys.argv[sys.argv.index('--speed') + 1]))
    print('仿真加速: 每渲染帧 %d 步 (~%.0fx 实时, --speed N 可调)' % (n_sub, n_sub * dt * 60.0))
    while not glfw.window_should_close(window):
        step_start = time.time()
        if auto_demo:
            t = st.boot_t
            if 2.0 <= t < 2.05:
                st.cmd_jump = True            # 2s: 静止跳跃
            elif 5.0 <= t < 8.0:
                st.cmd_vel = 1.0              # 5-8s: 行驶 1m/s
            else:
                st.cmd_vel = 0.0              # 8s: 松开急停
        for _ in range(n_sub):
            control(m, d, st)
            mujoco.mj_step(m, d)
            if auto_demo and st.boot_t >= 12.0:
                glfw.set_window_should_close(window, True)
                break
        if auto_demo:
            glfw.set_window_title(window,
                'wheelleg demo | t=%.1fs vx=%+.2f m/s pitch=%+.3f z=%.3f 阶段:%s' % (
                    st.boot_t, d.qvel[0], euler(d)[1], d.qpos[2], st.jp))
        elif st.leg_dir != 0:   # 按住 ↑/↓ 持续调身高 (限速 0.08 m/s)
            st.leg_ref = min(L_MAX - L_STAND, max(L_SQUAT_MIN - L_STAND,
                                                  st.leg_ref + st.leg_dir * 0.08 * n_sub * dt))
        # 帧缓冲实际尺寸 (HiDPI: 逻辑像素 2x, 否则画面只占 1/4 其余黑); 未就绪时兜底
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        if fb_w == 0 or fb_h == 0:
            fb_w, fb_h = WIN_W, WIN_H
        viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, mjr)
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()
