"""DM8009 + MF9025 nominal hardware geometry/mass model.

Dimensions, masses and inertias affect MuJoCo dynamics; power, thermal and
regeneration behavior are intentionally idealized for simulation.
"""
import math

import mujoco

HIP_MOTOR_MASS = .896
HIP_RADIUS, HIP_WIDTH = .049, .0617
HIP_RATIO, HIP_EFFICIENCY = 9.0, 1.0  # 仿真理想效率；只保留几何/质量/惯量等硬件参数。
# 输出轴目录值已包含减速器，不再乘9。
HIP_RATED_TORQUE, HIP_PEAK_TORQUE = 20.0, 40.0
HIP_BUS_VOLTAGE, WHEEL_BUS_VOLTAGE = 42.0, 24.0
# ponytail: 按电压线性估算转速，默认仅10S满电；实测压降/联合曲线后回标。
HIP_RATED_RPM, HIP_NO_LOAD_RPM = (rpm * HIP_BUS_VOLTAGE / 24.0 for rpm in (100.0, 160.0))
# ponytail: 转子惯量暂估2e-4 kg·m²，按9:1折算；CAD/台架后替换。
HIP_OUTPUT_INERTIA = 2e-4 * HIP_RATIO**2
MOTOR_MASS = .963
MOTOR_RATED_TORQUE, MOTOR_PEAK_TORQUE = 2.42, 4.5
MOTOR_RATED_RPM, MOTOR_NO_LOAD_RPM = 490.0, 710.0
MOTOR_INERTIA = 4656e-7  # 厂家MF9025V2手册，g·cm² -> kg·m²。
WHEEL_HUB_RADIUS, MOTOR_WIDTH = .0445, .053
WHEEL_RADIUS, WHEEL_WIDTH = .050, .055  # 自定100mm外胎/55mm宽，用户已取消70mm约束。
TIRE_MASS, HUB_MASS = .100, .050  # 外胎/适配环设计预算。
# ponytail: 未公开转子质量；用轴向惯量等效为外缘薄环，其余质量固定在小腿。
ROTOR_MASS = MOTOR_INERTIA / WHEEL_HUB_RADIUS**2
STATOR_MASS = MOTOR_MASS - ROTOR_MASS
WHEEL_ASSEMBLY_MASS = ROTOR_MASS + TIRE_MASS + HUB_MASS
TRACK_WIDTH = .300  # 髋电机内嵌：中央箱体160mm，上平台220mm。
THIGH_MASS, SHANK_MASS = .055, .095  # 7kg仿真轻量连杆预算；实体结构需强度回标。
FRAME_MASS = .170  # 机架、电源、计算、线束预算；仿真另留传感器载荷。
PAYLOAD_MASS = .400  # 雷达、相机、支架和ROS2载荷预留，不代表已选具体器件。
MASS_ITEMS = {
    '4 DM8009 with integrated drivers': 4 * HIP_MOTOR_MASS,
    '2 MF9025 with integrated drivers': 2 * MOTOR_MASS,
    '2 tires and adapter rings': 2 * (TIRE_MASS + HUB_MASS),
    '4 thighs and 4 shanks': 4 * (THIGH_MASS + SHANK_MASS),
    'frame, power, compute, wiring': FRAME_MASS,
    'radar, camera, mounts, ROS2 payload reserve': PAYLOAD_MASS,
    '2 safety casters': .020,
}
DESIGN_MASS = sum(MASS_ITEMS.values())
BASELINE_MASS = 1.018  # 原控制器归一化基准，不是新整机质量。
# 理想供电/热/回生：不把缺少实测的损耗、温升、峰值持续时间写成控制限制。
IDEAL_POWER_MODEL = True
PEAK_WINDOW = math.inf
JUMP_FORCE_SCALE, AIR_HIP_DAMPING = .48, .20


def configure_spec(spec, track_width=None):
    """编译前设置参数，让MuJoCo重算复合惯量与碰撞缓存。"""
    width = TRACK_WIDTH if track_width is None else track_width
    if not math.isfinite(width) or not .30 <= width <= .60:
        raise ValueError('DM8009名义轮距范围为0.30～0.60 m')
    for side, sign in (('L', -1), ('R', 1)):
        for name in ('leg'+side, 'leg'+side+'_D'):
            spec.body(name).pos[1] = sign * width/2
        for joint in ('alpha', 'beta'):
            motor = spec.geom('motor_'+joint+side)
            motor.pos[1] = sign * (width-HIP_WIDTH)/2
            motor.size[:2] = (HIP_RADIUS, HIP_WIDTH/2)
            motor.mass = HIP_MOTOR_MASS
            spec.joint(joint+side).armature = HIP_OUTPUT_INERTIA
            spec.actuator('motor_'+joint+side).ctrlrange = (-HIP_PEAK_TORQUE, HIP_PEAK_TORQUE)
        for name in ('thighA_', 'thighB_'):
            spec.geom(name+side).mass = THIGH_MASS
        for name in ('shankA_', 'shankB_'):
            spec.geom(name+side).mass = SHANK_MASS
        spec.geom('wheel_stator_'+side).mass = STATOR_MASS
        body = spec.body('wheel'+side)
        body.explicitinertial = True
        body.ipos, body.iquat = (0, 0, 0), (1, 0, 0, 0)
        body.mass = WHEEL_ASSEMBLY_MASS
        spin = MOTOR_INERTIA + TIRE_MASS*WHEEL_RADIUS**2 + .5*HUB_MASS*WHEEL_HUB_RADIUS**2
        transverse = spin/2 + WHEEL_ASSEMBLY_MASS*WHEEL_WIDTH**2/12
        body.inertia = (transverse, spin, transverse)
        geom = spec.geom('wheel_collide_'+side)
        geom.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        geom.size = (WHEEL_RADIUS, WHEEL_WIDTH/2, WHEEL_RADIUS)
        geom.friction = (.8, .02, .001)
        spec.geom('wheel_geom_'+side).size[:2] = (WHEEL_RADIUS, WHEEL_WIDTH/2)
        spec.actuator('motor_wheel'+side).ctrlrange = (-MOTOR_PEAK_TORQUE, MOTOR_PEAK_TORQUE)
    spec.geom('floor').friction = (.8, .02, .001)
    for name in ('wheel1', 'wheel2'):
        spec.joint(name).armature = 0.0


def torque_limit(requested, speed, hip, peak_time, dt):
    """理想电源下保留名义扭矩/转速边界，不模拟热、压降或回生。"""
    rated_torque = HIP_RATED_TORQUE if hip else MOTOR_RATED_TORQUE
    peak_torque = HIP_PEAK_TORQUE if hip else MOTOR_PEAK_TORQUE
    rated_rpm = HIP_RATED_RPM if hip else MOTOR_RATED_RPM
    no_load_rpm = HIP_NO_LOAD_RPM if hip else MOTOR_NO_LOAD_RPM
    rpm = abs(speed)*60/(2*math.pi)
    allowed = peak_torque if IDEAL_POWER_MODEL or peak_time < PEAK_WINDOW else rated_torque
    if rpm > rated_rpm:
        allowed *= max(0.0, (no_load_rpm-rpm)/(no_load_rpm-rated_rpm))
    limited = max(-allowed, min(allowed, requested))
    peak_time = peak_time+dt if abs(limited) > rated_torque else max(0.0, peak_time-dt)
    return limited, peak_time
