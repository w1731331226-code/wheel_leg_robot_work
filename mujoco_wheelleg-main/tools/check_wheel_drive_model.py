#!/usr/bin/env python3
"""轮驱硬件基础：独立名义机械模型与几何/传动检查，不运行生产控制器。"""
import json
import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

import hardware_profile as hw
import wheelleg_sim as sim
import check_wheel_drive_transmission as transmission


ROOT = Path(__file__).resolve().parents[1]
RADIUS, WIDTH = .060325 / 2, .01524
RATIO = 24 / 38
MOTOR_RADIUS, MOTOR_LENGTH = .1135 / 2, .0362
ROTOR_MASS, ROTOR_INERTIA = .324, 8700e-7
MOTOR_MASS, TIRE_MASS = .710, .9 * .0283495
GAP = .005
AXIAL_OFFSET = .005 + GAP + MOTOR_LENGTH / 2  # 小腿半径+间隙+电机半厚。


def build():
    spec = mujoco.MjSpec.from_file(str(ROOT / 'xml/wheelleg.xml'))
    hw.configure_spec(spec, .18)
    baseline = spec.compile()
    # 添加body会改变编译后索引；提前保存两侧句柄，避免名称查询命中旧索引。
    parts = [(side, sign, joint, spec.body('wheel'+side), spec.body('kneeA_'+side),
              spec.geom('wheel_collide_'+side), spec.geom('wheel_geom_'+side),
              spec.actuator('motor_wheel'+side))
             for side, sign, joint in (('L', -1, 'wheel1'), ('R', 1, 'wheel2'))]
    casters = [spec.geom(n) for n in ('caster_front', 'caster_back')]
    key = spec.key('stand')
    for side, sign, wheel_joint, wheel, parent, geom, visual, actuator in parts:
        wheel.mass = TIRE_MASS + .050
        spin = TIRE_MASS * RADIUS**2 + .5 * .050 * hw.WHEEL_HUB_RADIUS**2
        transverse = wheel.mass * (3 * RADIUS**2 + WIDTH**2) / 12
        wheel.inertia = (transverse, spin, transverse)
        geom.size = (RADIUS, WIDTH / 2, RADIUS)
        visual.size[:2] = (RADIUS, WIDTH / 2)

        # 标准375mm带要求109.432mm中心距；沿A链小腿向膝外延伸电机座。
        offset = np.asarray(wheel.pos, dtype=float)
        mount_offset = offset * (1 - transmission.center_distance() / np.linalg.norm(offset))
        mount_offset[1] = sign * AXIAL_OFFSET
        # 电机固定于A链小腿，在膝轴外侧；新增质量不集中到车轮/车身。
        stator_mass = MOTOR_MASS - ROTOR_MASS  # 含厂家总质量与定/转子质量之差的壳体。
        stator_spin = .5 * stator_mass * MOTOR_RADIUS**2
        stator_transverse = stator_mass * (3 * MOTOR_RADIUS**2 + MOTOR_LENGTH**2) / 12
        mount = parent.add_body(name='wheel_drive_' + side, pos=mount_offset,
            mass=stator_mass, inertia=(stator_transverse, stator_spin, stator_transverse),
            explicitinertial=True)
        mount.add_geom(name='wheel_drive_case_' + side, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(MOTOR_RADIUS, MOTOR_LENGTH / 2, 0), quat=(math.sqrt(.5), math.sqrt(.5), 0, 0),
            mass=0, rgba=(.8, .45, .1, 1), contype=1, conaffinity=1)
        # ponytail: 转子横向按轴对称薄环+均匀轴厚估计，CAD后替换；轴向用厂家值。
        rotor_transverse = ROTOR_INERTIA / 2 + ROTOR_MASS * MOTOR_LENGTH**2 / 12
        rotor = mount.add_body(name='wheel_drive_rotor_' + side, mass=ROTOR_MASS,
            inertia=(rotor_transverse, ROTOR_INERTIA, rotor_transverse), explicitinertial=True)
        rotor_joint = 'wheel_drive_spin_' + side
        rotor.add_joint(name=rotor_joint, type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 1, 0),
                        damping=0, armature=0)
        # 原生相对关节约束保留转子与小腿的惯性耦合；不能仅给wheel加i²J。
        spec.add_equality(name='wheel_drive_belt_' + side, type=mujoco.mjtEq.mjEQ_JOINT,
            objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=rotor_joint, name2=wheel_joint,
            data=[0, RATIO] + [0] * 9, solref=(.002, 1))
        actuator.target = rotor_joint
        actuator.ctrlrange = (-12, 12)  # 转子侧目录峰值；不是已实现的电压/热限幅。

        # ponytail: 每侧传动/支架0.100kg为预算，按两轴之间包围盒估计惯性；待CAD。
        span = offset - mount_offset
        half = np.array([abs(span[0]) / 2, .015, abs(span[2]) / 2])
        inertia = .100 / 3 * (np.sum(half**2) - half**2)
        parent.add_body(name='wheel_drive_support_' + side,
            pos=(offset + mount_offset) / 2 + np.array([0, sign * AXIAL_OFFSET / 2, 0]),
            mass=.100, inertia=inertia, explicitinertial=True)

    for caster in casters:
        caster.pos[2] -= RADIUS - hw.WHEEL_RADIUS
    # 新增转子改变qpos布局，按关节名迁移原key，绝不调用旧固定索引控制器。
    old_key = baseline.key_qpos[0].copy()
    key.qpos = []
    model = spec.compile()
    qpos = model.qpos0.copy()
    qpos[:7] = old_key[:7]
    qpos[2] += RADIUS - hw.WHEEL_RADIUS
    for j in range(1, baseline.njnt):
        name = baseline.joint(j).name
        qpos[model.joint(name).qposadr[0]] = old_key[baseline.jnt_qposadr[j]]
    key.qpos = qpos
    return spec


def pose(model, data, height, roll, pitch):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    alpha, beta = sim.ik(height)
    b = sim.L1 * np.array([math.cos(sim.PHI1_STAND-alpha), math.sin(sim.PHI1_STAND-alpha)])
    d = np.array([sim.L5, 0]) + sim.L4 * np.array([math.cos(sim.PHI4_STAND-beta), math.sin(sim.PHI4_STAND-beta)])
    c = np.array([sim.L5 / 2, -height])
    pa = math.atan2(-.0798, .0683) - math.atan2(*(c-b)[::-1]) - alpha
    pc = math.atan2(-.0798, -.0683) - math.atan2(*(c-d)[::-1]) - beta
    for side in ('L', 'R'):
        for name, value in (('alpha'+side, alpha), ('beta'+side, beta),
                            ('passA_'+side, pa), ('passC_'+side, pc)):
            value = (value + math.pi) % (2 * math.pi) - math.pi
            jid = model.joint(name).id
            assert model.jnt_range[jid, 0] <= value <= model.jnt_range[jid, 1], (name, value)
            data.qpos[model.jnt_qposadr[jid]] = value
    cr, sr, cp, sp = math.cos(roll/2), math.sin(roll/2), math.cos(pitch/2), math.sin(pitch/2)
    data.qpos[3:7] = (cr*cp, sr*cp, cr*sp, -sr*sp)  # Ry(pitch) Rx(roll)
    mujoco.mj_forward(model, data)
    bottoms = []
    for side in ('L', 'R'):
        gid = model.geom('wheel_collide_'+side).id
        support = np.linalg.norm(model.geom_size[gid] * data.geom_xmat[gid].reshape(3, 3)[2])
        bottoms.append(data.geom_xpos[gid, 2] - support)
    data.qpos[2] -= min(bottoms)
    mujoco.mj_forward(model, data)
    eq = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert np.max(np.abs(data.efc_pos[eq])) < 1e-8, 'kinematic closure failed'


def check(model):
    assert model.nu == 6 and model.nq == 19
    assert math.isclose(float(sum(model.body_mass)), 4.9246291, abs_tol=1e-9)
    assert 2 * RADIUS <= .070
    for side in ('L', 'R'):
        delta = model.body_pos[model.body('wheel'+side).id] - model.body_pos[model.body('wheel_drive_'+side).id]
        assert math.isclose(np.linalg.norm(delta[[0, 2]]), transmission.center_distance(), abs_tol=1e-12)
    data = mujoco.MjData(model)
    targets = ['floor', 'chassis_box', 'chassis_lid', 'caster_front', 'caster_back']
    for side in ('L', 'R'):
        targets += [name+side for name in ('thighA_', 'thighB_', 'shankA_', 'shankB_',
                                           'wheel_collide_', 'motor_alpha', 'motor_beta')]
    worst = dict(distance_m=1.0)
    count = 0
    for height in np.linspace(.055, .135, 81):
        for roll in (-5, 0, 5):
            for pitch in (-5, 0, 5):
                pose(model, data, height, math.radians(roll), math.radians(pitch))
                count += 1
                for side in ('L', 'R'):
                    case = model.geom('wheel_drive_case_'+side).id
                    for name in targets + ['wheel_drive_case_' + ('R' if side == 'L' else 'L')]:
                        distance = mujoco.mj_geomDistance(model, data, case, model.geom(name).id, .5, None)
                        if distance < worst['distance_m']:
                            worst = dict(distance_m=float(distance), side=side, other=name,
                                         height_m=float(height), roll_deg=roll, pitch_deg=pitch)
    # 传动反例：错误齿比位置必须被原生约束捕获；并核对转子侧力矩输入。
    pose(model, data, .09, 0, 0)
    for side, wheel in (('L', 'wheel1'), ('R', 'wheel2')):
        rotor = 'wheel_drive_spin_'+side
        data.qpos[model.joint(wheel).qposadr[0]] = .2
        data.qpos[model.joint(rotor).qposadr[0]] = RATIO * .2
    mujoco.mj_forward(model, data)
    eq = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert np.max(np.abs(data.efc_pos[eq])) < 1e-8
    data.qpos[model.joint('wheel_drive_spin_L').qposadr[0]] += .01
    mujoco.mj_forward(model, data)
    assert np.max(np.abs(data.efc_pos[eq])) >= .009
    data.ctrl[model.actuator('motor_wheelL').id] = .1
    mujoco.mj_forward(model, data)
    assert math.isclose(data.qfrc_actuator[model.joint('wheel_drive_spin_L').dofadr[0]], .1)
    return dict(status='GEOMETRY_ONLY', poses=count, mass_kg=float(sum(model.body_mass)),
                wheel_diameter_mm=2*RADIUS*1000,
                belt_center_distance_mm=transmission.center_distance()*1000,
                width_over_motor_cases_m=.18 + 2 * (AXIAL_OFFSET + MOTOR_LENGTH / 2),
                minimum_clearance=worst, clearance_pass=worst['distance_m'] >= GAP-1e-6,
                transmission_check=True, production_controller_compatible=False,
                limitations=['kinematic grid only, not motion acceptance', 'legacy hip/frame shapes',
                    'support/belt/pulley/shaft collision envelopes absent',
                    'ideal lossless belt; efficiency, elasticity, current and thermal limits absent'])


if __name__ == '__main__':
    spec = build()
    model = spec.compile()
    result = check(model)
    source_files = ('tools/check_wheel_drive_model.py', 'tools/check_wheel_drive_transmission.py',
                    'tools/check_wheel_drive_voltage.py', 'tools/hardware_profile.py',
                    'tools/wheelleg_sim.py', 'xml/wheelleg.xml')
    result['source_sha256'] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_files
    }
    result['command'] = 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/check_wheel_drive_model.py'
    result['returncode'] = 0 if result['clearance_pass'] else 1
    out = ROOT / 'tools/results/continuation_2026-09-09'
    out.mkdir(parents=True, exist_ok=True)
    # 编译器输出的资源路径仍指向原工程；此文件不是可独立搬运的资源包。
    (out/'wheel_drive_catalog_model.xml').write_text(spec.to_xml())
    (out/'wheel_drive_catalog_model.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['clearance_pass'] else 1)
