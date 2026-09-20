"""轮胎编译缓存回归：实际相交时，碰撞筛选不能漏掉单轮凸台。"""
import math

import mujoco
import numpy as np

import hardware_profile as hw
from test_robustness import XML


def check():
    # 先验证当前主方案，再用历史大轮尺寸覆盖曾经发生的缓存漏更新。
    radius = hw.WHEEL_RADIUS
    try:
        for hw.WHEEL_RADIUS in (radius, .0492125):
            check_radius()
    finally:
        hw.WHEEL_RADIUS = radius


def check_radius():
    radius = hw.WHEEL_RADIUS
    height = radius / 2
    spec = mujoco.MjSpec.from_file(XML)
    spec.worldbody.add_geom(name='probe', type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[-2, .09, height / 2], size=[.25, .035, height / 2])
    reference = spec.compile()
    hw.configure_spec(spec, .18)
    model = spec.compile()
    data = mujoco.MjData(model)
    chassis = model.body('chassis').id
    assert math.isclose(sum(model.body_mass), hw.DESIGN_MASS)
    assert np.allclose(model.body_iquat, reference.body_iquat)
    assert np.allclose(model.body_ipos, reference.body_ipos)
    assert np.allclose(model.body_inertia[chassis], reference.body_inertia[chassis]
                       * model.body_mass[chassis] / reference.body_mass[chassis])
    wheel, box = model.geom('wheel_collide_R').id, model.geom('probe').id
    misses = 0
    for phase in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        mujoco.mj_resetDataKeyframe(model, data, model.keyframe('stand').id)
        data.qpos[0] = -1.75 + .8 * radius - .03
        data.qpos[2] += .04 * radius  # 爬升后旧包围盒下缘高于凸台，实际轮胎仍与棱边相交。
        data.qpos[model.joint('wheel2').qposadr[0]] = phase
        mujoco.mj_forward(model, data)
        distance = mujoco.mj_geomDistance(model, data, wheel, box, .1, None)
        assert distance < -.02 * radius, distance
        assert any({c.geom1, c.geom2} == {wheel, box} for c in data.contact), phase
        # 反例：旧硬件转换漏更新的 25mm 包围盒必须被本检查捕获。
        node = model.body_bvhadr[model.body('wheelR').id]
        bounds = model.bvh_aabb[node].copy()
        model.bvh_aabb[node, 3:] = .025
        mujoco.mj_forward(model, data)
        misses += not any({c.geom1, c.geom2} == {wheel, box} for c in data.contact)
        model.bvh_aabb[node] = bounds
    if radius > .025:
        assert misses > 0, '回归必须能捕获旧缓存缺陷'
    print(f'PASS: radius={radius} 24 wheel phases; stale bounds miss {misses}/24; nominal inertia preserved')


if __name__ == '__main__':
    check()
