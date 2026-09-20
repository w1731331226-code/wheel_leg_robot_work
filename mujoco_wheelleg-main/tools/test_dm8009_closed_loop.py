"""检查实际控制器的左右差模；控制反馈仍为六状态，滤波记忆仅用于离线求导。"""
import contextlib
import copy
import io
import math

import mujoco
import numpy as np

import hardware_profile as hw
import model_lqr as ml
import wheelleg_sim as sim


def check(mass):
    nominal, _ = sim.load_model(ml.XML, True)
    with contextlib.redirect_stdout(io.StringIO()):
        state = sim.make_state(nominal, True, True)
    spec = mujoco.MjSpec.from_file(ml.XML)
    hw.configure_spec(spec)
    spec.geom('chassis_lid').mass += round(mass-hw.DESIGN_MASS, 9)
    model = spec.compile()
    reference = ml.equilibrium(model, sim.L_STAND)
    state.boot_t = state.low_speed_t = 5.0
    state.L_cur = state.L_prev = sim.L_STAND
    state.lqr6.stop_xy = np.zeros(2)
    state.th_prev = (sim.fk_joints(reference.qpos[7], reference.qpos[10])['phi5']
                     + math.pi/2-sim.euler(reference)[1])
    attrs = ['L_prev', 'th_prev', 'legd_f', 'thd_f', 'rw_f', 'pw_f', 'vf_f', 'yw_f']
    n = 2*model.nv+len(attrs)
    matrix = np.empty((n, n))
    eps = 1e-7
    for column in range(n):
        outputs = []
        for sign in (-1, 1):
            data = mujoco.MjData(model)
            mujoco.mj_copyData(data, model, reference)
            trial = copy.deepcopy(state)
            delta = np.zeros(n)
            delta[column] = sign*eps
            mujoco.mj_integratePos(model, data.qpos, delta[:model.nv], 1)
            data.qvel[:] += delta[model.nv:2*model.nv]
            for i, name in enumerate(attrs):
                setattr(trial, name, getattr(trial, name)+delta[2*model.nv+i])
            mujoco.mj_forward(model, data)
            sim.control(model, data, trial)
            mujoco.mj_step(model, data)
            position = np.empty(model.nv)
            mujoco.mj_differentiatePos(model, position, 1, reference.qpos, data.qpos)
            outputs.append(np.r_[position, data.qvel, [getattr(trial, name) for name in attrs]])
        matrix[:, column] = (outputs[1]-outputs[0])/(2*eps)
    radius = float(max(abs(np.linalg.eigvals(matrix))))
    # 绝对轮相位/自由航向含中性模态；只容许有限差分舍入误差，不容许发散极点。
    assert radius <= 1+1e-6, (mass, radius)
    print(f'PASS mass={mass:.2f}kg, fixed controller={hw.DESIGN_MASS:.2f}kg, radius={radius:.9f}')


if __name__ == '__main__':
    for mass in (7, 7.5, 8):
        check(mass)
