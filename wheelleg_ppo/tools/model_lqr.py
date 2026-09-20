"""从现用 MJCF 求静态工作点与矢状面离散 LQR；独立验证，不替换跳跃控制。

保留基座高度、被动关节和轮速动态；仅消去左右差模与无关的轮相位。
"""
import argparse
import math
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import null_space, solve_discrete_are
from scipy.optimize import least_squares

import wheelleg_sim as sim
from state_estimation import leg_kinematics


XML = str(Path(__file__).resolve().parents[1] / 'xml' / 'wheelleg.xml')


def sagittal_basis(model):
    """15 状态：x/z/pitch/四腿关节位置，及其速度与轮速；双腿同向。"""
    left = [model.jnt_dofadr[model.joint(n).id] for n in
            ('alphaL', 'passA_L', 'betaL', 'passC_L', 'wheel1')]
    right = [model.jnt_dofadr[model.joint(n).id] for n in
             ('alphaR', 'passA_R', 'betaR', 'passC_R', 'wheel2')]
    velocity = np.zeros((model.nv, 8))
    velocity[[0, 2, 4], [0, 1, 2]] = 1.0
    for column, (a, b) in enumerate(zip(left, right), 3):
        velocity[[a, b], column] = 1.0
    basis = np.zeros((2 * model.nv, 15))
    basis[:model.nv, :7] = velocity[:, :7]
    basis[model.nv:, 7:] = velocity
    inputs = np.zeros((model.nu, 3))
    for column, names in enumerate((('motor_wheelL', 'motor_wheelR'),
                                    ('motor_alphaL', 'motor_alphaR'),
                                    ('motor_betaL', 'motor_betaR'))):
        for name in names:
            inputs[model.actuator(name).id, column] = 1.0
    return basis, inputs


def equilibrium(model, height):
    if not math.isfinite(height) or not sim.L_SQUAT_MIN <= height <= sim.L_MAX:
        raise ValueError('工作腿长必须在 0.160～0.380 m 内')
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe('stand').id)
    qa = [model.jnt_qposadr[model.joint(n).id] for n in
          ('alphaL', 'passA_L', 'betaL', 'passC_L')]
    qb = [model.jnt_qposadr[model.joint(n).id] for n in
          ('alphaR', 'passA_R', 'betaR', 'passC_R')]
    basis, inputs = sagittal_basis(model)
    velocity = basis[model.nv:, 7:]
    # 新尺寸必须从闭合几何出发，不能用旧增益预运行来猜新模型平衡点。
    alpha, beta = sim.ik(height)
    bpos = sim.L1 * np.array([math.cos(sim.PHI1_STAND-alpha), math.sin(sim.PHI1_STAND-alpha)])
    dpos = np.array([sim.L5, 0]) + sim.L4 * np.array([math.cos(sim.PHI4_STAND-beta), math.sin(sim.PHI4_STAND-beta)])
    cpos = np.array([sim.L5/2, -height])
    avec = model.body_pos[model.body('wheelL').id][[0, 2]]
    cvec = model.site_pos[model.site('couplerB_L_end').id][[0, 2]]
    passive_a = math.atan2(avec[1], avec[0]) - math.atan2(*(cpos-bpos)[::-1]) - alpha
    passive_c = math.atan2(cvec[1], cvec[0]) - math.atan2(*(cpos-dpos)[::-1]) - beta
    data.qpos[qa] = data.qpos[qb] = [alpha, passive_a, beta, passive_c]
    data.qpos[2] = height + model.geom_size[model.geom('wheel_collide_L').id, 0]
    hip = sim.vmc(alpha, beta, sum(model.body_mass)*9.81/2, 0, False)
    data.ctrl[:] = inputs @ np.r_[0, hip]
    initial = np.r_[data.qpos[2], sim.euler(data)[1], data.qpos[qa],
                    np.linalg.pinv(inputs) @ data.ctrl]
    joints = [model.joint(n).id for n in ('alphaL', 'passA_L', 'betaL', 'passC_L')]
    limits = np.min(np.where(inputs.T != 0, model.actuator_ctrlrange[:, 1], np.inf), axis=1)
    lower = np.r_[0.02, -math.pi / 2, model.jnt_range[joints, 0], -limits]
    upper = np.r_[sim.L1+sim.L2+2*sim.hw.WHEEL_RADIUS, math.pi / 2, model.jnt_range[joints, 1], limits]

    def residual(values):
        mujoco.mj_resetData(model, data)
        data.qpos[2] = values[0]
        pitch = values[1]
        data.qpos[3:7] = (math.cos(pitch / 2), 0, math.sin(pitch / 2), 0)
        data.qpos[qa] = data.qpos[qb] = values[2:6]
        data.ctrl[:] = inputs @ values[6:]
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        forces = velocity.T @ (data.qfrc_inverse - data.qfrc_actuator)
        length = sim.fk_joints(values[2], values[4])['leg_len']
        # 静态内力允许多个站姿；增加平台水平约束，避免求解器任取俯仰工作点。
        return np.r_[forces, 100 * (length - height), pitch]

    result = least_squares(residual, initial, xtol=1e-12, ftol=1e-12,
                           gtol=1e-12, max_nfev=500, x_scale='jac', bounds=(lower, upper))
    error = residual(result.x)
    mujoco.mj_forward(model, data)
    if not result.success or np.max(abs(error)) > 1e-6 or np.max(abs(data.qacc)) > 1e-4:
        raise RuntimeError(f'平衡点未收敛: residual={max(abs(error)):.3g}, '
                           f'qacc={max(abs(data.qacc)):.3g}')
    if np.any(abs(data.ctrl) > model.actuator_ctrlrange[:, 1]):
        raise RuntimeError('平衡点超过执行器力矩上限')
    wheels = {model.geom(n).id for n in ('wheel_collide_L', 'wheel_collide_R')}
    floor = model.geom('floor').id
    pairs = [{c.geom1, c.geom2} for c in data.contact]
    if len(pairs) != 2 or not all({wheel, floor} in pairs for wheel in wheels):
        raise RuntimeError('平衡点必须仅有左右轮与平地接触')
    return data


def linearize(model, data, eps=1e-6):
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, data)
    full_a = np.empty((2 * model.nv, 2 * model.nv))
    full_b = np.empty((2 * model.nv, model.nu))
    mujoco.mjd_transitionFD(model, scratch, eps, True, full_a, full_b, None, None)
    basis, inputs = sagittal_basis(model)
    project = np.linalg.pinv(basis)
    return project @ full_a @ basis, project @ full_b @ inputs


def design(model, height):
    data = equilibrium(model, height)
    a, b = linearize(model, data)
    for pole in np.linalg.eigvals(a):
        if abs(pole) >= 1 - 1e-8 and np.linalg.matrix_rank(
                np.c_[pole * np.eye(len(a)) - a, b], tol=1e-9) < len(a):
            raise RuntimeError('存在不可稳定的离散模态')
    # ponytail: 先保留柔性闭环/接触模态；六状态降阶须先验证被省略动态的影响。
    scales = np.array([0.03, 0.005, 0.03, 0.1, 0.1, 0.1, 0.1,
                       0.3, 0.1, 0.3, 2.0, 2.0, 2.0, 2.0, 20.0])
    q = np.diag(1 / scales ** 2)
    mass_scale = sum(model.body_mass) / sim.hw.BASELINE_MASS
    r = np.diag(1 / (mass_scale * np.array([0.2, 0.5, 0.5])) ** 2)
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    poles = np.linalg.eigvals(a - b @ gain)
    if max(abs(poles)) >= 1:
        raise RuntimeError('LQR 闭环不稳定')
    return data, a, b, gain


def check_linearization(model, reference, a, b):
    """独立扰动一步前向仿真，核对线性预测和轮矩符号；不用于压力注入。"""
    basis, inputs = sagittal_basis(model)
    project = np.linalg.pinv(basis)
    dx = np.random.default_rng(5).normal(size=15) * 1e-7
    du = np.array([1.0, -0.2, 0.3]) * 1e-6
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, reference)
    full_delta = basis @ dx
    mujoco.mj_integratePos(model, scratch.qpos, full_delta[:model.nv], 1.0)
    scratch.qvel[:] += full_delta[model.nv:]
    scratch.ctrl[:] += inputs @ du
    mujoco.mj_step(model, scratch)
    delta = np.empty(model.nv)
    mujoco.mj_differentiatePos(model, delta, 1.0, reference.qpos, scratch.qpos)
    actual = project @ np.r_[delta, scratch.qvel]
    predicted = a @ dx + b @ du
    error = np.linalg.norm(actual - predicted) / np.linalg.norm(predicted)
    a_half, b_half = linearize(model, reference, 5e-7)
    assert error < 0.005, f'一步线性预测误差 {error:.3%}'
    assert np.allclose(a, a_half, rtol=0.01, atol=1e-4)
    assert np.allclose(b, b_half, rtol=0.01, atol=1e-6)
    # 逐输入实测符号；近奇异腿高的首步响应不能沿用正常腿高的符号假设。
    for column in range(inputs.shape[1]):
        mujoco.mj_copyData(scratch, model, reference)
        scratch.ctrl[:] += inputs[:, column] * 1e-6
        mujoco.mj_step(model, scratch)
        mujoco.mj_differentiatePos(model, delta, 1.0, reference.qpos, scratch.qpos)
        measured = project @ np.r_[delta, scratch.qvel]
        assert np.allclose(measured, b[:, column] * 1e-6, rtol=0.01, atol=1e-9)
    print(f'linearization error={error:.3%} wheel_dvx_dtau={b[7, 0]:.6f}', flush=True)


def validate(model, reference, gain, impulse):
    """非线性物理脉冲检查；LQR 仅输出真实六电机力矩。"""
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, reference)
    basis, inputs = sagittal_basis(model)
    project = np.linalg.pinv(basis)
    common = inputs @ np.linalg.pinv(inputs)
    delta = np.empty(model.nv)
    peak_attitude = np.zeros(3)
    peak_position = 0.0
    chassis = model.body('chassis').id
    wheels = {model.geom(n).id for n in ('wheel_collide_L', 'wheel_collide_R')}
    nonwheel_contact = False
    last_unsettled = 0.55
    hardware = abs(sum(model.body_mass) - sim.hw.DESIGN_MASS) < 1e-6
    state = sim.St(hardware)
    height = sim.fk_joints(reference.qpos[7], reference.qpos[10])['leg_len']
    state.L_cur = state.L_prev = height
    state.leg_ref = height - sim.L_STAND
    state.boot_t = 5.0
    state.th_prev = (sim.fk_joints(reference.qpos[7], reference.qpos[10])['phi5']
                     + math.pi / 2 - sim.euler(reference)[1])
    peak_times = {}
    for step in range(round(4 / model.opt.timestep)):
        t = step * model.opt.timestep
        mujoco.mj_differentiatePos(model, delta, 1.0, reference.qpos, data.qpos)
        error = project @ np.r_[delta, data.qvel]
        # 仅替换矢状面共模，保留已有 roll/yaw 差模，避免人为冻结左右自由度。
        state.motor_peak_t = peak_times.copy()
        sim.control(model, data, state)
        differential = data.ctrl - common @ data.ctrl
        data.ctrl[:] = np.clip(reference.ctrl - inputs @ gain @ error + differential,
                               model.actuator_ctrlrange[:, 0],
                               model.actuator_ctrlrange[:, 1])
        if hardware:
            for aid in range(model.nu):
                joint = model.actuator_trnid[aid, 0]
                name = model.actuator(aid).name
                speed = data.qvel[model.jnt_dofadr[joint]]
                data.ctrl[aid], peak_times[name] = sim.hw.torque_limit(
                    data.ctrl[aid], speed, 'wheel' not in name,
                    peak_times.get(name, 0.0), model.opt.timestep)
        data.xfrc_applied[chassis, 0] = impulse / 0.05 if 0.5 <= t < 0.55 else 0.0
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert np.isfinite(data.ctrl).all()
        nonwheel_contact |= any(c.geom1 not in wheels and c.geom2 not in wheels
                                for c in data.contact)
        peak_attitude = np.maximum(peak_attitude, np.abs(sim.euler(data)))
        peak_position = max(peak_position, abs(data.qpos[0]))
        if t >= 0.55 and (max(abs(v) for v in sim.euler(data)) > math.radians(3)
                          or abs(data.qvel[0]) > 0.03):
            last_unsettled = t
    recovery = max(0.0, last_unsettled - 0.55)
    print(f'nonlinear impulse={impulse:+.3f}Ns r/p/y={np.degrees(peak_attitude).round(3)}deg '
          f'peak_x={peak_position:.4f} final_x={data.qpos[0]:.5f} '
          f'final_v={data.qvel[0]:.5f} recovery={recovery:.2f}s '
          f'nonwheel={nonwheel_contact}', flush=True)
    # 路线第 5.1 节：物理碰撞瞬态 15°、恢复后 3°；跳跃的 5° 门不变。
    return (max(peak_attitude) <= math.radians(15)
            and not nonwheel_contact and recovery <= 2.0
            and max(abs(v) for v in sim.euler(data)) <= math.radians(3)
            and abs(data.qpos[0]) < 0.01 and abs(data.qvel[0]) < 0.01)


def vmc_coordinates(reference):
    """局部输出 [theta,theta_dot,x,vx,pitch,pitch_dot,L,L_dot] 与每侧虚拟输入。"""
    qa, qb = reference.qpos[7], reference.qpos[10]
    jac = np.array([sim.vmc(qa, qb, 1, 0, False),
                    sim.vmc(qa, qb, 0, 1, False)]).T
    # 用闭环两圆约束的解析微分独立核对生产 VMC 的数值雅可比。
    fk = sim.fk_joints(qa, qb)
    _, _, _, analytic = leg_kinematics([qa, qb], [0, 0])
    assert np.allclose(jac, analytic, rtol=1e-6, atol=1e-8), 'VMC 符号/虚功映射错误'
    c = np.zeros((8, 15))
    c[0, 2] = c[1, 9] = -1
    c[0, [3, 5]] = c[1, [10, 12]] = jac[:, 1]
    c[2, 0] = c[3, 7] = c[4, 2] = c[5, 9] = 1
    c[6, [3, 5]] = c[7, [10, 12]] = jac[:, 0]
    # [每轮 Tw, 每侧 Thub, 每侧径向 F] -> [每轮 Tw, 每侧 tau_alpha, tau_beta]
    g = np.zeros((3, 3))
    g[0, 0], g[1:, 1], g[1:, 2] = 1, jac[:, 1], jac[:, 0]
    # F 的单位为 N、Thub 为 Nm；按腿长归一后报告条件数，避免混用量纲。
    condition = np.linalg.cond(jac @ np.diag([1, fk['leg_len']]))
    return c, g, float(condition)


def reduce_dynamics(a, b, c, method='modal'):
    """保留慢模态，再换回原物理输出坐标；不增加在线反馈状态。"""
    order = len(c)
    if method == 'modal':
        poles, modes = np.linalg.eig(a)
        indices = np.argsort(abs(poles - 1))
        keep, omitted = indices[:order], indices[order:]
        output_modes = c @ modes[:, keep]
        condition = float(np.linalg.cond(output_modes))
        if not np.isfinite(condition) or condition > 1e8:
            raise RuntimeError('保留模态不能可靠映射至指定物理状态')
        ar = output_modes @ np.diag(poles[keep]) @ np.linalg.inv(output_modes)
        br = output_modes @ np.linalg.inv(modes)[keep] @ b
        for value in (ar, br):
            if np.max(abs(value.imag)) > 1e-9 * max(1., np.max(abs(value.real))):
                raise RuntimeError('降阶截断了共轭模态对')
        return ar.real, br.real, float(max(abs(poles[omitted]))), condition
    if method != 'static':
        raise ValueError('未知降阶方法')
    transform = np.r_[c, null_space(c).T]
    at = transform @ a @ np.linalg.inv(transform)
    bt = transform @ b
    aff = at[order:, order:]
    fast = np.linalg.solve(np.eye(len(a) - order) - aff,
                           np.c_[at[order:, :order], bt[order:]])
    ar = at[:order, :order] + at[:order, order:] @ fast[:, :order]
    br = bt[:order] + at[:order, order:] @ fast[:, order:]
    return ar, br, float(max(abs(np.linalg.eigvals(aff)))), float(np.linalg.cond(transform))


def reduced_design(model, reference, a, b, order, method='modal'):
    """离线降阶设计；检查省略动态、密集频响及完整15状态闭环。"""
    if order not in (6, 8):
        raise ValueError('只比较六状态与带腿长/腿速的八状态')
    c8, g, condition = vmc_coordinates(reference)
    c = c8[:order]
    mass = sum(model.body_mass) / sim.hw.BASELINE_MASS
    outer = np.zeros((3, 15))
    if order == 6:
        # 六状态候选仍保留既有径向腿长 PD；不能凭空删除竖直支撑控制。
        outer[2] = mass * (sim.KP_LEG * c8[6] + sim.KD_LEG * c8[7])
    count = 2 if order == 6 else 3
    full_a, full_b = a - b @ g @ outer, b @ g[:, :count]
    ar, br, omitted_radius, projection_condition = reduce_dynamics(full_a, full_b, c, method)
    scales = np.array([0.05, 0.3, 0.03, 0.3, 0.03, 0.3, 0.005, 0.1])[:order]
    efforts = mass * np.array([0.2, 0.5, 10.0])[:count]
    q, r = np.diag(scales**-2), np.diag(efforts**-2)
    p = solve_discrete_are(ar, br, q, r)
    kr = np.linalg.solve(r + br.T @ p @ br, br.T @ p @ ar)
    gain = g @ outer + g[:, :count] @ kr @ c
    frequencies = np.unique(np.r_[np.geomspace(0.2, 5.0, 201), [0.2, 0.5, 1., 2., 5.]]).tolist()
    errors = []
    for hz in frequencies:
        z = np.exp(2j * math.pi * hz * model.opt.timestep)
        full = c @ np.linalg.solve(z * np.eye(15) - full_a, full_b)
        reduced = np.linalg.solve(z * np.eye(order) - ar, br)
        full = full * efforts[None, :] / scales[:, None]
        reduced = reduced * efforts[None, :] / scales[:, None]
        errors.append(float(np.linalg.norm(full - reduced) / np.linalg.norm(full)))
    radius = float(max(abs(np.linalg.eigvals(a - b @ gain))))
    report = dict(order=order, method=method, vmc_condition=condition,
                  projection_condition=projection_condition, frequencies_hz=frequencies,
                  relative_frf_errors=errors, omitted_radius=omitted_radius,
                  max_relative_frf_error=max(errors),
                  reduced_closed_radius=float(max(abs(np.linalg.eigvals(ar - br @ kr)))),
                  full_closed_radius=radius,
                  linear_pass=bool(omitted_radius < 1 and radius < 1 and max(errors) <= 0.10))
    # 门槛：离散稳定且所列 0.2～5 Hz 采样点归一频响误差均 <=10%。
    # 这是候选筛选门，不是连续频段、参数调度或实机稳定性证明。
    print('reduction ' + json.dumps({k: v for k, v in report.items()
                                    if k not in ('frequencies_hz', 'relative_frf_errors')}), flush=True)
    return gain, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--reduction', type=int, choices=(6, 8), help='独立比较降阶候选；失败返回非零')
    parser.add_argument('--reduction-method', choices=('modal', 'static'), default='modal')
    parser.add_argument('--report', type=Path, help='保存降阶工作点与中点 JSON 报告')
    parser.add_argument('--track-width', type=float, help='仅硬件名义轮距敏感性模型，m')
    parser.add_argument('--check-midpoints', action='store_true', help='验证相邻增益平均值在中点的闭环')
    parser.add_argument('--heights', type=float, nargs='+', default=[sim.L_PREP, sim.L_STAND, sim.L_MAX])
    args = parser.parse_args()
    if args.report and not args.reduction:
        parser.error('--report 需要 --reduction')
    if (not all(math.isfinite(h) and sim.L_SQUAT_MIN <= h <= sim.L_MAX for h in args.heights)
            or any(b <= a for a, b in zip(args.heights, args.heights[1:]))):
        parser.error('--heights 必须在 0.160～0.380 m 内且严格递增')
    if args.track_width is not None:
        if not args.hardware:
            parser.error('--track-width 需要 --hardware')
        from check_track_width import model_at_width
        model, _ = model_at_width(args.track_width)
    else:
        model, _ = sim.load_model(XML, args.hardware)
    gains, passed, reports = [], [], []
    virtual_gains = []
    for height in args.heights:
        data, a, b, gain = design(model, height)
        print(f'L={height:.4f} qacc={max(abs(data.qacc)):.2e} '
              f'pitch={math.degrees(sim.euler(data)[1]):.3f}deg '
              f'open={max(abs(np.linalg.eigvals(a))):.6f} '
              f'closed={max(abs(np.linalg.eigvals(a-b@gain))):.6f}', flush=True)
        print('u_eq=', data.ctrl, flush=True)
        check_linearization(model, data, a, b)
        if args.reduction:
            gain, report = reduced_design(model, data, a, b, args.reduction, args.reduction_method)
            report['height'] = height
            reports.append(report)
        gains.append(gain)
        if args.reduction == 6:
            c, g, _ = vmc_coordinates(data)
            virtual_gains.append(np.linalg.solve(g, gain) @ np.linalg.pinv(c))
        point_checks = []
        for impulse in (-0.2, 0.2):
            # 完整线性闭环已发散的候选不再送入非线性压力；保留明确失败记录。
            ok = (not args.reduction or report['full_closed_radius'] < 1) and validate(model, data, gain, impulse)
            point_checks.append(bool(ok))
            passed.append(bool(ok))
        if args.reduction:
            report['nonlinear_pass'] = all(point_checks)
    if args.check_midpoints:
        for index, (low, high) in enumerate(zip(args.heights, args.heights[1:])):
            height = (low + high) / 2
            data = equilibrium(model, height)
            a, b = linearize(model, data)
            gain = (gains[index] + gains[index + 1]) / 2
            if args.reduction == 6:
                # 在线插值的是物理状态增益，当前构型的 VMC 映射不能随端点平均。
                c, g, _ = vmc_coordinates(data)
                gain = g @ ((virtual_gains[index] + virtual_gains[index + 1]) / 2) @ c
            radius = max(abs(np.linalg.eigvals(a - b @ gain)))
            print(f'interpolated L={height:.4f} closed={radius:.6f}', flush=True)
            if not args.reduction:
                assert radius < 1, '插值增益在中点不稳定'
            point_checks = [bool(radius < 1 and validate(model, data, gain, impulse))
                            for impulse in (-0.2, 0.2)]
            passed.extend(point_checks)
            if args.reduction:
                reports.append(dict(height=height, interpolated=True,
                                    full_closed_radius=float(radius),
                                    nonlinear_pass=all(point_checks)))
    print(f'nonlinear checks: {sum(passed)}/{len(passed)} PASS', flush=True)
    if args.report:
        args.report.write_text(json.dumps(dict(hardware=args.hardware, track_width=args.track_width,
                                               timestep=model.opt.timestep, results=reports),
                                          indent=2, allow_nan=False) + '\n')
    assert all(passed), '非线性脉冲恢复未过门，不能接入正式控制'
    assert all(r.get('linear_pass', True) for r in reports), '降阶误差/被消去动态未过门，不能接入正式控制' 
