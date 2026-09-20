"""当前7kg六状态模型应一次完成下蹲，不靠超时撤回后重试。"""
import contextlib
import io

import mujoco

import model_lqr as ml
import wheelleg_sim as sim


def check(speed):
    model, data = sim.load_model(ml.XML, True)
    with contextlib.redirect_stdout(io.StringIO()):
        state = sim.make_state(model, True, True)
    entered = None
    dt = model.opt.timestep
    request_t = 4.0 if speed else 2.0
    for step in range(int(8.0/dt)):
        t = step*dt
        state.cmd_vel = speed if t > 1.0 else 0.0
        state.cmd_jump = request_t < t < request_t+.05
        previous = state.jp
        sim.control(model, data, state)
        if previous == 'DRIVE' and state.jp == 'SQUAT':
            assert entered is None, '下蹲不应重试'
            entered = t
        assert not (previous == 'SQUAT' and state.jp == 'DRIVE'), '下蹲超时或安全撤回'
        if previous == 'SQUAT' and state.jp == 'JUMP':
            length = (sim.fk_joints(data.qpos[7], data.qpos[10])['leg_len']
                      + sim.fk_joints(data.qpos[12], data.qpos[15])['leg_len'])/2
            assert entered is not None and t-entered < 1.0
            assert length <= sim.L_SQUAT+.001
            assert not speed or speed*state.jp_v_keep >= .90
            print(f'PASS v={speed:+.0f}: first SQUAT→JUMP in {t-entered:.4f}s, L={length:.6f}m')
            return
        mujoco.mj_step(model, data)
    raise AssertionError(f'v={speed:+.0f}: 未进入蹬地')


if __name__ == '__main__':
    for speed in (0, 1, -1):
        check(speed)
