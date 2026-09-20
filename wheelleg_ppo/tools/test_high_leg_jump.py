"""最高腿位跳跃及请求取消/提交回归；沿用原动作与姿态判据。"""
import test_jump_height as test
import wheelleg_sim as sim


if __name__ == '__main__':
    options = dict(hardware=True, six_state=True)
    results = [test.run_height(speed, sim.L_MAX, **options) for speed in (0, 1, -1)]
    results += [test.cancel_on_release(**options), test.prepare_then_commit(**options)]
    assert all(results), '最高腿位跳跃或请求取消/提交未通过'
