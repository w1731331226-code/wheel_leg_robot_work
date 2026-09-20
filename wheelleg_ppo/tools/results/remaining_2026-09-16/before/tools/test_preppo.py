"""Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python tools/test_preppo.py"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from ppo_env import (MODES, XML, Residual, Scenario, WheelLegEnv, build_model,
                     heldout_combination, sample_scenario)
import wheelleg_sim as sim

OUT = Path(__file__).resolve().parent / 'results/preppo_2026-09-16'


def run():
    rows = []
    env = WheelLegEnv(scenario=Scenario())
    with contextlib.redirect_stdout(io.StringIO()):
        check_env(env)
    rows.append('SB3/Gymnasium interface')

    # Original source snapshot runs in an isolated interpreter, not mixed module globals.
    capture = '''
import sys, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import wheelleg_sim as sim
xml = str(Path(sys.argv[1]).parent / 'xml/wheelleg.xml')
m, d = sim.load_model(xml, True); st = sim.make_state(m, True, True)
trajectory=[]
for i in range(12000):
    st.cmd_vel = float(sys.argv[3]) if 2000 <= i < 8000 else 0.
    sim.control(m, d, st); mujoco.mj_step(m, d)
    trajectory.append(np.r_[d.qpos, d.qvel, d.ctrl, list(st.motor_peak_t.values())])
np.save(sys.argv[2], trajectory)
'''
    env_vars = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for speed in (-1, 1):
        files = []
        for label, source in (('old', OUT/'baseline/tools'), ('new', Path(__file__).parent)):
            dest = OUT/f'trace_{label}_{speed}.npy'
            result = subprocess.run([sys.executable, '-c', capture, str(source), str(dest), str(speed)],
                                    env=env_vars, capture_output=True, text=True)
            (OUT/f'trace_{label}_{speed}.log').write_text(result.stdout + result.stderr)
            assert result.returncode == 0, result.stderr
            files.append(np.load(dest))
        np.testing.assert_array_equal(*files)
    rows.append('original baseline: +/- drive/stop 24,000 physics steps bit-identical')

    # Also verify the Gym wrapper itself, including nominal model compilation and zero residual.
    wrapped=WheelLegEnv(scenario=Scenario()); wrapped.reset(seed=1); wrapped.substeps=1
    plain, data_plain=sim.load_model(XML,True); state_plain=sim.make_state(plain,True,True)
    np.testing.assert_array_equal(wrapped.model.body_inertia,plain.body_inertia)
    for _ in range(6000):
        state_plain.cmd_vel=wrapped._command()
        wrapped.step(np.zeros(3))
        sim.control(plain,data_plain,state_plain); mujoco.mj_step(plain,data_plain)
        np.testing.assert_array_equal(wrapped.data.qpos,data_plain.qpos)
        np.testing.assert_array_equal(wrapped.data.ctrl,data_plain.ctrl)
    rows.append('Gym zero residual: nominal compiled model and 6,000 physics steps bit-identical')

    # Unknown mass cannot change gains/feedforward/mass_scale or commands at identical state.
    m0, m1 = build_model(Scenario()), build_model(Scenario(mass=8))
    s0, s1 = [sim.make_state(m, True, True) for m in (m0, m1)]
    assert s0.lqr6.table is s1.lqr6.table and s0.lqr6.mass_scale == s1.lqr6.mass_scale
    assert not s0.lqr6.heights.flags.writeable and not s0.lqr6.table[0][0].flags.writeable
    data = []
    for m in (m0, m1):
        d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, m.keyframe('stand').id)
        mujoco.mj_forward(m, d); data.append(d)
    for m, d, st in zip((m0, m1), data, (s0, s1)):
        st.cmd_vel = .7
        sim.control(m, d, st)
    np.testing.assert_array_equal(data[0].ctrl, data[1].ctrl)
    rows.append('unknown mass: complete frozen design and identical-state commands')

    # The six-dimensional virtual space contains exactly the differential candidate.
    a = np.array([.2, -.3, .4])
    r3, r6 = Residual('diff3'), Residual('virtual6')
    np.testing.assert_allclose(r3.map(m0, data[0], a), r6.map(m0, data[0], np.ravel(a[:, None]*[1,-1])))
    before = r3.map(m0, data[0], a)
    data[0].qpos[7] += .01
    assert not np.allclose(before, r3.map(m0, data[0], a))
    rows.append('same VMC mapping; configuration-dependent recomputation')

    # Check final-output timing, bounded output, and nonzero residual after six-state overwrite.
    env.reset(seed=2)
    r = env.residual
    r.set_action([1., 1., 1.])
    for _ in range(10):
        prior = env.state.motor_peak_t.copy()
        sim.control(env.model, env.data, env.state, r)
        np.testing.assert_allclose(env.data.ctrl-r.base, r.executed, atol=1e-14)
        assert np.any(r.executed)
        for aid in range(6):
            name = env.model.actuator(aid).name
            speed = env.data.qvel[env.model.jnt_dofadr[env.model.actuator_trnid[aid, 0]]]
            torque, timer = sim.hw.torque_limit(env.data.ctrl[aid], speed, aid < 4, prior.get(name,0), env.dt)
            assert torque == env.data.ctrl[aid] and timer == env.state.motor_peak_t[name]
        mujoco.mj_step(env.model, env.data)
    with patch.object(env.state.lqr6, 'apply', side_effect=lambda m,d,s: d.ctrl.__setitem__(slice(None),[21]*4+[3]*2)):
        env.state.motor_peak_t = {}
        sim.control(env.model, env.data, env.state)
        assert all(t == env.dt for t in env.state.motor_peak_t.values())
    d = data[0]
    d.ctrl[:] = [.9, -.9, 0, 0, 0, 0]
    rr = Residual('torque6'); rr.target[:] = rr.action[:] = 1
    rr.apply(m0, d, s0, -np.ones(6), np.ones(6))
    assert np.isclose(rr.lam, .1) and np.all(abs(d.ctrl)<=1+1e-12)
    np.testing.assert_allclose(rr.executed, rr.lam*rr.requested)
    rr.apply(m0,d,s0,-np.ones(6),np.ones(6),base_infeasible=True)
    assert rr.event == 'base_infeasible' and rr.lam == 0 and not np.any(rr.executed)
    rows.append('single final-output timer; residual not overwritten; common lambda')

    # Repeated reset reproduces a random episode and clears every mutable control state.
    e = WheelLegEnv()
    first, info = e.reset(seed=921)
    actions = np.random.default_rng(1).uniform(-.1, .1, (4,3)).astype(np.float32)
    path = [e.step(a) for a in actions]
    again, again_info = e.reset(seed=921)
    np.testing.assert_array_equal(first, again); assert info == again_info
    assert not e.state.motor_peak_t and e.state.boot_t == 0 and e.state.lqr6.stop_xy is None
    assert np.all(e.residual.action == 0) and np.all(e.residual.target == 0)
    assert e.state.vf_f == 0 and e.state.t_stop == -1 and np.all(e.state.xy_stop == 0)
    for action, expected in zip(actions, path):
        actual = e.step(action)
        np.testing.assert_array_equal(actual[0], expected[0]); assert actual[1:] == expected[1:]
    other = WheelLegEnv(); other.reset(seed=921)
    snapshot = other.data.qpos.copy()
    e.state.motor_peak_t['test'] = 123
    e.state.lqr6.stop_xy = np.ones(2)
    assert 'test' not in other.state.motor_peak_t and other.state.lqr6.stop_xy is None
    np.testing.assert_array_equal(other.data.qpos, snapshot)
    rows.append('seed replay, reset filters/anchors/actions/timers, environment isolation')

    a, b = WheelLegEnv(scenario=Scenario()), WheelLegEnv(scenario=Scenario())
    a.reset(seed=4); b.reset(seed=4); b.substeps = 1
    action = np.array([.1, -.1, .1], dtype=np.float32)
    reward = a.step(action)[1]
    rewards = sum(b.step(action)[1] for _ in range(40))
    assert np.isclose(reward, rewards, atol=1e-12)
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    np.testing.assert_array_equal(a.peak, b.peak)
    euler = sim.euler
    def spike(d):
        if .005 < d.time < .006:
            return .11, 0., 0.
        return euler(d)
    a.reset(seed=4)
    with patch.object(sim, 'euler', spike):
        a.step(np.zeros(3))
    assert a.peak[0] == .11 and a.history[-1][0] != .11
    rows.append('40 substeps: identical reward integration and transient peak capture')

    delayed = WheelLegEnv(scenario=Scenario(mu_l=.6, mu_r=1., delay_ms=10))
    delayed.reset(seed=5); delayed.step(np.zeros(3))
    assert len(delayed.history) == 21
    np.testing.assert_array_equal(delayed._observation(), delayed.history[-1])
    assert not np.array_equal(delayed.history[0], delayed.history[-1])
    assert delayed.contact_mu == {'wheel_collide_L': .6, 'wheel_collide_R': 1.}
    # Friction randomization must not silently change contact stiffness/damping.
    reference=WheelLegEnv(scenario=Scenario()); reference.reset()
    for c in delayed.data.contact:
        np.testing.assert_allclose(c.solref, reference.data.contact[0].solref)
        np.testing.assert_allclose(c.solimp, reference.data.contact[0].solimp)
    rows.append('Actor delay buffer and effective left/right contact friction')

    timeout = WheelLegEnv(scenario=Scenario(), max_seconds=.001)
    timeout.reset(); _, _, terminated, truncated, info = timeout.step(np.zeros(3))
    assert not terminated and truncated and not info['success'] and info['physical_steps']==2
    falling = WheelLegEnv(scenario=Scenario()); falling.reset()
    falling.data.qpos[2] = .01; mujoco.mj_forward(falling.model, falling.data)
    _, _, terminated, truncated, info = falling.step(np.zeros(3))
    assert terminated and not truncated and not info['success']
    for bad in ([float('nan')]*3, [2.,0,0], [0,0]):
        falling.reset()
        try:
            falling.step(bad)
        except ValueError:
            assert falling.data.time == 0
        else:
            raise AssertionError('invalid action accepted')
    rows.append('timeouts versus terminations; invalid action boundaries')

    for split in ('train','validation','test_iid','test_combination'):
        for seed in range(50):
            s = sample_scenario(split,seed)
            assert heldout_combination(s) == (split=='test_combination')
    for mode in MODES:
        e = WheelLegEnv(mode=mode, scenario=Scenario()); e.reset(seed=1)
        o, reward, _, _, _ = e.step(np.zeros(MODES[mode],dtype=np.float32))
        assert e.observation_space.contains(o) and np.isfinite(reward)
    rows.append('held-out joint region and all five action parameterizations')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'acceptance.json').write_text(json.dumps({'passed':True,'checks':rows},indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':rows},indent=2))


if __name__ == '__main__':
    run()
