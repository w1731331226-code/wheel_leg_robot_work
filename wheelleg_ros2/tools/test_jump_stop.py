"""跳跃落地松键回归：双向制动距离与停车抖动。"""
import math
import os
import argparse

import mujoco
import numpy as np

import wheelleg_sim as sim


XML = os.path.join(os.path.dirname(sim.__file__), '..', 'xml', 'wheelleg_dm8009.xml')


def run(direction, release_delay, hardware=False, track_width=None, six_state=False):
    model, data = sim.load_model(XML, hardware, track_width)
    state = sim.make_state(model, hardware, six_state)
    land_t = release_t = None
    release_x = stop_x = None
    stopped_t = None
    tail_x = []
    tail_w = []
    pitch_max = 0.0
    roll_max = slide = 0.0
    invalid = nonwheel_contact = False
    wheels = {model.geom(n).id for n in ('wheel_collide_L', 'wheel_collide_R')}
    for step in range(int(16.0 / model.opt.timestep)):
        t = step * model.opt.timestep
        release = land_t is not None and t >= land_t + release_delay
        state.cmd_vel = 0.0 if release else float(direction) if t > 1.0 else 0.0
        state.cmd_jump = 4.0 < t < 4.05
        if release and release_t is None:
            release_t, release_x = t, float(data.qpos[0])
        previous = state.jp
        sim.control(model, data, state)
        mujoco.mj_step(model, data)
        invalid |= not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                        and np.isfinite(data.ctrl).all())
        nonwheel_contact |= any(c.geom1 not in wheels and c.geom2 not in wheels for c in data.contact)
        if previous == 'FLY' and state.jp == 'LAND':
            land_t = t
        if release_t is not None:
            pitch_max = max(pitch_max, abs(sim.euler(data)[1]))
            roll_max = max(roll_max, abs(sim.euler(data)[0]))
            slide = max(slide, abs(float(data.qpos[0]) - release_x))
            if stopped_t is None and state.jp == 'DRIVE' and abs(state.vf_f) < 0.03:
                stopped_t, stop_x = t, float(data.qpos[0])
            if t > 14.0:
                tail_x.append(float(data.qpos[0]))
                tail_w.append(max(abs(state.ws1), abs(state.ws2)))
    slide = slide if release_t is not None else float('inf')
    jitter = max(tail_x) - min(tail_x) if tail_x else float('inf')
    wheel_jitter = max(tail_w) if tail_w else float('inf')
    passed = (stopped_t is not None and slide <= 0.45 and jitter <= 0.003
              and wheel_jitter <= 0.055 and max(pitch_max, roll_max) <= sim.PLATFORM_PITCH_LIMIT
              and not invalid and not nonwheel_contact)
    print(f'{"前进" if direction > 0 else "后退"} delay={release_delay:.2f}s: '
          f'{"PASS" if passed else "FAIL"} slide={slide:.3f}m '
          f'stop={stopped_t - release_t if stopped_t else -1:.2f}s '
          f'jitter={jitter:.4f}m wheel={wheel_jitter:.3f}rad/s '
          f'pitch/roll={math.degrees(pitch_max):.2f}/{math.degrees(roll_max):.2f}° '
          f'valid={not invalid} nonwheel={nonwheel_contact}')
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--track-width', type=float)
    parser.add_argument('--six-state', action='store_true')
    args = parser.parse_args()
    assert all([run(direction, delay, args.hardware, args.track_width, args.six_state)
                for direction in (1, -1) for delay in (0.0, 0.25, 0.50)])
