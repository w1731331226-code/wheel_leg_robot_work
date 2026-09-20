#!/usr/bin/env python3
"""Render a few reproducible MuJoCo frames for the research presentation."""
from pathlib import Path
import sys

import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import wheelleg_sim as sim

OUT = Path(__file__).resolve().parent / "images"


def save(renderer, data, camera, name):
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(OUT / name)


def main():
    OUT.mkdir(exist_ok=True)
    model, data = sim.load_model(str(ROOT / "xml" / "wheelleg.xml"))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1600, 900
    state = sim.St()
    renderer = mujoco.Renderer(model, height=900, width=1600)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.03, 0.0, 0.10)
    camera.distance, camera.azimuth, camera.elevation = 0.48, 135, -12

    captured = set()
    previous = "DRIVE"
    for step in range(int(8.0 / model.opt.timestep)):
        t = step * model.opt.timestep
        state.cmd_jump = 2.0 < t < 2.05
        sim.control(model, data, state)
        mujoco.mj_step(model, data)
        camera.lookat[:] = (float(data.qpos[0]) + 0.03, 0.0, max(0.10, float(data.qpos[2])))

        if t >= 1.5 and "overview" not in captured:
            save(renderer, data, camera, "01_robot_overview.png")
            captured.add("overview")
        if state.jp_pending and state.L_cur <= 0.092 and "prep" not in captured:
            save(renderer, data, camera, "02_jump_preparation.png")
            captured.add("prep")
        if state.jp == "JUMP" and previous != "JUMP" and "takeoff" not in captured:
            save(renderer, data, camera, "03_jump_takeoff.png")
            captured.add("takeoff")
        if state.jp == "FLY" and data.qvel[2] <= 0 and "peak" not in captured:
            save(renderer, data, camera, "04_jump_peak.png")
            captured.add("peak")
        if state.jp == "LAND" and previous != "LAND" and "landing" not in captured:
            save(renderer, data, camera, "05_jump_landing.png")
            captured.add("landing")
        previous = state.jp

    renderer.close()
    assert len(captured) == 5, captured


if __name__ == "__main__":
    main()
