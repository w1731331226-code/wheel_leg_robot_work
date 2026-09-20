#!/bin/bash
# 轮腿机器人仿真启动脚本 (自动进入正确目录)
cd /home/wmt/wheel_leg_robot_work/mujoco_wheelleg-main
exec /home/wmt/wheel_leg_robot_work/.venv/bin/python3 -u tools/wheelleg_sim.py "$@"
