#!/usr/bin/env python3
"""完整启动流程验证 (新模型 + 参考 balance-simulation 算法):
站立位形起步 -> 五连杆腿长PID+横滚补偿+腿角控制 -> 车轮倾斜参考平衡
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mujoco_wheelleg-main/tools"))
import test_stand_balance  # 复用验证过的控制器 (站立平衡 12s)
