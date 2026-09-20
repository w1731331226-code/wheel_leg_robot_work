#pragma once
#ifndef WHEELLEG_H
#define WHEELLEG_H

#include <mujoco/mujoco.h>
#include <cstddef>

// 五连杆轮腿机器人控制器 (参照 gitee shuo_kai/balance-simulation 五连杆轮足算法)
// 控制架构:
//   1) 五连杆正解 (每侧): 髋角 -> 腿长/腿角/轮心 (物理角 φ = φ_stand - q)
//   2) 腿长控制:   F_leg = KP·(L_stand + dL_roll - hgt) - KD·legd + 重力补偿
//   3) 横滚补偿:   左右腿长目标差 dL = KP_R·roll + KD_R·roll_w
//   4) 腿角控制:   轮毂力矩 T_hub = -(KP_T·θ + KD_T·θ̇) (保持腿垂直)
//   5) VMC 映射:   髋力矩 τ = Jᵀ·(F_leg·u) + T_hub·gθ (数值雅可比)
//   6) 俯仰平衡:   位置/速度误差 -> 倾斜参考 pref; T_w = KP_P·(φ-pref) + KD_P·φ̇
// 调参: WHEELLEG_KP_LEG / KD_LEG / KP_R / KD_R / KP_T / KD_T / KP_P / KD_P 等环境变量
namespace wheelleg {

// 绑定当前 mjModel/mjData 指针 (main.cc 调用)
void setModelData(const mjModel* m, mjData* d);

// 每步控制入口 (main.cc 在 mj_step 前调用)
void runCrouchingControl(void);

// 移动指令 (simulate 窗口键盘 W/S/X + ↑/↓ 设置):
//   vel  = 前进速度指令 (m/s), 经倾斜参考执行
//   leg  = 腿长指令增量 (m): ↑ +1cm 站高 / ↓ -1cm 蹲低, 限速轨迹 + 髋回中目标 ik(L) 跟随
void SetCommandVel(float vel);
// 键盘按住状态 (主线程 PollEvents 轮询写入, 物理线程读取): vel/turn = -1/0/1
void SetKeyHold(int vel, int turn);
int GetKeyHoldVel();
int GetKeyHoldTurn();
void SetLegRef(float leg);
float GetLegRef();
void SetCommandTurn(float turn);
// 跳跃触发 (空格键/headless): 静止/前进中均可跳跃
void SetJumpReq(bool on);
bool GetJumpReq();
void SetJumpArmed(bool armed);
bool GetJumpArmed();
// 复位控制器内部状态 (测试/重启用)
void ResetController();

// 调试: 在某步数触发 (0 = 关闭)
void setDebugBreakpointAfterSteps(size_t steps);

}  // namespace wheelleg

#endif  // WHEELLEG_H
