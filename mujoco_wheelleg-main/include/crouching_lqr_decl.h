#pragma once
// 平衡 LQR 增益定义 (站立工作点, 五连杆刚性腿 + 双轮)
// 状态:  [s, d_s, phi, d_phi]  (s=机器人位移, phi=机身俯仰)
// 控制:  u = -K·x  (轮毂电机总力矩, 正 = 驱动车身前进方向)
// 由 lqr_design.py 基于编译模型线性化 + Riccati 求解得到。

namespace chassis {
namespace logic {
namespace balance_lqr {

extern const float K[4];  // [s, ds, phi, phid]

}  // namespace balance_lqr
}  // namespace logic
}  // namespace chassis
