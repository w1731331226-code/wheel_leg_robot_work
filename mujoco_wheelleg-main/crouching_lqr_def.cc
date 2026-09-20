#include "crouching_lqr_decl.h"

namespace chassis {
namespace logic {
namespace balance_lqr {

// LQR 增益 (lqr_v3.py, Q=diag(2,1,60,6)):
//   K[0] 关零: 位置环由 wheelleg.cc 位置外环->速度环承担
//   K[1]=+1.4555: 速度负反馈阻尼 (lqr_v3 的 -1.4555 因模型 s 符号约定相反, 取反)
//   平衡反馈 (phi/phid): 模型 轮式倒立摆 m=0.982kg, 质心高 0.1097m, Ip=2.5e-3
const float K[4] = {-0.9327f, -1.4555f, -9.2872f, -1.8759f};

}  // namespace balance_lqr
}  // namespace logic
}  // namespace chassis
