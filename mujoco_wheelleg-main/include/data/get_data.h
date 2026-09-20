// get_data.h - inline accessors populated from cached sensors/joints
// 五连杆模型: IMU(姿态/角速度) + 4 腿关节 + 2 轮关节
#ifndef MUJOCO_GET_DATA_H_
#define MUJOCO_GET_DATA_H_

#include <mujoco/mujoco.h>
#include "data.h"
#include "data_names.h"
#include <cmath>
#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

// Inline variables (single precision) populated by UpdateInlineGetData
inline float body_pitch = 0.0f;   // 前倾角 (绕 y, 正 = 向前倾)
inline float body_roll  = 0.0f;   // 侧倾角 (绕 x)
inline float body_yaw   = 0.0f;   // 偏航 (绕 z)

inline float body_gpitch = 0.0f;  // 俯仰角速度 (绕 y)
inline float body_groll  = 0.0f;  // 侧倾角速度 (绕 x)
inline float body_gyaw   = 0.0f;  // 偏航角速度 (绕 z)

inline float wheel_angle1 = 0.0f;   // 左轮关节角 (rad, 绕+y轴)
inline float wheel_angle2 = 0.0f;   // 右轮关节角
inline float wheel_speed1 = 0.0f;   // 左轮角速度 (rad/s)
inline float wheel_speed2 = 0.0f;   // 右轮角速度

inline float alpha_L = 0.0f;  // 左腿电机 α 关节角
inline float beta_L  = 0.0f;  // 左腿电机 β 关节角
inline float alpha_R = 0.0f;
inline float beta_R  = 0.0f;
inline float alpha_Lv = 0.0f; // 关节角速度 (从 qvel 读取)
inline float beta_Lv  = 0.0f;
inline float alpha_Rv = 0.0f;
inline float beta_Rv  = 0.0f;

// Helper: convert quaternion (qw,qx,qy,qz) to Euler ZYX (yaw,pitch,roll) [rad]
static inline void QuatToEulerZYX(const double q[4], double& yaw, double& pitch, double& roll) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double siny_cosp = 2.0*(w*z + x*y);
  double cosy_cosp = 1.0 - 2.0*(y*y + z*z);
  yaw = std::atan2(siny_cosp, cosy_cosp);
  double sinp = 2.0*(w*y - z*x);
  if (std::abs(sinp) >= 1.0) pitch = std::copysign(M_PI/2.0, sinp);
  else pitch = std::asin(sinp);
  double sinr_cosp = 2.0*(w*x + y*z);
  double cosr_cosp = 1.0 - 2.0*(x*x + y*y);
  roll = std::atan2(sinr_cosp, cosr_cosp);
}

// 关节 qvel 读取辅助: 关节 id -> 第一个 dof 在 qvel 中的下标
static inline float ReadJointVel(const mjModel* m, const mjData* d, int jid) {
  if (jid < 0 || !d->qvel) return 0.0f;
  if (m->jnt_dofadr) {
    int dofadr = m->jnt_dofadr[jid];
    if (dofadr >= 0 && dofadr < m->nv) return (float)d->qvel[dofadr];
  }
  return 0.0f;
}

// Populate all inline variables by reading cached sensors and joints.
static inline void UpdateInlineGetData(const mjModel* m, const mjData* d) {
  if (!m || !d) return;

  double buf[16];
  std::memset(buf, 0, sizeof(buf));

  // --- Sensors ---
  for (auto& p : g_cached_sensors) {
    if (!p) continue;
    const char* nm = p->getDataName();
    if (!nm) continue;
    if (std::strcmp(nm, "body_orientation") == 0) {
      // 姿态: quat (dim 4) 或 euler (dim 3)
      int dim = p->getDataDim(m);
      p->getData(m, d, buf);
      if (dim == 3) {
        body_pitch = (float)buf[0]; body_roll = (float)buf[1]; body_yaw = (float)buf[2];
      } else if (dim >= 4) {
        double yaw, pitch, roll;
        QuatToEulerZYX(buf, yaw, pitch, roll);
        body_pitch = (float)pitch; body_roll = (float)roll; body_yaw = (float)yaw;
      }
    } else if (std::strcmp(nm, "body_gyro") == 0) {
      p->getData(m, d, buf);
      // gyro: 绕传感器系 x/y/z 角速度 -> [roll, pitch, yaw]
      body_groll = (float)buf[0]; body_gpitch = (float)buf[1]; body_gyaw = (float)buf[2];
    } else if (std::strcmp(nm, "body_accel") == 0) {
      p->getData(m, d, buf);
    } else if (std::strcmp(nm, "wheel_joint_angle1") == 0) {
      p->getData(m, d, buf);
      wheel_angle1 = (float)buf[0];
    } else if (std::strcmp(nm, "wheel_joint_angle2") == 0) {
      p->getData(m, d, buf);
      wheel_angle2 = (float)buf[0];
    } else if (std::strcmp(nm, "wheel_joint_speed1") == 0) {
      p->getData(m, d, buf);
      wheel_speed1 = (float)buf[0];
    } else if (std::strcmp(nm, "wheel_joint_speed2") == 0) {
      p->getData(m, d, buf);
      wheel_speed2 = (float)buf[0];
    }
  }

  // --- Joints: 角度 (qpos) 与速度 (qvel) ---
  for (auto& p : g_cached_joints) {
    if (!p) continue;
    const char* nm = p->getDataName();
    if (!nm) continue;
    int jid = mj_name2id(m, mjOBJ_JOINT, nm);
    if (std::strcmp(nm, "alphaL") == 0) {
      p->getData(m, d, buf); alpha_L = (float)buf[0]; alpha_Lv = ReadJointVel(m, d, jid);
    } else if (std::strcmp(nm, "betaL") == 0) {
      p->getData(m, d, buf); beta_L = (float)buf[0]; beta_Lv = ReadJointVel(m, d, jid);
    } else if (std::strcmp(nm, "alphaR") == 0) {
      p->getData(m, d, buf); alpha_R = (float)buf[0]; alpha_Rv = ReadJointVel(m, d, jid);
    } else if (std::strcmp(nm, "betaR") == 0) {
      p->getData(m, d, buf); beta_R = (float)buf[0]; beta_Rv = ReadJointVel(m, d, jid);
    }
  }
}

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_GET_DATA_H_
