#ifndef CHASSIS_LOGIC_H
#define CHASSIS_LOGIC_H

#include "algorithm/fsm.h"
#include "chassis.h"
#include "ins.h"
#include "lqr.h"
#include "remote.h"
#include "vmc.h"
#include <cmath>

namespace chassis::logic {

namespace event {

class WantStandUp : public os::Event {};
class WantCrouch : public os::Event {};
class BodyTipped : public os::Event {};
class LegError : public os::Event {};
class WheelLost : public os::Event {};
class InitDone : public os::Event {};
class AdjustDone : public os::Event {};

}  // namespace event

namespace action {

inline void updateAllData() {
    getLeftLeg().updateData();
    getRightLeg().updateData();
    getLeftWheel().updateData();
    getRightWheel().updateData();
}

inline void setWheelTorque(float left_torque, float right_torque) {
    getLeftWheel().setOutput(left_torque);
    getRightWheel().setOutput(right_torque);
}

inline void applyWheelOutput() {
    getLeftWheel().applyOutput();
    getRightWheel().applyOutput();
}

inline void setLegForceTorque(float left_T, float left_F, float right_T, float right_F) {
    getLeftLeg().setForceTorque(left_T, left_F);
    getRightLeg().setForceTorque(right_T, right_F);
}

inline void applyLegOutput() {
    getLeftLeg().setOutputWithTAndF();
    getRightLeg().setOutputWithTAndF();
}

inline void stopAllOutput() {
    getLeftLeg().stopOutput();
    getRightLeg().stopOutput();
    getLeftWheel().stopOutput();
    getRightWheel().stopOutput();
}

inline void initializeAll() {
    getLeftLeg().initialize();
    getRightLeg().initialize();
    getLeftWheel().initialize();
    getRightWheel().initialize();
}

inline bool isAllInitialized() {
    return getLeftLeg().isInitialized() && getRightLeg().isInitialized() && getLeftWheel().isInitialized() &&
           getRightWheel().isInitialized();
}

}  // namespace action

namespace state {

class Dead final : public os::State {
    void onEnter() override { action::stopAllOutput(); }

    void onExecute() override { action::initializeAll(); }
};

class Debugging final : public os::State {
    void onEnter() override {}

    void onExecute() override { action::updateAllData(); }
};

/**
 * @brief 走地鸡状态 (Crouching) - LQR 增益矩阵
 * Q = diag([10 800 1200 10 50 10 50 10 10 10])
 * R = diag([15 15 1 1])
 *
 * ⚠️ 重要：此LQR使用的状态向量定义为：
 * x = [theta, v_avg-ref, yaw_err, yaw_dot, phi_l_err, d_phi_l, phi_r_err, d_phi_r, L_l_err, L_r_err]
 * 其中：
 *   - [0] theta: 机身倾角 (暂不使用, 置0)
 *   - [1] v_avg-ref: 轮子平均速度 - 参考速度
 *   - [2] yaw_err: 偏航角误差 (限制在[-π, π])
 *   - [3] yaw_dot: 偏航角速度
 *   - [4] phi_l_err: 左腿摆角误差 (目标-实际, 取负)
 *   - [5] d_phi_l: 左腿摆角角速度 (取负)
 *   - [6] phi_r_err: 右腿摆角误差 (目标-实际, 取负)
 *   - [7] d_phi_r: 右腿摆角角速度 (取负)
 *   - [8] L_l_err: 左腿长度误差 (暂不使用, 置0)
 *   - [9] L_r_err: 右腿长度误差 (暂不使用, 置0)
 *
 * 注意：这与 lqr.h 中的 WheelLegState 定义完全不同！
 * 增益矩阵K_1-K_6需要对应此处的状态定义。
 */
namespace crouching_lqr {
// Q = diag([1 10 10 1 50 10 50 10 200 10 ])
// R = diag([30 30 1 1 ])

inline os::math::Matrix<4, 10> const K_1 {-0.118005, -0.503039, -0.406628, -0.147041, -1.337975, -0.437260, -1.063868, -0.343902, -0.508338, -0.157198, -0.118005, -0.503039, 0.406628, 0.147041, -1.063868, -0.343902, -1.337975, -0.437260, -0.508338, -0.157198, 0.182090, 0.748883, -0.321360, -0.126236, 5.406007, 2.174167, -2.178498, -1.107607, -9.630287, -2.149750, 0.182090, 0.748883, 0.321360, 0.126236, -2.178498, -1.107607, 5.406007, 2.174167, -9.630287, -2.149750};

inline os::math::Matrix<4, 10> const K_2 {-0.304111, -1.064610, 0.688140, 0.313360, -8.661194, -1.348104, 2.248198, 0.576427, 0.664603, 0.206836, 0.243392, 1.060673, 0.518436, 0.262975, -4.535392, -0.155678, 4.971173, 0.973597, 0.793530, 0.294256, 1.458912, 5.588668, -1.763607, -0.682989, 2.868662, -3.676922, -1.763441, -2.101797, -7.573192, -1.683985, -1.939965, -7.737417, 1.504026, 0.494204, -3.736816, 1.515381, -3.493728, 1.362221, 5.840039, 1.124371};

inline os::math::Matrix<4, 10> const K_3 {0.418704, 1.582475, -0.733586, -0.294616, 6.993200, 0.651375, -0.335055, -0.139938, 0.271016, 0.048544, -0.200883, -0.831189, -0.627072, -0.320825, 5.083780, -0.337338, -5.873598, -1.030292, -0.316015, -0.168959, -2.716425, -10.491938, 2.946355, 1.011696, -5.202638, 3.946734, -4.393276, 1.533063, 8.268812, 1.672982, 2.177641, 8.532880, -2.558253, -0.741060, -1.650622, -2.616339, 5.603045, -1.545021, -6.492020, -1.201798};

inline os::math::Matrix<4, 10> const K_4 {0.243392, 1.060673, -0.518436, -0.262975, 4.971173, 0.973597, -4.535392, -0.155678, 0.793530, 0.294256, -0.304111, -1.064610, -0.688140, -0.313360, 2.248198, 0.576427, -8.661194, -1.348104, 0.664603, 0.206836, -1.939965, -7.737417, -1.504026, -0.494204, -3.493728, 1.362221, -3.736816, 1.515381, 5.840039, 1.124371, 1.458912, 5.588668, 1.763607, 0.682989, -1.763441, -2.101797, 2.868662, -3.676922, -7.573192, -1.683985};

inline os::math::Matrix<4, 10> const K_5 {-0.200883, -0.831189, 0.627072, 0.320825, -5.873598, -1.030292, 5.083780, -0.337338, -0.316015, -0.168959, 0.418704, 1.582475, 0.733586, 0.294616, -0.335055, -0.139938, 6.993200, 0.651375, 0.271016, 0.048544, 2.177641, 8.532880, 2.558253, 0.741060, 5.603045, -1.545021, -1.650622, -2.616339, -6.492020, -1.201798, -2.716425, -10.491938, -2.946355, -1.011696, -4.393276, 1.533063, -5.202638, 3.946734, 8.268812, 1.672982};

inline os::math::Matrix<4, 10> const K_6 {-0.136389, -0.842916, -0.076723, -0.071712, 2.745319, 0.156909, -2.168519, -0.703235, -1.261333, -0.415614, -0.136389, -0.842916, 0.076723, 0.071712, -2.168519, -0.703235, 2.745319, 0.156909, -1.261333, -0.415614, 0.943108, 4.012766, -1.387252, -0.526574, 7.588651, 1.342995, 0.043351, 0.003057, 0.445040, 0.282004, 0.943108, 4.012766, 1.387252, 0.526574, 0.043351, 0.003057, 7.588651, 1.342995, 0.445040, 0.282004};


inline control::LQR<10, 4> lqr{K_1, K_2, K_3, K_4, K_5, K_6};
}  // namespace crouching_lqr

inline control::LQRInput<10> debug_crouching;
inline float debug_output[4];

/**
 * @brief 走地鸡状态 (Crouching)
 *
 * 在此模式下：
 * - T（摆角力矩）：由 LQR 计算，用于维持倒立摆平衡
 * - F（轴向力）：只做重力补偿，不主动控制腿长
 * - 轮子：由 LQR 输出控制
 */
class Crouching final : public os::State {
    void onEnter() override { targetPhi_ = 0.0f; }

    void onExecute() override {
        action::updateAllData();

        // 获取腿部状态
        float leftL = getLeftLeg().getL();
        float rightL = getRightLeg().getL();
        float leftPhi = getLeftLeg().getPhi();          // ← 反向符号
        float rightPhi = getRightLeg().getPhi();        // ← 反向符号
        float leftPhiDot = getLeftLeg().getPhiDot();    // ← 反向符号
        float rightPhiDot = getRightLeg().getPhiDot();  // ← 反向符号

        // 获取轮子速度
        float leftWheelVel = getLeftWheel().getVelocity();
        float rightWheelVel = getRightWheel().getVelocity();

        // 获取 INS 数据
        auto pitch = ins::getGimbalPitch().convert<units::angle::radian>().to<float>();
        auto pitchDot = ins::getPitchSpeed().convert<units::angular_velocity::radians_per_second>().to<float>();
        auto yaw_err = std::fmod(
                           playerSettings.chassis_ref.yaw.to<float>() -
                               ins::getGimbalYaw().convert<units::angle::radian>().to<float>() + PI,
                           2.0f * PI
                       ) -
                       PI;  // 限制在 [-π, π]
        auto yaw_dot = ins::getYawSpeed().convert<units::angular_velocity::radians_per_second>().to<float>();
        auto forward_ref = playerSettings.chassis_ref.vy.to<float>();  // 前进速度参考（来自遥控器）

        float wheelSpeedAvg = (leftWheelVel + rightWheelVel) * 0.5f;

        // ===== 判断是否处于静止状态 =====
        // 静止状态：速度很小且前进指令很小
        bool require_still = std::abs(wheelSpeedAvg) < 0.4f && std::abs(forward_ref) < 0.01f;

        // ===== 位置积分项（test_x）=====
        // 用于改进速度跟踪和减少稳态误差
        if (require_still) {
            // 静止时：积分位置偏差
            positionIntegral_ += std::abs(wheelSpeedAvg) < 0.1f ? wheelSpeedAvg * 0.001f : wheelSpeedAvg * 0.0007f;
            positionIntegral_ = std::clamp(positionIntegral_, -0.4f, 0.4f);
        } else {
            // 运动时：清零位置积分
            positionIntegral_ = 0.0f;
        }

        float state_v;
        if (require_still) {
            state_v = wheelSpeedAvg;
        } else {
            state_v = std::clamp(wheelSpeedAvg - forward_ref, -1.0f, 1.0f);
        }

        // ===== LQR 状态向量 (参考 WheelLeg crouching) =====
        // [0]: theta (暂不使用, 置0)
        // [1]: v_avg (轮子平均速度 - 参考速度)
        // [2]: yaw_err
        // [3]: yaw_dot
        // [4]: phi_l_err (左摆角误差，取负)
        // [5]: d_phi_l (左摆角角速度，取负)
        // [6]: phi_r_err (右摆角误差，取负)
        // [7]: d_phi_r (右摆角角速度，取负)
        // [8]: L_l_err (暂不使用, 置0)
        // [9]: L_r_err (暂不使用, 置0)
        control::LQRInput<10> input{};
        input.x[0] = 0.0f;
        input.x[1] = wheelSpeedAvg;
        input.x[2] = 0.0f;
        input.x[3] = 0;
        input.x[4] = (pitch - (leftPhi + 1.57f));
        input.x[5] = (pitchDot - leftPhiDot);
        input.x[6] = (pitch - (rightPhi + 1.57f));
        input.x[7] = (pitchDot - rightPhiDot);
        input.x[8] = pitch;
        input.x[9] = pitchDot;
        input.left_height = leftL;
        input.right_height = rightL;

        for (int i = 0; i < 10; ++i) {
            debug_crouching.x[i] = input.x[i];
        }
        debug_crouching.left_height = input.left_height;
        debug_crouching.right_height = input.right_height;

        // LQR 计算
        // 输出: [0] 左轮力矩, [1] 右轮力矩, [2] 左腿T, [3] 右腿T
        auto const& u = crouching_lqr::lqr.process(input);

        // 提取输出
        float wheel_l = u(0, 0)/5;
        float wheel_r = u(1, 0)/5;
        float leg_T_l = u(2, 0);
        float leg_T_r = u(3, 0);

        // // 低通滤波 + 阻尼 + 死区 + 速率限幅（参考 WheelLeg 设计）
        // static float filtPitchDot_c = 0.0f;
        // static float filtWheelV_c = 0.0f;
        // static float last_wl_c = 0.0f;
        // static float last_wr_c = 0.0f;
        // constexpr float lpf_alpha_c = 0.995f;  // stronger smoothing (more phase lag)
        // filtPitchDot_c = lpf_alpha_c * filtPitchDot_c + (1.0f - lpf_alpha_c) * pitchDot;
        // filtWheelV_c = lpf_alpha_c * filtWheelV_c + (1.0f - lpf_alpha_c) * ((leftWheelVel + rightWheelVel) * 0.5f);
        //
        // constexpr float kd_c = 2.0f;  // stronger damping
        // wheel_l -= kd_c * filtPitchDot_c;
        // wheel_r -= kd_c * filtPitchDot_c;
        //
        // constexpr float wheel_dead_c = 0.5f;  // larger deadband
        // if (std::abs(wheel_l) < wheel_dead_c) wheel_l = 0.0f;
        // if (std::abs(wheel_r) < wheel_dead_c) wheel_r = 0.0f;
        //
        // constexpr float max_delta_c = 0.05f;  // slower rate limit
        // wheel_l = std::clamp(wheel_l, last_wl_c - max_delta_c, last_wl_c + max_delta_c);
        // wheel_r = std::clamp(wheel_r, last_wr_c - max_delta_c, last_wr_c + max_delta_c);
        // last_wl_c = wheel_l;
        // last_wr_c = wheel_r;
        //
        // // 降低总体轮子输出幅度，降低振荡风险
        // constexpr float wheel_scale_c = 0.35f; // reduce overall wheel amplitude
        // wheel_l *= wheel_scale_c;
        // wheel_r *= wheel_scale_c;

        // 输出限幅
        constexpr float kWheelLimit = 0.3f;
        constexpr float kLegTorqueLimit = 2.0f;
        wheel_l = std::clamp(wheel_l, -kWheelLimit, kWheelLimit);
        wheel_r = std::clamp(wheel_r, -kWheelLimit, kWheelLimit);
        leg_T_l = std::clamp(leg_T_l, -kLegTorqueLimit, kLegTorqueLimit);
        leg_T_r = std::clamp(leg_T_r, -kLegTorqueLimit, kLegTorqueLimit);

        debug_output[0] = wheel_l;
        debug_output[1] = wheel_r;
        debug_output[2] = leg_T_l;
        debug_output[3] = leg_T_r;

        // T: LQR 输出（注意符号）
        float left_T = -leg_T_l;
        float right_T = -leg_T_r;

        // F: 只做重力补偿（不控制腿长）
        constexpr float body_mass = 0.0f;  // 半边机体质量 kg
        constexpr float g = 9.81f;
        // float left_F = body_mass * g * std::cos(pitch - leftPhi + targetPhi_);
        // float right_F = body_mass * g * std::cos(pitch - rightPhi + targetPhi_);
        float left_F = 0;
        float right_F = 0;

        // 设置输出
        action::setLegForceTorque(left_T, left_F, right_T, right_F);
        action::applyLegOutput();

        action::setWheelTorque(wheel_l, wheel_r);
        action::applyWheelOutput();
    }

private:
    float targetPhi_{0.0f};
    float positionIntegral_{0.0f};
};

/**
 * @brief 站立状态 (Standing) - LQR 增益矩阵
 * 站立状态下使用更长的腿长，需要更好的动态平衡控制
 * Q = diag([10 800 500 10 100 10 100 10 3000 10])
 * R = diag([10 10 1 1])
 *
 * ⚠️ 重要：此LQR使用的状态向量定义为：
 * x = [test_x, v, yaw_err, yaw_dot, pitch-(phi_l-target), pitch_dot-d_phi_l,
 *      pitch-(phi_r-target), pitch_dot-d_phi_r, pitch, pitch_dot]
 * 其中：
 *   - [0] test_x: 位置积分项 (用于改进速度跟踪)
 *   - [1] v: 轮子速度或速度误差
 *   - [2] yaw_err: 偏航角误差 (限制在[-π, π])
 *   - [3] yaw_dot: 偏航角速度
 *   - [4] pitch-(phi_l-target): 俯仰-左摆角误差
 *   - [5] pitch_dot-d_phi_l: 俯仰角速度 - 左摆角速度
 *   - [6] pitch-(phi_r-target): 俯仰-右摆角误差
 *   - [7] pitch_dot-d_phi_r: 俯仰角速度 - 右摆角速度
 *   - [8] pitch: 机体俯仰角
 *   - [9] pitch_dot: 机体俯仰角速度
 *
 * 注意：这与蹲伏状态和 lqr.h 中的定义都不同！
 * 增益矩阵K_1-K_6需要对应此处的状态定义。
 */
namespace standing_lqr {
// Q = diag([1 10 10 1 50 10 50 10 200 10 ])
// R = diag([30 30 1 1 ])

inline os::math::Matrix<4, 10> const K_1 {-0.121959, -0.518759, -0.404122, -0.140008, -1.329065, -0.441004, -1.134889, -0.359319, -0.310679, -0.094935, -0.121959, -0.518759, 0.404122, 0.140008, -1.134889, -0.359319, -1.329065, -0.441004, -0.310679, -0.094935, 0.110988, 0.456607, -0.312696, -0.109935, 4.757876, 1.948687, -2.785086, -1.296972, -9.863665, -2.210867, 0.110988, 0.456607, 0.312696, 0.109935, -2.785086, -1.296972, 4.757876, 1.948687, -9.863665, -2.210867};

inline os::math::Matrix<4, 10> const K_2 {-0.065039, -0.140114, -0.018691, -0.019238, -6.731670, -0.735098, 3.780713, 0.889981, 0.599083, 0.182750, 0.023211, 0.212352, -0.018081, -0.019390, -6.649779, -0.811033, 3.898983, 0.795345, 0.299129, 0.121166, 1.754185, 6.777418, -0.059613, -0.010310, 4.983177, -2.602665, -0.220784, -1.984026, -4.356437, -0.935880, -2.048899, -8.093271, -0.023832, -0.006945, -5.590612, 1.252729, -2.948629, 1.553906, 3.717113, 0.732317};

inline os::math::Matrix<4, 10> const K_3 {0.131036, 0.469157, 0.022067, 0.018981, 5.840982, 0.050105, -3.340786, -0.745451, -0.229428, -0.080760, 0.003671, -0.032584, 0.022852, 0.020883, 6.059950, 0.211392, -3.785580, -0.659395, 0.183852, 0.002724, -2.732029, -10.576502, 0.140975, 0.054397, -4.138140, 3.220315, -5.541532, 1.575966, 4.616021, 0.905128, 2.406023, 9.391534, 0.028021, -0.012642, 0.083819, -2.352441, 6.327016, -1.571780, -3.931561, -0.726002};

inline os::math::Matrix<4, 10> const K_4 {0.023211, 0.212352, 0.018081, 0.019390, 3.898983, 0.795345, -6.649779, -0.811033, 0.299129, 0.121166, -0.065039, -0.140114, 0.018691, 0.019238, 3.780713, 0.889981, -6.731670, -0.735098, 0.599083, 0.182750, -2.048899, -8.093271, 0.023832, 0.006945, -2.948629, 1.553906, -5.590612, 1.252729, 3.717113, 0.732317, 1.754185, 6.777418, 0.059613, 0.010310, -0.220784, -1.984026, 4.983177, -2.602665, -4.356437, -0.935880};

inline os::math::Matrix<4, 10> const K_5 {0.003671, -0.032584, -0.022852, -0.020883, -3.785580, -0.659395, 6.059950, 0.211392, 0.183852, 0.002724, 0.131036, 0.469157, -0.022067, -0.018981, -3.340786, -0.745451, 5.840982, 0.050105, -0.229428, -0.080760, 2.406023, 9.391534, -0.028021, 0.012642, 6.327016, -1.571780, 0.083819, -2.352441, -3.931561, -0.726002, -2.732029, -10.576502, -0.140975, -0.054397, -5.541532, 1.575966, -4.138140, 3.220315, 4.616021, 0.905128};

inline os::math::Matrix<4, 10> const K_6 {-0.075372, -0.619517, -0.011897, -0.002696, 1.122222, -0.079961, 0.244446, -0.296432, -0.758269, -0.245557, -0.075372, -0.619517, 0.011897, 0.002696, 0.244446, -0.296432, 1.122222, -0.079961, -0.758269, -0.245557, 0.572129, 2.435558, 0.579480, 0.173314, 4.237371, 0.841467, 0.231238, -0.095799, 0.131593, 0.093596, 0.572129, 2.435558, -0.579480, -0.173314, 0.231238, -0.095799, 4.237371, 0.841467, 0.131593, 0.093596};

inline control::LQR<10, 4> lqr{K_1, K_2, K_3, K_4, K_5, K_6};
}  // namespace standing_lqr

inline control::LQRInput<10> debug_wyc;
inline float debug_output_wyc[4];

class Standing final : public os::State {
    void onEnter() override {
        targetL_ = 0.10f;  // 站立时腿部目标长度 (设置为 0.1 m)

        // 初始化腿长 PID 控制器
        legLengthPid_.reset();
    }

    void onExecute() override {
        action::updateAllData();

        // 获取腿部状态
        float leftL = getLeftLeg().getL();
        float rightL = getRightLeg().getL();
        float leftPhi = -getLeftLeg().getPhi();          // ← 反向符号
        float rightPhi = -getRightLeg().getPhi();        // ← 反向符号
        float leftPhiDot = -getLeftLeg().getPhiDot();    // ← 反向符号
        float rightPhiDot = -getRightLeg().getPhiDot();  // ← 反向符号

        // 获取轮子速度
        float leftWheelVel = getLeftWheel().getVelocity();
        float rightWheelVel = getRightWheel().getVelocity();
        float wheelSpeedAvg = (leftWheelVel + rightWheelVel) * 0.5f;

        // 获取 INS 数据
        auto pitch = ins::getGimbalPitch().convert<units::angle::radian>().to<float>();
        auto pitchDot = ins::getPitchSpeed().convert<units::angular_velocity::radians_per_second>().to<float>();
        auto yaw_err = std::fmod(
                           playerSettings.chassis_ref.yaw.to<float>() -
                               ins::getGimbalYaw().convert<units::angle::radian>().to<float>() + PI,
                           2.0f * PI
                       ) -
                       PI;  // 限制在 [-π, π]
        auto yaw_dot = ins::getYawSpeed().convert<units::angular_velocity::radians_per_second>().to<float>();
        auto forward_ref = playerSettings.chassis_ref.vy.to<float>();

        // ===== 腿长 PID 控制 =====
        // 计算平均腿长
        float avgL = (leftL + rightL) * 0.5f;

        // 使用 PID 控制器计算所需的腿长控制力（用于两腿的共同轴向控制）
        legLengthPid_.setFeedback(avgL);
        legLengthPid_.setReference(targetL_);

        // 如果误差过大，重置积分器防止超调
        if (std::fabs(legLengthPid_.getError()) > 0.05f) {
            legLengthPid_.integrator().reset();
        }

        float legLengthForce = legLengthPid_.calculate();

        // ===== 判断是否处于静止状态 =====
        // 静止状态：速度很小且前进指令很小
        bool require_still = std::abs(wheelSpeedAvg) < 0.4f && std::abs(forward_ref) < 0.01f;

        // ===== 位置积分项（test_x）=====
        // 用于改进速度跟踪和减少稳态误差
        if (require_still) {
            // 静止时：积分位置偏差
            positionIntegral_ += std::abs(wheelSpeedAvg) < 0.1f ? wheelSpeedAvg * 0.001f : wheelSpeedAvg * 0.0007f;
            positionIntegral_ = std::clamp(positionIntegral_, -0.4f, 0.4f);
        } else {
            // 运动时：清零位置积分
            positionIntegral_ = 0.0f;
        }

        // ===== LQR 状态向量 =====
        // 参考 WheelLeg standing/normal.cpp 的实现
        // [0]: test_x (位置积分项)
        // [1]: v (速度或速度误差)
        // [2]: yaw_err
        // [3]: yaw_dot
        // [4]: pitch - (leftPhi - targetPhi)
        // [5]: pitchDot - leftPhiDot
        // [6]: pitch - (rightPhi - targetPhi)
        // [7]: pitchDot - rightPhiDot
        // [8]: pitch
        // [9]: pitchDot
        constexpr float targetPhi = 0.0f;  // 站立时目标摆角

        float state_v;
        if (require_still) {
            state_v = wheelSpeedAvg;
        } else {
            state_v = std::clamp(wheelSpeedAvg - forward_ref, -1.0f, 1.0f);
        }

        control::LQRInput<10> input{};
        input.x[0] = positionIntegral_;
        input.x[1] = state_v;
        input.x[2] = yaw_err;
        input.x[3] = yaw_dot;
        input.x[4] = pitch - (leftPhi - targetPhi);
        input.x[5] = pitchDot - leftPhiDot;
        input.x[6] = pitch - (rightPhi - targetPhi);
        input.x[7] = pitchDot - rightPhiDot;
        input.x[8] = pitch;
        input.x[9] = pitchDot;
        input.left_height = leftL;
        input.right_height = rightL;

        for (int i = 0; i < 10; ++i) {
            debug_wyc.x[i] = input.x[i];
        }
        debug_wyc.left_height = input.left_height;
        debug_wyc.right_height = input.right_height;

        // LQR 计算
        // 输出: [0] 左轮力矩, [1] 右轮力矩, [2] 左腿T, [3] 右腿T
        auto const& u = standing_lqr::lqr.process(input);

        // 提取输出
        float wheel_l = u(0, 0);
        float wheel_r = u(1, 0);
        float leg_T_l = u(2, 0);
        float leg_T_r = u(3, 0);

        debug_output_wyc[0] = wheel_l;
        debug_output_wyc[1] = wheel_r;
        debug_output_wyc[2] = leg_T_l;
        debug_output_wyc[3] = leg_T_r;

        // 低通滤波 + 阻尼 + 死区 + 速率限幅（参考 WheelLeg 设计）
        static float filtPitchDot = 0.0f;
        static float filtWheelV = 0.0f;
        static float last_wl = 0.0f;
        static float last_wr = 0.0f;
        constexpr float lpf_alpha = 0.995f;  // stronger smoothing (more phase lag)
        // 过滤俯仰角速度与轮速平均
        filtPitchDot = lpf_alpha * filtPitchDot + (1.0f - lpf_alpha) * pitchDot;
        filtWheelV = lpf_alpha * filtWheelV + (1.0f - lpf_alpha) * wheelSpeedAvg;

        // 阻尼项（吸收摆动能量）
        constexpr float kd = 2.0f;  // stronger damping
        wheel_l -= kd * filtPitchDot;
        wheel_r -= kd * filtPitchDot;

        // 死区：小信号置零，避免在平衡点被噪声驱动
        constexpr float wheel_dead = 0.08f;  // larger deadband
        if (std::abs(wheel_l) < wheel_dead) wheel_l = 0.0f;
        if (std::abs(wheel_r) < wheel_dead) wheel_r = 0.0f;

        // 速率限幅：限制每周期扭矩变化量
        constexpr float max_delta = 0.05f;  // N·m per cycle (slower)
        wheel_l = std::clamp(wheel_l, last_wl - max_delta, last_wl + max_delta);
        wheel_r = std::clamp(wheel_r, last_wr - max_delta, last_wr + max_delta);
        // 输出幅度缩放，降低控制激进度

        constexpr float wheel_scale = 0.35f;
        wheel_l *= wheel_scale;
        wheel_r *= wheel_scale;

        last_wl = wheel_l;
        last_wr = wheel_r;

        // 输出限幅
        constexpr float kWheelLimit = 8.0f;       // 站立时轮子力矩限幅
        constexpr float kLegTorqueLimit = 20.0f;  // 摆角力矩限幅
        wheel_l = std::clamp(wheel_l, -kWheelLimit, kWheelLimit);
        wheel_r = std::clamp(wheel_r, -kWheelLimit, kWheelLimit);
        leg_T_l = std::clamp(leg_T_l, -kLegTorqueLimit, kLegTorqueLimit);
        leg_T_r = std::clamp(leg_T_r, -kLegTorqueLimit, kLegTorqueLimit);

        // T: LQR 输出摆角力矩
        float left_T = -leg_T_l;
        float right_T = -leg_T_r;

        // F: 腿长 PID 力 + 重力补偿
        // 两腿独立计算但使用相同的 PID 输出作为基础
        constexpr float body_mass = 1.0f;  // 半边机体质量 kg
        constexpr float g = 9.81f;

        // 腿长轴向力 = PID 输出 + 重力补偿
        float left_F = legLengthForce + body_mass * g * std::cos(pitch - leftPhi + targetPhi);
        float right_F = legLengthForce + body_mass * g * std::cos(pitch - rightPhi + targetPhi);

        // 设置输出
        action::setLegForceTorque(left_T, left_F, right_T, right_F);
        action::applyLegOutput();

        action::setWheelTorque(wheel_l, wheel_r);
        action::applyWheelOutput();
    }

private:
    float targetL_{0.10f};
    float positionIntegral_{0.0f};  // 位置积分项，用于改进速度跟踪
    os::PID<> legLengthPid_{
        0.0f, 0.0f, 0.0f, 0.0f, os::pid::StandardI{0.0f}
    };  // kp, ki, kd, outputMax, integrator(i_max)
};

/**
 * @brief 调整状态 (Adjusting)
 *
 * 机身超出控制范围（倾倒或失控）时的恢复状态
 * 通过以下步骤恢复：
 * 1. 伸展腿部：增加腿长以获得更好的支撑
 * 2. 旋转腿部：调整腿的摆角以适应地面
 * 3. 矫正身体：使用 PID 控制矫正滚转角
 *
 * 成功条件：身体稳定且无倾倒信号，返回 Crouching 状态
 * 失败条件：调整超时，进入 Dead 状态
 */
class Adjusting final : public os::State {
    // 调整阶段枚举
    enum AdjustPhase : uint8_t {
        kExtendLeg = 0,   // 伸展腿部阶段
        kRotateLeg = 1,   // 旋转腿部阶段
        kCorrectBody = 2  // 矫正身体阶段
    };

    void onEnter() override {
        phase_ = kExtendLeg;
        phaseTimer_ = 0;
        adjustmentCompleted_ = false;
        bodyNotTippedCount_ = 0;

        // 初始化滚转 PID 控制器
        rollPid_.reset();
    }

    bool isAdjustmentCompleted() const { return adjustmentCompleted_; }

    void onExecute() override {
        action::updateAllData();

        // 获取传感器数据
        float leftL = getLeftLeg().getL();
        float rightL = getRightLeg().getL();
        float leftPhi = -getLeftLeg().getPhi();    // ← 反向符号
        float rightPhi = -getRightLeg().getPhi();  // ← 反向符号
        auto roll = ins::getGimbalRoll().convert<units::angle::radian>().to<float>();

        phaseTimer_++;

        switch (phase_) {
            case kExtendLeg: {
                // 伸展腿部：逐步增加腿长获得更好的支撑
                constexpr float legExtendThreshold = 0.30f;  // 目标腿长
                constexpr float extendForce = 4.0f;          // 伸展力

                float leftExtendForce = 0.0f;
                float rightExtendForce = 0.0f;

                if (leftL < legExtendThreshold) {
                    leftExtendForce = extendForce;
                } else if (leftL < 0.35f) {
                    leftExtendForce = extendForce * 0.25f;  // 近端减弱推力
                }

                if (rightL < legExtendThreshold) {
                    rightExtendForce = extendForce;
                } else if (rightL < 0.35f) {
                    rightExtendForce = extendForce * 0.25f;
                }

                action::setLegForceTorque(0.0f, leftExtendForce, 0.0f, rightExtendForce);
                action::applyLegOutput();
                action::setWheelTorque(0.0f, 0.0f);
                action::applyWheelOutput();

                // 检查两腿是否都已伸展
                bool leftExtended = leftL > legExtendThreshold;
                bool rightExtended = rightL > legExtendThreshold;

                if (leftExtended && rightExtended) {
                    phase_ = kRotateLeg;
                    phaseTimer_ = 0;
                }
                break;
            }

            case kRotateLeg: {
                // 旋转腿部：调整两腿摆角
                float angleDiffThreshold = 0.2f;  // 角度差阈值

                // 如果两腿角度接近，则进入矫正阶段
                if (std::fabs(leftPhi - rightPhi) < angleDiffThreshold) {
                    phase_ = kCorrectBody;
                    phaseTimer_ = 0;
                    bodyNotTippedCount_ = 0;
                } else {
                    // 调整摆角
                    float correctionTorque = 5.0f;
                    if (leftPhi > rightPhi) {
                        action::setLegForceTorque(-correctionTorque, 0.0f, correctionTorque, 0.0f);
                    } else {
                        action::setLegForceTorque(correctionTorque, 0.0f, -correctionTorque, 0.0f);
                    }
                    action::applyLegOutput();
                    action::setWheelTorque(0.0f, 0.0f);
                    action::applyWheelOutput();
                }
                break;
            }

            case kCorrectBody: {
                // 矫正身体：使用 PID 控制滚转角
                rollPid_.setFeedback(roll);
                rollPid_.setReference(0.0f);  // 目标滚转角为零

                float torqueCorrection = rollPid_.calculate();

                // 反向矫正倾倒
                int8_t correctionSign = roll > 0 ? -1 : 1;
                float baseTorque = 15.0f * correctionSign;

                // 继续伸展腿部维持支撑
                float extendForce = 0.0f;
                if (leftL < 0.35f) {
                    extendForce = 4.0f * 0.25f;
                }

                action::setLegForceTorque(
                    baseTorque + torqueCorrection, extendForce, baseTorque - torqueCorrection, extendForce
                );
                action::applyLegOutput();
                action::setWheelTorque(0.0f, 0.0f);
                action::applyWheelOutput();

                // 检查滚转角是否已矫正
                if (std::fabs(roll) < 0.1f) {
                    bodyNotTippedCount_++;
                    if (bodyNotTippedCount_ > 10) {
                        adjustmentCompleted_ = true;
                    }
                } else {
                    bodyNotTippedCount_ = 0;
                }
                break;
            }

            default:
                action::stopAllOutput();
                break;
        }

        // 超时检测：超过 10 秒则认为调整失败
        if (phaseTimer_ > 10000) {
            adjustmentCompleted_ = true;
        }
    }

private:
    AdjustPhase phase_{kExtendLeg};
    uint32_t phaseTimer_{0};
    bool adjustmentCompleted_{false};
    uint32_t bodyNotTippedCount_{0};

    // 滚转角 PID 控制器
    os::PID<> rollPid_{10.0f, 0.0f, 0.0f, 5.0f, os::pid::StandardI{0.0f}};
};
};  // namespace state

}  // namespace chassis::logic

/**
 * @brief 底盘状态机
 *
 * 状态转换通过事件触发：
 * - InitDone: 初始化完成，进入蹲伏
 * - WantStandUp: 想要站立，进入站立状态
 * - WantCrouch: 想要蹲下
 * - BodyTipped: 机身倾倒，进入调整状态
 * - LegError: 腿部错误，进入死亡状态
 */

// FSM 定义在全局命名空间
inline auto chassisFsm = os::make_StateMachine<chassis::logic::state::Dead>(
    (STATE(chassis::logic::state::Dead) + EVENT(chassis::logic::event::InitDone)) >>
        STATE(chassis::logic::state::Crouching),  // 目前不考虑走地鸡
    (STATE(chassis::logic::state::Crouching) + EVENT(chassis::logic::event::LegError)) >>
        STATE(chassis::logic::state::Dead),
    (STATE(chassis::logic::state::Crouching) + EVENT(chassis::logic::event::BodyTipped)) >>
        STATE(chassis::logic::state::Adjusting),
    (STATE(chassis::logic::state::Crouching) + EVENT(chassis::logic::event::WantStandUp)) >>
        STATE(chassis::logic::state::Standing),
    (STATE(chassis::logic::state::Standing) + EVENT(chassis::logic::event::LegError)) >>
        STATE(chassis::logic::state::Dead),
    (STATE(chassis::logic::state::Standing) + EVENT(chassis::logic::event::BodyTipped)) >>
        STATE(chassis::logic::state::Adjusting),
    (STATE(chassis::logic::state::Standing) + EVENT(chassis::logic::event::WantCrouch)) >>
        STATE(chassis::logic::state::Crouching),
    (STATE(chassis::logic::state::Adjusting) + EVENT(chassis::logic::event::AdjustDone)) >>
        STATE(chassis::logic::state::Crouching),
    (STATE(chassis::logic::state::Adjusting) + EVENT(chassis::logic::event::LegError)) >>
        STATE(chassis::logic::state::Dead),
    (STATE(chassis::logic::state::Adjusting) + EVENT(chassis::logic::event::BodyTipped)) >>
        STATE(chassis::logic::state::Dead),
    (STATE(chassis::logic::state::Standing) + EVENT(chassis::logic::event::WantCrouch)) >>
        STATE(chassis::logic::state::Crouching),
    (STATE(chassis::logic::state::Standing) + EVENT(chassis::logic::event::LegError)) >>
        STATE(chassis::logic::state::Dead),
    (STATE(chassis::logic::state::Standing) + EVENT(chassis::logic::event::BodyTipped)) >>
        STATE(chassis::logic::state::Dead)
);

namespace chassis::logic {

/**
 * @brief 获取 Adjusting 状态的指针
 * 用于在主循环中检测调整完成事件
 */
inline chassis::logic::state::Adjusting* getAdjustingState() {
    // 注意：这需要 FSM 提供访问状态的接口
    // 临时实现：返回 nullptr，实际应由 FSM 框架提供
    return nullptr;  // TODO: 实现 FSM 状态访问接口
}

}  // namespace chassis::logic

#endif  // CHASSIS_LOGIC_H
