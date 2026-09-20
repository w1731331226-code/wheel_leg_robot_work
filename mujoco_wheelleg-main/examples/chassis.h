#ifndef CHASSIS_H
#define CHASSIS_H

#include "algorithm/matrix.h"
#include "algorithm/pid.h"
#include "algorithm/ringbuff.h"
#include "device/damiao_motor.h"
#include "device/dji_motor.h"
#include "units.h"

using leftLegCanPort = os::Can<1>;
using rightLegCanPort = os::Can<2>;

inline units::angle::radian_t normalize_angle(
    units::angle::radian_t const deg, units::angle::radian_t const angle_max
) {
    units::angle::radian_t res = os::math::fmod(deg + angle_max, 2 * angle_max);
    if (res < 0_deg) res += 2 * angle_max;
    return res - angle_max;
}

namespace chassis {

/*
 * chassis direction
 *    front             x
 * left    right      y z
 *    back
 *
 * 轮腿机器人坐标系定义：
 * - X轴：指向机器人前方
 * - Y轴：指向机器人左侧
 * - Z轴：垂直向上
 */

// 方向枚举，用于区分电机安装方向
enum class Direction : uint8_t { kSame = 0, kOpposite = 1 };

// 腿部物理参数
constexpr units::length::meter_t L1 = 0.11_m;   // 大腿长度
constexpr units::length::meter_t L2 = 0.132_m;  // 小腿长度
constexpr float kWheelRadius = 0.042f;          // 轮子半径 (m)
constexpr float kLegMass = 0.173f;              // 腿部质量 (kg)
constexpr float kBodyMass = 2.033f;             // 机身质量 (kg)
constexpr float kGravity = 9.81f;               // 重力加速度

// 电机安装偏置角
constexpr units::angle::radian_t kLeftFrontAngleOffset = 0.441176444_rad;
constexpr units::angle::radian_t kLeftBackAngleOffset = -0.538071573_rad;
constexpr units::angle::radian_t kRightFrontAngleOffset = 0.441176444_rad;
constexpr units::angle::radian_t kRightBackAngleOffset = -0.538071573_rad;

// ----- 左腿关节电机（前后两个）-----
inline auto leftFrontJointMotor = DamiaoMotor<leftLegCanPort>{0x01, 0x101, 1, 0_deg, 12.5_rad, 30_rad_per_s, 10_Nm};
inline auto leftBackJointMotor = DamiaoMotor<leftLegCanPort>{0x02, 0x102, 1, 0_deg, 12.5_rad, 30_rad_per_s, 10_Nm};

// ----- 右腿关节电机（前后两个）-----
inline auto rightFrontJointMotor = DamiaoMotor<rightLegCanPort>{0x03, 0x103, 1, 0_deg, 12.5_rad, 30_rad_per_s, 10_Nm};
inline auto rightBackJointMotor = DamiaoMotor<rightLegCanPort>{0x04, 0x104, 1, 0_deg, 12.5_rad, 30_rad_per_s, 10_Nm};

// ----- 轮子电机 -----
inline auto leftWheelMotor = DamiaoMotor<leftLegCanPort>{0x05, 0x105, 1, 0_deg, 12.5_rad, 45_rad_per_s, 10_Nm};
inline auto rightWheelMotor = DamiaoMotor<rightLegCanPort>{0x06, 0x106, 1, 0_deg, 12.5_rad, 45_rad_per_s, 10_Nm};

using JointMotorType = decltype(leftFrontJointMotor);
using WheelMotorType = decltype(leftWheelMotor);

/**
 * @brief 腿部接口类
 */
class ILeg {
public:
    virtual ~ILeg() = default;
    virtual void initialize() {};
    virtual void enableLeg() {};
    virtual void updateData()  {};
    virtual void setOutputWithTAndF() {};
    virtual void setOutputDirectly() {};
    virtual void stopOutput() {};

    // 获取腿部状态
    [[nodiscard]] virtual bool isInitialized() const {}
    [[nodiscard]] virtual float getL() const {}
    [[nodiscard]] virtual float getLDot() const {}
    [[nodiscard]] virtual float getPhi() const {}
    [[nodiscard]] virtual float getPhiDot() const {}
    [[nodiscard]] virtual float getTheta1() const {}
    [[nodiscard]] virtual float getTheta2() const {}
    [[nodiscard]] virtual float getTheta1Dot() const {}
    [[nodiscard]] virtual float getTheta2Dot() const {}
    [[nodiscard]] virtual float getFPending() const {}

    // 输出设置
    virtual void setForceTorque(float T, float F) {}
    virtual void setDirectTorque(float front, float back) {}
};

/**
 * @brief 轮子接口类
 */
class IWheel {
public:
    virtual ~IWheel() = default;
    virtual void initialize() {}
    virtual void enableWheel() {}
    virtual void updateData() {}
    virtual void setOutput(float torque) {}
    virtual void applyOutput() {}
    virtual void stopOutput() {}

    [[nodiscard]] virtual bool isInitialized() const {}
    [[nodiscard]] virtual float getAngle() const {}
    [[nodiscard]] virtual float getSpeed() const {}
    [[nodiscard]] virtual float getDistance() const {}
    [[nodiscard]] virtual float getVelocity() const {}
    [[nodiscard]] virtual float getOutput() const {}
};

// 全局腿部和轮子访问函数
ILeg& getLeftLeg();
ILeg& getRightLeg();
IWheel& getLeftWheel();
IWheel& getRightWheel();

/**
 * @brief 底盘模式
 */
enum class ChassisMode : uint8_t {
    kDead = 0,   // 死亡状态
    kCrouching,  // 蹲伏状态
    kStanding,   // 站立状态
    kSpinning,   // 小陀螺模式
};

/**
 * @brief 底盘控制参考值
 */
struct ChassisRef_t {
    units::velocity::meters_per_second_t vx{0.0f};             // X方向速度参考 (m/s)
    units::velocity::meters_per_second_t vy{0.0f};             // Y方向速度参考 (m/s)
    units::angle::radian_t yaw{0.0_deg};                            // yaw角度期望
    units::length::meter_t left_height{0.086f};   // 左腿高度参考 (m)
    units::length::meter_t right_height{0.086f};  // 右腿高度参考 (m)
};

/**
 * @brief 底盘控制器接口
 */
class IChassisController {
public:
    virtual ~IChassisController() = default;
    virtual void update() = 0;
    virtual void setRef(ChassisRef_t const& ref) = 0;
    virtual ChassisRef_t getRef() const = 0;
};

/**
 * @brief 获取底盘控制器实例
 */
IChassisController& getChassisController();

}  // namespace chassis

#endif  // CHASSIS_H
