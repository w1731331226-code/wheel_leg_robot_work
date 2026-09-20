#include "include/chassis.h"
#include "algorithm/matrix.h"
#include "clock.h"
#include "include/chassis_logic.h"
#include "include/vmc.h"
#include "interface.h"
#include "main.h"
#include "sheriffos.h"
#include <algorithm>
#include <cmath>

namespace chassis {

// CAN callback registration

InitLate(reg_left_leg_group) {
    leftLegCanPort::instance().onReceive_.connect([](CanRxHeader const* header, uint8_t const* buffer) {
        leftFrontJointMotor.decode(header->Identifier, buffer);
        leftBackJointMotor.decode(header->Identifier, buffer);
        leftWheelMotor.decode(header->Identifier, buffer);
    });
}

InitLate(reg_right_leg_group) {
    rightLegCanPort::instance().onReceive_.connect([](CanRxHeader const* header, uint8_t const* buffer) {
        rightFrontJointMotor.decode(header->Identifier, buffer);
        rightBackJointMotor.decode(header->Identifier, buffer);
        rightWheelMotor.decode(header->Identifier, buffer);
    });
}

// Simple sliding window template

template <typename T, size_t N>
class DataBuffer {
public:
    void record(T value) {
        for (size_t i = N - 1; i > 0; --i) {
            data_[i] = data_[i - 1];
        }
        data_[0] = value;
    }

    T operator[](size_t i) const { return data_[i]; }
    T& operator[](size_t i) { return data_[i]; }

private:
    T data_[N] = {};
};

// Leg class implementation

template <typename FrontJointMotor, typename BackJointMotor>
class Leg final : public ILeg {
public:
    Leg(FrontJointMotor& frontJointMotor,
        BackJointMotor& backJointMotor,
        Direction direction,
        units::angle::radian_t frontAngleOffset,
        units::angle::radian_t backAngleOffset)
        : frontJointMotor_(frontJointMotor),
          backJointMotor_(backJointMotor),
          direction_(direction),
          frontAngleOffset_(frontAngleOffset),
          backAngleOffset_(backAngleOffset) {}

    ~Leg() override = default;

    void initialize() override {
        if (initialized_) return;
        if (frontJointMotor_.isEnable() && backJointMotor_.isEnable()) {
            updateData();
            updateData();
            initialized_ = true;
            return;
        }
        frontJointMotor_.enable();
        backJointMotor_.enable();
    }

    void enableLeg() override {
        frontJointMotor_.enable();
        backJointMotor_.enable();
        initialized_ = frontJointMotor_.isEnable() && backJointMotor_.isEnable();
    }

    void updateData() override {
        if (direction_ == Direction::kOpposite) {
            updateDataOpposite();
        } else {
            updateDataSame();
        }
    }

    void setOutputWithTAndF() override {
        // Map virtual hip torque (T) and axial leg force (F) to the two joint actuator torques
        // using the VMC Jacobian: tau_theta = J^T * [T; F]
        constexpr float max_output_torque = 4.0f;

        // compute Jacobian at current joint angles (theta1, theta2)
        chassis::VMC vmc(L1, L2, L1, L2);
        vmc.updateJacobian(data_.theta1, data_.theta2);
        auto J = vmc.getJacobian();

        Eigen::Vector2f g; // [T; F]
        g(0) = output_.T_F_torque_(0, 0);
        g(1) = output_.T_F_torque_(1, 0);
        Eigen::Vector2f tau = J.transpose() * g; // actuator torques for [theta1, theta2]

        float front = tau(0);
        float back = tau(1);
        float sum = std::abs(front) + std::abs(back);
        if (sum > max_output_torque) {
            float scale = max_output_torque / sum;
            front *= scale;
            back *= scale;
        }

        if (direction_ == Direction::kOpposite) {
            sendMotorMIT(frontJointMotor_, -front);
            sendMotorMIT(backJointMotor_, -back);
        } else {
            sendMotorMIT(frontJointMotor_, front);
            sendMotorMIT(backJointMotor_, back);
        }
    }

    void setOutputDirectly() override {
        constexpr float max_output_torque = 15.0f;

        float front = output_.front_back_torque_(0, 0);
        float back = output_.front_back_torque_(1, 0);
        float sum = std::abs(front) + std::abs(back);
        if (sum > max_output_torque) {
            float scale = max_output_torque / sum;
            front *= scale;
            back *= scale;
        }

        // Enforce per-motor torque limit ±2.0 N·m (hip joint limits)
        front = std::clamp(front, -1.0f, 1.0f);
        back = std::clamp(back, -1.0f, 1.0f);

        if (direction_ == Direction::kOpposite) {
            sendMotorMIT(frontJointMotor_, -front);
            sendMotorMIT(backJointMotor_, -back);
        } else {
            sendMotorMIT(frontJointMotor_, front);
            sendMotorMIT(backJointMotor_, back);
        }
    }

    void stopOutput() override {
        frontJointMotor_.disable();
        backJointMotor_.disable();
        initialized_ = false;
    }

    [[nodiscard]] bool isInitialized() const override { return initialized_; }
    [[nodiscard]] float getL() const override { return data_.L; }
    [[nodiscard]] float getLDot() const override { return data_.L_dot; }
    [[nodiscard]] float getPhi() const override { return data_.phi; }
    [[nodiscard]] float getPhiDot() const override { return data_.phi_dot; }
    [[nodiscard]] float getTheta1() const override { return data_.theta1; }
    [[nodiscard]] float getTheta2() const override { return data_.theta2; }
    [[nodiscard]] float getTheta1Dot() const override { return data_.theta1_dot; }
    [[nodiscard]] float getTheta2Dot() const override { return data_.theta2_dot; }
    [[nodiscard]] float getFPending() const override { return data_.F_pending; }

    void setForceTorque(float T, float F) override {
        output_.T_F_torque_[0][0] = T;
        output_.T_F_torque_[1][0] = F;
    }

    void setDirectTorque(float front, float back) override {
        output_.front_back_torque_[0][0] = front;
        output_.front_back_torque_[1][0] = back;
    }

private:
    template <typename Motor>
    void sendMotorMIT(Motor& motor, float torque) {
        motor.setSendMIT(0_deg, 0_rpm, 0, 0, units::torque::newton_meter_t{torque});
        // motor.setSendMIT(0_deg, 0_rpm, 0, 0, 0_Nm);

    }

    void updateDataOpposite() {
        auto frontData = frontJointMotor_.encoder_.getData();
        auto backData = backJointMotor_.encoder_.getData();

        units::angle::radian_t theta1 = -frontData.continuousAngle + frontAngleOffset_;
        units::angle::radian_t theta2 = -backData.continuousAngle + backAngleOffset_;

        theta1_history_.record(theta1.template to<float>());
        theta2_history_.record(theta2.template to<float>());

        float theta1_dot = -frontData.shaftSpeed.template to<float>() * kSpeedRatioRpmToRadS;
        float theta2_dot = -backData.shaftSpeed.template to<float>() * kSpeedRatioRpmToRadS;

        float theta1_f = theta1.template to<float>();
        float theta2_f = theta2.template to<float>();
        vmc_.updateJacobian(theta1_f, theta2_f);
        J_ = vmc_.getJacobian();

        // ✅ 验证雅可比矩阵不接近奇异点
        float det = vmc_.getJacobianDeterminant();
        if (std::abs(det) < 1e-3f) {
            // 警告：接近奇异位置，控制可能变得不稳定
            // DEBUG_RTT_PRINTF(0, "[Leg] 警告：雅可比接近奇异！det=%.6f\n", det);
        }

        float L, phi;
        vmc_.getLFromPhi(theta1_f, theta2_f, L, phi);

        auto thetaDotVec = os::math::Matrix<2, 1>({theta1_dot, theta2_dot});
        auto LPhiDot = J_ * thetaDotVec;
        float L_dot = LPhiDot(0, 0);    // 第0行是L的导数
        float phi_dot = LPhiDot(1, 0);  // 第1行是phi的导数

        data_.L = L;
        data_.L_dot = L_dot;
        data_.phi = phi;
        data_.phi_dot = phi_dot;
        data_.theta1 = theta1_f;
        data_.theta2 = theta2_f;
        // debug: joint angle sum and difference
        data_.joint_angle_sum = theta1_f + theta2_f;
        data_.joint_angle_diff = theta2_f - theta1_f;
        data_.theta1_dot = theta1_dot;
        data_.theta2_dot = theta2_dot;
        data_.F_pending = frontData.torque.template to<float>() + backData.torque.template to<float>();
    }

    void updateDataSame() {
        auto frontData = frontJointMotor_.encoder_.getData();
        auto backData = backJointMotor_.encoder_.getData();

        units::angle::radian_t theta1 = frontData.continuousAngle + frontAngleOffset_;
        units::angle::radian_t theta2 = backData.continuousAngle + backAngleOffset_;

        theta1_history_.record(theta1.template to<float>());
        theta2_history_.record(theta2.template to<float>());

        float theta1_dot = frontData.shaftSpeed.template to<float>() * kSpeedRatioRpmToRadS;
        float theta2_dot = backData.shaftSpeed.template to<float>() * kSpeedRatioRpmToRadS;

        float theta1_f = theta1.template to<float>();
        float theta2_f = theta2.template to<float>();
        vmc_.updateJacobian(theta1_f, theta2_f);
        J_ = vmc_.getJacobian();

        // ✅ 验证雅可比矩阵不接近奇异点
        float det = vmc_.getJacobianDeterminant();
        if (std::abs(det) < 1e-3f) {
            // 警告：接近奇异位置，控制可能变得不稳定
            // DEBUG_RTT_PRINTF(0, "[Leg] 警告：雅可比接近奇异！det=%.6f\n", det);
        }

        float L, phi;
        vmc_.getLFromPhi(theta1_f, theta2_f, L, phi);

        auto thetaDotVec = os::math::Matrix<2, 1>({theta1_dot, theta2_dot});
        auto LPhiDot = J_ * thetaDotVec;
        float L_dot = LPhiDot(0, 0);    // 第0行是L的导数
        float phi_dot = LPhiDot(1, 0);  // 第1行是phi的导数

        data_.L = L;
        data_.L_dot = L_dot;
        data_.phi = phi;
        data_.phi_dot = phi_dot;
        data_.theta1 = theta1_f;
        data_.theta2 = theta2_f;
        // debug: joint angle sum and difference
        data_.joint_angle_sum = theta1_f + theta2_f;
        data_.joint_angle_diff = theta2_f - theta1_f;
        data_.theta1_dot = theta1_dot;
        data_.theta2_dot = theta2_dot;
        data_.F_pending = frontData.torque.template to<float>() + backData.torque.template to<float>();
    }

private:
    FrontJointMotor& frontJointMotor_;
    BackJointMotor& backJointMotor_;
    Direction direction_;
    units::angle::radian_t frontAngleOffset_;
    units::angle::radian_t backAngleOffset_;

    VMC vmc_{L1, L2, L1, L2};

    bool initialized_{false};

    struct LegData {
        float L{0.0f};
        float L_dot{0.0f};
        float phi{0.0f};
        float phi_dot{0.0f};
        float theta1{0.0f};
        float theta2{0.0f};
        // debug: sum and difference of joint angles (radians)
        float joint_angle_sum{0.0f};
        float joint_angle_diff{0.0f};
        float theta1_dot{0.0f};
        float theta2_dot{0.0f};
        float F_pending{0.0f};
    } data_;

    struct LegOutput {
        os::math::Matrix<2, 1> T_F_torque_{};
        os::math::Matrix<2, 1> front_back_torque_{};
    } output_;

    os::math::Matrix<2, 2> J_{};
    os::math::Matrix<2, 1> output_from_T_F_{};

    static constexpr float kSpeedRatioRpmToRadS = 2.0f * 3.14159265359f / 60.0f;
    DataBuffer<float, 5> theta1_history_;
    DataBuffer<float, 5> theta2_history_;
};

// Wheel class implementation

template <typename WheelMotor>
class Wheel final : public IWheel {
public:
    Wheel(WheelMotor& motor, Direction direction) : motor_(motor), direction_(direction) {}

    ~Wheel() override = default;

    void initialize() override {
        if (initialized_) return;
        if (motor_.isEnable()) {
            updateData();
            updateData();
            initialized_ = true;
            return;
        }
        motor_.enable();
    }

    void enableWheel() override {
        motor_.enable();
        initialized_ = motor_.isEnable();
    }

    void updateData() override {
        auto wheelData = motor_.encoder_.getData();

        float angle = wheelData.continuousAngle.template to<float>();
        float speed = wheelData.shaftSpeed.template to<float>() * kSpeedRatioRpmToRadS;

        if (direction_ == Direction::kOpposite) {
            angle = -angle;
            speed = -speed;
        }

        wheel_angle_history_.record(angle);
        wheel_speed_history_.record(speed);

        data_.angle = angle;
        data_.speed = speed;
        data_.distance = angle * kWheelRadius;
        data_.velocity = speed * kWheelRadius;
    }

    void setOutput(float torque) override { output_ = std::clamp(torque, -kMaxWheelTorque, kMaxWheelTorque); }

    void applyOutput() override {
        float out = output_;
        if (direction_ == Direction::kOpposite) {
            out = -out;
        }
        motor_.setSendMIT(0_deg, 0_rpm, 0, 0, units::torque::newton_meter_t{out});
        // motor_.setSendMIT(0_deg, 0_rpm, 0, 0, 0_Nm);

    }

    void stopOutput() override {
        motor_.disable();
        initialized_ = false;
    }

    [[nodiscard]] bool isInitialized() const override { return initialized_; }
    [[nodiscard]] float getAngle() const override { return data_.angle; }
    [[nodiscard]] float getSpeed() const override { return data_.speed; }
    [[nodiscard]] float getDistance() const override { return data_.distance; }
    [[nodiscard]] float getVelocity() const override { return data_.velocity; }
    [[nodiscard]] float getOutput() const override { return output_; }

private:
    WheelMotor& motor_;
    Direction direction_;
    bool initialized_{false};

    struct WheelData {
        float angle{0.0f};
        float speed{0.0f};
        float distance{0.0f};
        float velocity{0.0f};
    } data_;

    float output_{0.0f};

    static constexpr float kSpeedRatioRpmToRadS = 2.0f * 3.14159265359f / 60.0f;
    static constexpr float kMaxWheelTorque = 1.0f;

    DataBuffer<float, 5> wheel_angle_history_;
    DataBuffer<float, 5> wheel_speed_history_;
};

// Global interface functions

ILeg& getLeftLeg() {
    static Leg<decltype(leftFrontJointMotor), decltype(leftBackJointMotor)> leftLeg{
        leftFrontJointMotor,
        leftBackJointMotor,
        Direction::kSame,  // ← 改成 kSame
        kLeftFrontAngleOffset,
        kLeftBackAngleOffset
    };
    return leftLeg;
}
ILeg& getRightLeg() {
    static Leg<decltype(rightFrontJointMotor), decltype(rightBackJointMotor)> rightLeg{
        rightFrontJointMotor,
        rightBackJointMotor,
        Direction::kOpposite,  // ← 改成 kOpposite
        kRightFrontAngleOffset,
        kRightBackAngleOffset
    };
    return rightLeg;
}
IWheel& getLeftWheel() {
    static Wheel<decltype(leftWheelMotor)> leftWheel{leftWheelMotor, Direction::kSame};
    return leftWheel;
}
IWheel& getRightWheel() {
    static Wheel<decltype(rightWheelMotor)> rightWheel{rightWheelMotor, Direction::kOpposite};
    return rightWheel;
}

}  // namespace chassis

uint8_t test_wyc = 0;

// Control thread using CREATE_THREAD_STATIC macro

CREATE_THREAD_STATIC(chassisCtrl, 1024, NULL, 5) {
    os::sleep(1000_ms);

    // ===== 状态机事件触发的辅助变量 =====
    bool lastWantStandUpCmd = false;
    bool lastWantCrouchCmd = false;

    for (;;) {
        ::chassisFsm.execute();

        // ===== 事件触发逻辑 =====

        // 1. 初始化完成事件
        if (!chassis::logic::action::isAllInitialized()) {
            chassis::logic::action::initializeAll();
                os::sleep(2_ms);
            continue;
        }
        // switch (playerSettings.chassis_mode) {
            // case chassis::ChassisMode::kDead:
                // chassisFsm.checkoutTo<chassis::logic::state::Dead>();
            // case chassis::ChassisMode::kCrouching:
                chassisFsm.checkoutTo<chassis::logic::state::Crouching>();
            // default:
                // break;
        // }
        // 2. 站立/蹲伏命令事件
        // 从遥控器或其他控制模块获取命令
        bool wantStandUp = false;  // TODO: 从遥控器读取
        bool wantCrouch = false;   // TODO: 从遥控器读取

        // 上升沿触发：从不想站立 -> 想站立
        if (wantStandUp && !lastWantStandUpCmd) {
            ::chassisFsm.react<chassis::logic::event::WantStandUp>();
        }
        lastWantStandUpCmd = wantStandUp;

        // 上升沿触发：从不想蹲伏 -> 想蹲伏
        if (wantCrouch && !lastWantCrouchCmd) {
            ::chassisFsm.react<chassis::logic::event::WantCrouch>();
        }
        lastWantCrouchCmd = wantCrouch;

        // 3. Adjusting 完成事件
        // 注意：这需要 FSM 框架提供获取当前状态的接口
        // 当前的临时实现方式：
        // - Adjusting 状态内部设置 adjustmentCompleted_ 标志
        // - 在主循环中检测这个标志并触发事件
        //
        // 实现方案 1（推荐）：在 Adjusting::onExit() 中自动触发事件
        // 实现方案 2：添加 FSM 状态访问接口，在主循环中轮询检测
        // 实现方案 3：使用状态转移的自动触发机制（需要 FSM 支持）

        // 临时方案：等待 FSM 框架完善
        // auto* adjustingState = chassis::logic::getAdjustingState();
        // if (adjustingState && adjustingState->isAdjustmentCompleted() && !lastAdjustmentCompleted) {
        //     chassis::logic::chassisFsm.react<chassis::logic::event::AdjustDone>();
        //     lastAdjustmentCompleted = true;
        // }

        // 4. 错误事件检测（从 INS、腿部等模块获取）
        // TODO: 检测机体是否倾倒 (pitch/roll 超过阈值)
        // bool bodyTipped = checkBodyTipped();
        // if (bodyTipped) {
        //     chassis::logic::chassisFsm.react<chassis::logic::event::BodyTipped>();
        // }

        // TODO: 检测腿部是否出错（电机故障、传感器异常等）
        // bool legError = checkLegError();
        // if (legError) {
        //     chassis::logic::chassisFsm.react<chassis::logic::event::LegError>();
        // }

        os::sleep(2_ms);
    }
}
