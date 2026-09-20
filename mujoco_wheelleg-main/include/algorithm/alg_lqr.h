#ifndef LQR_H
#define LQR_H

#ifdef __cplusplus

#include <Eigen/Dense>
#include <cstdint>
#include <array>

namespace control {

/**
 * @brief LQR控制器输入结构
 *
 * @tparam xLength 状态向量维度
 */
template <uint16_t xLength>
struct LQRInput {
    float x[xLength];
    float left_height;
    float right_height;
};

/**
 * @brief 轮腿机器人状态向量定义 (10维)
 */
struct WheelLegState {
    float theta{0.0f};
    float d_theta{0.0f};

    float phi_l{0.0f};
    float d_phi_l{0.0f};
    float x_l{0.0f};
    float dx_l{0.0f};

    float phi_r{0.0f};
    float d_phi_r{0.0f};
    float x_r{0.0f};
    float dx_r{0.0f};

    void toArray(float* arr) const {
        arr[0] = theta; arr[1] = d_theta; arr[2] = phi_l; arr[3] = d_phi_l; arr[4] = x_l;
        arr[5] = dx_l; arr[6] = phi_r; arr[7] = d_phi_r; arr[8] = x_r; arr[9] = dx_r;
    }

    void fromArray(const float* arr) {
        theta = arr[0]; d_theta = arr[1]; phi_l = arr[2]; d_phi_l = arr[3]; x_l = arr[4];
        dx_l = arr[5]; phi_r = arr[6]; d_phi_r = arr[7]; x_r = arr[8]; dx_r = arr[9];
    }
};

/**
 * @brief 轮腿机器人控制输出 (4维)
 */
struct WheelLegControl {
    float T_l{0.0f};
    float T_r{0.0f};
    float F_l{0.0f};
    float F_r{0.0f};

    template<int R, int C>
    void fromMatrix(const Eigen::Matrix<float, R, C>& mat) {
        static_assert(R == 4 && C == 1, "fromMatrix expects a 4x1 vector");
        T_l = mat(0,0);
        T_r = mat(1,0);
        F_l = mat(2,0);
        F_r = mat(3,0);
    }
};

template<int Rows, int Cols>
using Matrixf = Eigen::Matrix<float, Rows, Cols>;

/**
 * @brief LQR控制器类
 *
 * 使用基于腿长的增益调度
 */
template <uint16_t xLength, uint16_t uLength>
class LQR {
public:
    using KMatrix = Matrixf<uLength, xLength>;
    using XVector = Matrixf<xLength, 1>;
    using UVector = Matrixf<uLength, 1>;

    const KMatrix& K_0_;
    const KMatrix& K_l_;
    const KMatrix& K_r_;
    const KMatrix& K_ll_;
    const KMatrix& K_lr_;
    const KMatrix& K_rr_;

    float left_height{0.0f};
    float right_height{0.0f};

    KMatrix K;
    KMatrix implementK_;

    XVector x;
    UVector u;

    LQR() = delete;
    LQR(const KMatrix& K_0,
        const KMatrix& K_l,
        const KMatrix& K_ll,
        const KMatrix& K_r,
        const KMatrix& K_rr,
        const KMatrix& K_lr)
        : K_0_(K_0), K_l_(K_l), K_r_(K_r), K_ll_(K_ll), K_lr_(K_lr), K_rr_(K_rr) {}
    ~LQR() = default;

    void calculate() {
        K = K_0_ + K_l_ * left_height + K_r_ * right_height + K_ll_ * left_height * left_height
            + K_lr_ * left_height * right_height + K_rr_ * right_height * right_height;
        u = -K * x;
    }

    void clear() {
        x.setZero();
        u.setZero();
    }

    const UVector& process(const LQRInput<xLength>& input) {
        left_height = input.left_height;
        right_height = input.right_height;
        for (uint16_t i = 0; i < xLength; ++i) x(i,0) = input.x[i];
        calculate();
        return u;
    }

    WheelLegControl process(const WheelLegState& state, float l_height, float r_height) {
        static_assert(xLength == 10 && uLength == 4, "This method requires 10-state 4-control LQR");
        left_height = l_height;
        right_height = r_height;
        float tmp[10];
        state.toArray(tmp);
        for (uint16_t i = 0; i < xLength; ++i) x(i,0) = tmp[i];
        calculate();
        WheelLegControl ctrl;
        ctrl.fromMatrix(u);
        return ctrl;
    }

    void reset() { u.setZero(); }

    void setState(uint16_t index, float value) {
        if (index < xLength) x(index,0) = value;
    }

    float getOutput(uint16_t index) const {
        if (index < uLength) return u(index,0);
        return 0.0f;
    }
};

using WheelLegLQR = LQR<10, 4>;

} // namespace control

#endif // __cplusplus

#endif // LQR_H
