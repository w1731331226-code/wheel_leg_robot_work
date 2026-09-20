#ifndef VMC_H
#define VMC_H

#ifdef __cplusplus

#include <Eigen/Dense>
#include <cmath>

namespace chassis {

/**
 * 五连杆机构 (按 НИРСе.pdf 逆运动学公式 2-15/2-16 实现)
 *
 * 构型: 电机 O(0,0) 与 D(l5,0) 为固定杆两端; 主动杆 l1 从 O, 主动杆 l4 从 D;
 *       连杆 l2, l3 分别从两端主动杆末端汇于轮心 B(x,y)。
 *       角度约定: alpha/beta 自 +x 轴逆时针 (水平面下方为负)。
 *
 * 参数: l1=l4=50mm(主动杆), l2=l3=105mm(连杆), l5=60mm(电机间距)。
 */
class FiveBar {
public:
    FiveBar(float l1, float l2, float l3, float l4, float l5)
        : l1_(l1), l2_(l2), l3_(l3), l4_(l4), l5_(l5) {}

    // 正运动学: (alpha, beta) -> 轮心 (x, y) 与两个连杆角 theta1/theta2
    // branch = -1: 轮在电机轴下方 (站立支), +1: 上方支
    void Fk(float alpha, float beta, float& x, float& y,
            float& theta1, float& theta2, int branch = -1) const {
        const float ax = l1_ * std::cos(alpha), ay = l1_ * std::sin(alpha);
        const float cx = l5_ + l4_ * std::cos(beta), cy = l4_ * std::sin(beta);
        // 两圆交点 (A, l2) 与 (C, l3)
        float bx, by;
        CircleIntersect(ax, ay, l2_, cx, cy, l3_, branch, bx, by);
        x = bx; y = by;
        theta1 = std::atan2(by - ay, bx - ax);
        theta2 = std::atan2(by - cy, bx - cx);
    }

    // 逆运动学: 轮心 (x, y) -> (alpha, beta)。取站立支 (轮在电机轴下方)。
    void Ik(float x, float y, float& alpha, float& beta) const {
        // 论文 (2-15)/(2-16): 圆-圆求交的二次方程解
        float a = 2 * x * l1_;
        float b = 2 * y * l1_;
        float c = x * x + y * y + l1_ * l1_ - l2_ * l2_;
        float disc = a * a + b * b - c * c;
        if (disc < 0) { alpha = beta = 0; return; }
        float z1 = (b - std::sqrt(disc)) / (a + c);       // 站立支 (- 号)
        float z2 = (b + std::sqrt(disc)) / (a + c);
        // 优先取落在"轮在轴下方"这一支: 解析验证表明 z1 对应下方支
        float z = std::isfinite(z1) ? z1 : z2;
        alpha = 2 * std::atan(z);

        float d = 2 * (x - l5_) * l4_;
        float e = 2 * y * l4_;
        float f = (x - l5_) * (x - l5_) + y * y + l4_ * l4_ - l3_ * l3_;
        disc = d * d + e * e - f * f;
        if (disc < 0) { alpha = beta = 0; return; }
        float w1 = (e - std::sqrt(disc)) / (d + f);
        float w2 = (e + std::sqrt(disc)) / (d + f);
        // 与 alpha 同一支: beta = -(alpha) 的镜像支 → 对称时 (alpha + beta) = 180°
        // 取 |beta - (pi - alpha)| 更小者
        float b1 = 2 * std::atan(w1), b2 = 2 * std::atan(w2);
        beta = std::abs(b1 - (M_PI - alpha)) < std::abs(b2 - (M_PI - alpha)) ? b1 : b2;
    }

    // 雅可比 d(x,y)/d(alpha,beta)
    Eigen::Matrix2f Jacobian(float alpha, float beta) const {
        float x, y, t1, t2;
        Fk(alpha, beta, x, y, t1, t2);
        // 数值微分 (步长 1e-6 rad)
        const float h = 1e-6f;
        float xa, ya, t1a, t2a; Fk(alpha + h, beta, xa, ya, t1a, t2a);
        float xb, yb, t1b, t2b; Fk(alpha, beta + h, xb, yb, t1b, t2b);
        Eigen::Matrix2f J;
        J(0,0) = (xa - x) / h;  J(0,1) = (xb - x) / h;
        J(1,0) = (ya - y) / h;  J(1,1) = (yb - y) / h;
        return J;
    }

private:
    // 两圆交点: 圆心 (x1,y1) 半径 r1, 圆心 (x2,y2) 半径 r2; branch 决定选哪个交点
    void CircleIntersect(float x1, float y1, float r1, float x2, float y2,
                         float r2, int branch, float& xo, float& yo) const {
        float dx = x2 - x1, dy = y2 - y1;
        float dd = std::sqrt(dx * dx + dy * dy);
        if (dd < 1e-9) { xo = x1; yo = y1 - r1; return; }
        // 交点位于连线中点偏移
        float a = (r1 * r1 - r2 * r2 + dd * dd) / (2 * dd);
        float h2 = r1 * r1 - a * a;
        float h = h2 > 0 ? std::sqrt(h2) : 0.0f;
        float xm = x1 + a * dx / dd, ym = y1 + a * dy / dd;
        // 法向 (垂直于连线): (−dy, dx)/dd
        float nx = -dy / dd, ny = dx / dd;
        xo = xm + branch * h * nx;
        yo = ym + branch * h * ny;
    }

    float l1_, l2_, l3_, l4_, l5_;
};

}  // namespace chassis

#endif  // __cplusplus

#endif  // VMC_H
