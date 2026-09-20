#pragma once
// 腿长插值 LQR 增益 (参照 balance-simulation LQR_calc.m)
// 状态 x = [theta, theta_dot, dist, vel, pitch, pitch_w]; u = -K(L)*x, K = a*L^2+b*L+c
// 参数: 髋距60mm 大腿50mm 小腿105mm 轮R25mm, 整机~1kg (每侧 mw=0.1 mp=0.064 M=0.345)
namespace wheelleg {
struct LegLqrGain { float a, b, c; };
// 行序: [T_wheel, T_hip] x [theta, theta_dot, dist, vel, pitch, pitch_w]
static const LegLqrGain kLqr[12] = {
  { 20.515659f, -14.102131f, -0.607494f},
  {-0.627967f, -2.445294f, -0.019919f},
  { 0.000000f, -0.000000f, -0.000000f},
  { 0.944587f, -0.309956f, -0.646640f},
  { 2.071300f, -0.664599f,  0.044743f},
  { 0.085134f, -0.018910f,  0.008408f},
  { 0.089594f, -0.073995f, -0.040579f},
  { 0.080777f, -0.080077f, -0.004617f},
  {-0.000000f,  0.000000f, -0.000000f},
  {-0.582427f,  0.230471f, -0.045618f},
  {-0.793362f,  0.281277f,  1.438598f},
  {-0.060962f,  0.021299f,  0.117535f},
};
}
