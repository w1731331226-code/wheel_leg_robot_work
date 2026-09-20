#include "wheelleg.h"
#include "data.h"
#include "get_data.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

// =====================================================================
// 轮腿机器人控制器 —— 参照 gitee shuo_kai/balance-simulation 五连杆轮足算法
// 结构 (每步, 已在仿真验证站立平衡):
//   1) 五连杆正解 (每侧): 髋角 -> 腿长 leg_len / 腿角 theta / 轮心 C
//   2) 腿长控制:   F_leg = KP_LEG·(L_stand + dL - hgt) - KD_LEG·legd·cosθ + GRAV
//                  横滚补偿 dL = KP_R·roll + KD_R·roll_w (左右腿长差)
//   3) 腿角控制:   T_hub = KP_T·(θ_cmd−θ) − KD_T·θ̇  (轮毂力矩, θ_cmd = 速度摆+平台修正)
//   4) VMC 映射:   髋力矩 τ = Jᵀ·(F_leg·u) + T_hub·gθ  (J=∂C/∂q, 数值雅可比)
//   5) 平台恒平:   轮力矩 T_w = KP_P·φ + KD_P·φ̇  (无倾斜参考; 前进/刹车靠腿摆质心)
//   6) 横滚/腿角/腿长目标经左右腿独立执行
// 跳跃 (空格触发, 静止/前进均可): DRIVE -> SQUAT(下蹲到理论极限) -> JUMP(猛蹬)
//      -> FLY(腾空收腿) -> LAND(按起跳速度缩放急停) -> DRIVE
// 调参: 环境变量 WHEELLEG_KP_LEG / KD_LEG / KP_P / KD_P / KP_R / KD_R / KP_T / KD_T 等
// =====================================================================
namespace wheelleg {

// ---- 五连杆几何 (髋距60mm 大腿50mm 小腿105mm, 关节零位=站姿) ----
namespace {
constexpr float kL1 = 0.150f;      // 大腿
constexpr float kL2 = 0.270f;      // 小腿
constexpr float kL5 = 0.150f;      // 髋电机间距
constexpr float kPhi1Stand = -2.3873937f;
constexpr float kPhi4Stand = -0.75419895f;
constexpr float kWheelR = 0.050f;
constexpr float kPi = 3.14159265358979f;
}

// ---- 调试/调参开关 (环境变量, 无需重编译) ----
namespace {
bool GetEnvFlag(const char* name) {
  const char* s = std::getenv(name);
  return s && std::strcmp(s, "1") == 0;
}
float GetEnvFloat(const char* name, float def) {
  const char* s = std::getenv(name);
  return s ? static_cast<float>(std::atof(s)) : def;
}
}

// ---- 全局状态 ----
static const mjModel* g_model = nullptr;
static mjData* g_data = nullptr;
static std::vector<double> g_ctrl_buf;
static bool g_buf_initialized = false;
static std::mutex g_buf_mutex;

namespace {
std::atomic<float> g_cmd_vel{0.0f};
std::atomic<float> g_cmd_turn{0.0f};
std::atomic<float> g_cmd_leg{0.0f};   // 腿长指令增量 (m)
std::atomic<int> g_hold_vel{0};       // 键盘按住状态 (主线程轮询写入)
std::atomic<int> g_hold_turn{0};
std::atomic<bool> g_jump_req{false};  // 跳跃请求 (空格键/headless)
std::atomic<bool> g_jump_armed{false}; // 跳跃使能
}
void SetKeyHold(int vel, int turn) { g_hold_vel.store(vel); g_hold_turn.store(turn); }
int GetKeyHoldVel() { return g_hold_vel.load(); }
int GetKeyHoldTurn() { return g_hold_turn.load(); }

void SetCommandVel(float vel) { g_cmd_vel.store(vel); }
void SetLegRef(float leg) { g_cmd_leg.store(leg); }
float GetLegRef() { return g_cmd_leg.load(); }
void SetCommandTurn(float turn) { g_cmd_turn.store(turn); }
void SetJumpReq(bool on) { g_jump_req.store(on); }
bool GetJumpReq() { return g_jump_req.load(); }
void SetJumpArmed(bool armed) { g_jump_armed.store(armed); }
bool GetJumpArmed() { return g_jump_armed.load(); }

static std::atomic<size_t> g_debug_steps_target{0};
static std::atomic<size_t> g_debug_steps_counter{0};
void setDebugBreakpointAfterSteps(size_t steps) {
  g_debug_steps_counter.store(0);
  g_debug_steps_target.store(steps);
}

void setModelData(const mjModel* m, mjData* d) {
  std::lock_guard<std::mutex> lk(g_buf_mutex);
  g_model = m;
  g_data = d;
}

static void CommitCtrls(const float ctrl[6]) {
  std::lock_guard<std::mutex> lk(g_buf_mutex);
  if (!g_model || !g_data) return;
  if (!g_buf_initialized) {
    g_ctrl_buf.assign(6, 0.0);
    g_buf_initialized = true;
  }
  const char* names[6] = {"motor_alphaL", "motor_betaL", "motor_wheelL",
                          "motor_alphaR", "motor_betaR", "motor_wheelR"};
  for (int i = 0; i < 6; ++i) {
    int aid = mj_name2id(g_model, mjOBJ_ACTUATOR, names[i]);
    if (aid >= 0 && aid < g_model->nu) g_data->ctrl[aid] = ctrl[i];
  }
}

// ---- 五连杆正解: 物理杆角 (phi1, phi4) -> 几何 ----
struct FiveBarOut {
  float xC, zC;        // 轮心 (髋系, x 前 z 上)
  float phi2, phi3;    // 小腿杆角
  float phi5;          // 轮心相对髋中点连线角
  float leg_len;       // 腿长 (髋中点 -> 轮心)
};
static FiveBarOut FiveBarFk(float phi1, float phi4) {
  FiveBarOut o;
  const float xB = kL1 * std::cos(phi1), zB = kL1 * std::sin(phi1);
  const float xD = kL5 + kL1 * std::cos(phi4), zD = kL1 * std::sin(phi4);
  const float BD = std::hypot(xD - xB, zD - zB);
  const float A0 = 2 * kL2 * (xD - xB);
  const float B0 = 2 * kL2 * (zD - zB);
  const float C0 = 2 * kL2 * kL2 + BD * BD - 2 * kL2 * kL2;  // = BD² (L2=L3)
  const float disc = A0 * A0 + B0 * B0 - C0 * C0;
  const float phi2 = 2 * std::atan2(B0 - std::sqrt(std::max(0.0f, disc)), A0 + C0);
  o.xC = xB + kL2 * std::cos(phi2);
  o.zC = zB + kL2 * std::sin(phi2);
  o.phi2 = phi2;
  o.phi3 = std::atan2(o.zC - zD, o.xC - xD);
  o.phi5 = std::atan2(o.zC, o.xC - kL5 / 2);
  o.leg_len = std::hypot(o.xC - kL5 / 2, o.zC);
  return o;
}

// ---- 单侧: 髋角 (q_alpha, q_beta) -> FK (物理角 phi = phi_stand - q, MuJoCo +y 铰链约定) ----
struct SideState {
  float leg_len, theta, phi5;
  float xC, zC;
};
static SideState SideFk(float qa, float qb) {
  const float phi1 = kPhi1Stand - qa;
  const float phi4 = kPhi4Stand - qb;
  const FiveBarOut o = FiveBarFk(phi1, phi4);
  SideState s;
  s.leg_len = o.leg_len;
  s.phi5 = o.phi5;
  s.xC = o.xC;
  s.zC = o.zC;
  s.theta = 0.0f;  // theta 需减去机身俯仰, 由调用方补齐
  return s;
}

// ---- 逆解: 目标腿长 -> (qa_ref, qb_ref) (轮心在髋中点正下方) ----
static void LegIk(float L, float& qa, float& qb) {
  const float xC = kL5 / 2, zC = -L;
  const float AC = std::hypot(xC, zC);
  const float CE = std::hypot(xC - kL5, zC);
  const float psi_AC = std::atan2(zC, xC);
  const float psi_CE = std::atan2(zC, xC - kL5);
  auto clamp1 = [](float v) { return std::max(-1.0f, std::min(1.0f, v)); };
  const float g1 = std::acos(clamp1((kL1 * kL1 + AC * AC - kL2 * kL2) / (2 * kL1 * AC)));
  const float g4 = std::acos(clamp1((kL1 * kL1 + CE * CE - kL2 * kL2) / (2 * kL1 * CE)));
  qa = kPhi1Stand - (psi_AC - g1);
  qb = kPhi4Stand - (psi_CE + g4);
}

// ---- VMC: F_leg(沿腿力) + T_hub(轮毂力矩) -> 髋力矩 (数值雅可比) ----
// vert: 推力取世界系竖直方向 (无水平分量 -> 防跳跃前倾注入)
static void Vmc(float qa, float qb, float F_leg, float T_hub, float& tau_a, float& tau_b, bool vert = false) {
  constexpr float eps = 1e-6f;
  auto c_at = [&](float a, float b) {
    const float p1 = kPhi1Stand - a, p4 = kPhi4Stand - b;
    const FiveBarOut o = FiveBarFk(p1, p4);
    return std::make_pair(o.xC, o.zC);
  };
  const auto [xC, zC] = c_at(qa, qb);
  const SideState s = SideFk(qa, qb);
  const float L = s.leg_len;
  // 腿方向单位向量 (髋中点 -> 轮心)
  const float ux = (xC - kL5 / 2) / L, uz = zC / L;
  // 雅可比 J = dC/d(qa,qb)
  const auto [x1, z1] = c_at(qa + eps, qb);
  const auto [x2, z2] = c_at(qa - eps, qb);
  const auto [x3, z3] = c_at(qa, qb + eps);
  const auto [x4, z4] = c_at(qa, qb - eps);
  const float J00 = (x1 - x2) / (2 * eps), J10 = (z1 - z2) / (2 * eps);
  const float J01 = (x3 - x4) / (2 * eps), J11 = (z3 - z4) / (2 * eps);
  // tau = Jᵀ·(F·u) : [tau_a; tau_b] = [J00 J10; J01 J11]·[Fx; Fz]
  const float Fx = vert ? 0.0f : F_leg * ux;     // 竖直推力: 无水平分量 (防前倾注入)
  const float Fz = vert ? -F_leg : F_leg * uz;
  float ta = J00 * Fx + J10 * Fz;
  float tb = J01 * Fx + J11 * Fz;
  // T_hub 项: 轮毂力矩绕 y, 虚功 T_hub·δθ, θ = phi5 (+π/2 常数无关)
  auto phi5_at = [&](float a, float b) { return FiveBarFk(kPhi1Stand - a, kPhi4Stand - b).phi5; };
  const float gth_a = (phi5_at(qa + eps, qb) - phi5_at(qa - eps, qb)) / (2 * eps);
  const float gth_b = (phi5_at(qa, qb + eps) - phi5_at(qa, qb - eps)) / (2 * eps);
  ta += T_hub * gth_a;
  tb += T_hub * gth_b;
  tau_a = ta;
  tau_b = tb;
}

// ---- 跳跃状态机 ----
enum JumpPhase { JP_DRIVE, JP_SQUAT, JP_JUMP, JP_FLY, JP_LAND };
static JumpPhase jp_phase = JP_DRIVE;
static float jp_t0 = 0.0f;        // 当前阶段起始 boot_t
static float jp_v_to = 0.0f;      // 起跳速度 (落地急停强度 ∝)
static float jp_v_keep = 0.0f;    // 跳跃前速度指令 (落地后恢复)
static bool jp_was_air = false;   // 是否经历过腾空

// 控制器内部状态 (文件级, ResetController 统一清零)
static float boot_t = -1.0f;
static float legd_f = 0, theta_dot_f = 0, rw_f = 0, pw_f = 0, vf_f = 0;
static float leg_len_prev = 0, theta_prev = 0;
static bool inited = false;
static float L_cur = 0.1119f;
static bool L_inited = false;
static float vel_int = 0.0f;
static float th_cmd = 0.0f;
static float yaw_f = 0, yw_f = 0;
static bool sq_deep = false;      // 已到达下蹲理论极限 (待预伸)
static float sq_target = 0.047f;  // SQUAT 当前腿长目标
static bool jp_req_prev = false;  // 跳跃触发边沿检测
static float pos_int = 0.0f;      // 位置保持环积分 (站立防前倾爬升/慢漂)
static float t_stop = -1.0f;      // 急停计时 (松开按键时刻)
static float x_stop = 0.0f;       // 松开按键时的车位置 (位置环参考点)

// ---- 停车状态机 (松开按键后: 急停刹车 -> 姿态回正 -> 静态锁定) ----
// SP_DRIVE: 行驶中; SP_BRAKE: 急停 (速度刹车 + 腿前伸, 髋力矩限幅提高);
// SP_RECOVER: 回正 (俯仰参考斜坡, 缓慢回到直立, 防急停后俯仰摆振);
// SP_SETTLE: 静态锁定 (车完全静止时轮角伺服锁轮 -> 轮子完全静止; 扰动超限则解锁回正)
enum StopPhase { SP_DRIVE, SP_BRAKE, SP_RECOVER, SP_SETTLE };
static StopPhase sp_phase = SP_DRIVE;
static float sp_t0 = 0.0f;        // 停车阶段起始 boot_t
static float rec_pref = 0.0f;     // RECOVER 俯仰参考 (斜坡回正)
static bool db_on = false;        // 俯仰比例项死区状态 (Schmitt 迟滞)
static float w_hold0[2] = {0.0f, 0.0f};  // SETTLE 锁轮时的轮角参考 (轮角伺服)

// 复位控制器全部内部状态 (测试/重启用)
void ResetController(void) {
  jp_phase = JP_DRIVE; jp_t0 = 0.0f; jp_v_to = 0.0f; jp_v_keep = 0.0f; jp_was_air = false;
  boot_t = -1.0f;
  legd_f = theta_dot_f = rw_f = pw_f = vf_f = 0.0f;
  leg_len_prev = theta_prev = 0.0f; inited = false;
  L_cur = 0.1119f; L_inited = false;
  vel_int = 0.0f; th_cmd = 0.0f;
  yaw_f = yw_f = 0.0f;
  sq_deep = false; sq_target = 0.047f;
  jp_req_prev = false;
  pos_int = 0.0f;
  t_stop = -1.0f; x_stop = 0.0f;
  sp_phase = SP_DRIVE; sp_t0 = 0.0f; rec_pref = 0.0f;
  w_hold0[0] = w_hold0[1] = 0.0f;
  g_jump_req.store(false);
}

// 轮子是否触地 (接触 geom 名含 "wheel")
static bool WheelContact(const mjModel* m, const mjData* d) {
  if (!m || !d) return false;
  for (int i = 0; i < d->ncon; ++i) {
    const mjContact& c = d->contact[i];
    if (c.geom1 >= 0 && c.geom2 >= 0) {
      const char* n1 = mj_id2name(m, mjOBJ_GEOM, c.geom1);
      const char* n2 = mj_id2name(m, mjOBJ_GEOM, c.geom2);
      if ((n1 && std::strstr(n1, "wheel")) || (n2 && std::strstr(n2, "wheel"))) return true;
    }
  }
  return false;
}

// ---- 主控制循环 (每步调用) ----
void runCrouchingControl(void) {
  if (!g_model || !g_data) return;
  const float dt = (float)(g_model->opt.timestep > 0 ? g_model->opt.timestep : 0.001);
  static const float kPi = 3.14159265358979f;

  // ---------- 调参 (环境变量, 默认 = 仿真验证值) ----------
  const float kp_leg = GetEnvFloat("WHEELLEG_KP_LEG", 500.0f);
  const float kd_leg = GetEnvFloat("WHEELLEG_KD_LEG", 25.0f);
  const float grav_full = GetEnvFloat("WHEELLEG_GRAV", 4.0f);
  if (boot_t < 0.0f) boot_t = 0.0f;
  boot_t += dt;
  const float grav = grav_full * std::min(1.0f, boot_t / 0.15f);  // 支撑力 0.15s 斜坡 (防慢漂)
  const float kp_r  = GetEnvFloat("WHEELLEG_KP_R", 0.30f);
  const float kd_r  = GetEnvFloat("WHEELLEG_KD_R", 0.12f);
  const float kp_t  = GetEnvFloat("WHEELLEG_KP_T", 4.0f);   // 腿角 PD: th -> th_cmd (摆腿驱动)
  const float kd_t  = GetEnvFloat("WHEELLEG_KD_T", 0.5f);
  const float k_tha = GetEnvFloat("WHEELLEG_K_THA", -0.3f); // 速度误差 -> 腿角参考 (加速腿后摆/刹车腿前摆)
  const float ki_a  = GetEnvFloat("WHEELLEG_KI_A", 0.08f);  // 速度积分 -> 腿角参考 (消除静差)
  const float kp_p2 = GetEnvFloat("WHEELLEG_KP_P2", 0.5f);  // 平台前倾 -> 腿角修正 (平台恒平)
  const float kp_q  = GetEnvFloat("WHEELLEG_KP_Q", 0.8f);   // 髋位置回中 (直膝支, 防弯膝漂移)
  const float kd_q  = GetEnvFloat("WHEELLEG_KD_Q", 0.08f);
  const float kd_w  = GetEnvFloat("WHEELLEG_KD_W", 0.06f);  // 轮速阻尼 (抑制轮子粘滑)
  const float kd_wy = GetEnvFloat("WHEELLEG_KD_WY", 0.02f); // 松开转向: yaw 角速度阻尼 (停转不回正)
  const float kp_p  = GetEnvFloat("WHEELLEG_KP_P", 6.0f);   // 平台恒平: 轮力矩 pitch 负反馈
  const float kd_p  = GetEnvFloat("WHEELLEG_KD_P", 0.5f);
  const float tw_lim = GetEnvFloat("WHEELLEG_TW_LIM", 0.3f);   // 摩擦容量内, 防打滑
  const float th_lim = GetEnvFloat("WHEELLEG_TH_LIM", 0.8f);
  const float f_lim  = GetEnvFloat("WHEELLEG_F_LIM", 40.0f);
  const float l_stand = GetEnvFloat("WHEELLEG_L_STAND", 0.300f);
  // 跳跃参数
  const float l_squat = GetEnvFloat("WHEELLEG_L_SQUAT", 0.180f);    // 新尺寸下蹲
  const float l_squat_min = GetEnvFloat("WHEELLEG_L_SQUAT_MIN", 0.160f);
  const float l_jump  = GetEnvFloat("WHEELLEG_L_JUMP", 0.400f);     // 蹬地伸展腿长
  const float kp_q_jump = GetEnvFloat("WHEELLEG_KP_Q_JUMP", 4.0f);  // 蹬地关节位置环 (解折叠)
  const float t_squat = GetEnvFloat("WHEELLEG_T_SQUAT", 0.5f);      // 下蹲保持时长
  const float t_push  = GetEnvFloat("WHEELLEG_T_PUSH", 0.6f);       // 蹬地时长
  const float t_land  = GetEnvFloat("WHEELLEG_T_LAND", 0.5f);       // 落地恢复时长
  // 停车参数
  const float k_brake    = GetEnvFloat("WHEELLEG_K_BRAKE", 0.8f);    // 急停: 速度 -> 后仰参考
  const float tw_brake   = GetEnvFloat("WHEELLEG_TW_BRAKE", 0.5f);   // 急停轮力矩上限
  const float rec_rate   = GetEnvFloat("WHEELLEG_REC_RATE", 0.30f);  // 回正俯仰参考速率 (rad/s)
  const float rec_tw     = GetEnvFloat("WHEELLEG_REC_TW", 0.20f);    // 回正轮力矩上限 (轻轮防打滑)
  const float kd_p_stop  = GetEnvFloat("WHEELLEG_KD_P_STOP", 1.5f);  // 停车阶段俯仰率阻尼 (抑制摆振)
  const float st_pos     = GetEnvFloat("WHEELLEG_ST_POS", 0.012f);   // 近静止: 位置误差带 (m)
  const float st_pitch   = GetEnvFloat("WHEELLEG_ST_PITCH", 0.006f); // 近静止: 俯仰带 (rad)
  const float v_brake_lo = GetEnvFloat("WHEELLEG_V_BRAKE_LO", 0.25f); // 低于此速度结束急停

  // ---------- 状态 (缓存层, 与跳跃前一致) ----------
  UpdateInlineGetData(g_model, g_data);
  const float pitch = body_pitch;
  const float roll  = body_roll;
  const float qa[2] = {alpha_L, alpha_R};
  const float qb[2] = {beta_L,  beta_R};
  const float wheel_speed1 = ::wheel_speed1;
  const float wheel_speed2 = ::wheel_speed2;

  // ---------- 五连杆正解 (每侧) ----------
  SideState s[2] = {};
  float leg_len = 0, theta = 0;
  for (int i = 0; i < 2; ++i) {
    s[i] = SideFk(qa[i], qb[i]);
    s[i].theta = s[i].phi5 + kPi / 2 - pitch;   // 腿角 (前倾+)
    leg_len += s[i].leg_len / 2;
    theta += s[i].theta / 2;
  }

  // ---------- 滤波 (低通) ----------
  if (!inited) { leg_len_prev = leg_len; theta_prev = theta; inited = true; }
  const float legd_raw = (leg_len - leg_len_prev) / dt;
  const float thd_raw = (theta - theta_prev) / dt;
  leg_len_prev = leg_len; theta_prev = theta;
  const float tau_f = 0.01f;
  legd_f += (legd_raw - legd_f) * (dt / tau_f);
  theta_dot_f += (thd_raw - theta_dot_f) * (dt / tau_f);
  const float rw_raw = body_groll;
  rw_f += (rw_raw - rw_f) * (dt / 0.01f);
  const float pw_raw = body_gpitch;
  pw_f += (pw_raw - pw_f) * (dt / 0.01f);
  const float vx = g_data ? (float)g_data->qvel[0] : 0.0f;
  vf_f += (vx - vf_f) * (dt / 0.02f);

  // ---------- 速度指令 (键盘按住: ±1.0 m/s) ----------
  float vel_cmd_raw = g_cmd_vel.load() + g_hold_vel.load() * 1.0f;
  // 落地恢复后 0.5s 斜坡 (先稳定再加速, 防恢复段倒车)
  if (jp_phase == JP_DRIVE && boot_t - jp_t0 < 0.5f) vel_cmd_raw *= (boot_t - jp_t0) / 0.5f;
  float vel_cmd = vel_cmd_raw;                        // 有效速度指令 (跳跃腾空/落地阶段清零)

  // ---------- 跳跃状态机 ----------
  const float wz = (float)g_data->qpos[2] + (s[0].zC + s[1].zC) * 0.5f;  // 轮心世界 z
  const bool contact = WheelContact(g_model, g_data);
  const bool airborne = !contact && wz > 0.03f;   // 无接触且轮心离地 => 腾空
  const float jp_el = boot_t - jp_t0;             // 当前阶段时长
  const bool req_now = g_jump_req.load();
  bool jump_active = jp_phase != JP_DRIVE;
  if (jp_phase == JP_DRIVE && req_now && !jp_req_prev) {
    jp_phase = JP_SQUAT; jp_t0 = boot_t;
    jp_v_keep = vel_cmd_raw;            // 前进跳: 保留速度指令; 原地跳: 0
    jp_v_to = 0.0f; jp_was_air = false;
  } else if (jp_phase == JP_SQUAT) {
    // 下蹲到位且前倾回正即蹬地 (防起跳前倾注入); t_squat 超时兜底
    if (jp_el > t_squat || (leg_len < l_squat + 0.012f && std::fabs(pitch) < 0.04f)) {
      jp_phase = JP_JUMP; jp_t0 = boot_t;
    }
  } else if (jp_phase == JP_JUMP) {
    jp_v_to = vf_f;                     // 记录起跳速度 (每步刷新, 起跳时刻为准)
    // 弹起即转 FLY (空中不收腿伸长 -> 角动量前倾); t_push 为未弹起超时
    if ((airborne && jp_el > 0.05f) || jp_el > t_push) { jp_phase = JP_FLY; jp_t0 = boot_t; }
  } else if (jp_phase == JP_FLY) {
    jp_v_to = vf_f;                        // 记录落地水平速度 (急停强度 ∝)
    if (airborne) jp_was_air = true;
    if (jp_el > 1.0f && !jp_was_air) { jp_phase = JP_DRIVE; jp_t0 = boot_t; }  // 没跳起来: 放弃
    else if (jp_was_air && !airborne && jp_el > 0.05f) { jp_phase = JP_LAND; jp_t0 = boot_t; }
  } else if (jp_phase == JP_LAND) {
    if ((jp_el > t_land && std::fabs(L_cur - l_stand) < 0.008f && std::fabs(vf_f) < 0.15f) ||
        jp_el > 1.5f) {
      jp_phase = JP_DRIVE; jp_t0 = boot_t;   // 落地站稳 -> 恢复 (1.5s 超时兜底)
    }
  }
  jump_active = jp_phase != JP_DRIVE;
  jp_req_prev = req_now;

  // ---------- 停车状态机 (松开按键后; 跳跃阶段让位) ----------
  // SP_DRIVE -> (松开且 vf 大) SP_BRAKE / (已静止) SP_SETTLE
  // SP_BRAKE -> vf 降到阈值下 (或后仰过深) SP_RECOVER
  // SP_RECOVER -> 俯仰回正且速度归零 SP_SETTLE (2s 超时兜底)
  // SP_SETTLE -> 扰动超限 SP_RECOVER; 按键 SP_DRIVE
  if (!jump_active && boot_t - jp_t0 > 1.0f) {   // 跳跃恢复 1s 内不激活停车状态机 (防干扰落地稳定)
    const float sp_el = boot_t - sp_t0;
    if (vel_cmd >= 0.01f) {
      sp_phase = SP_DRIVE; sp_t0 = boot_t;
    } else if (sp_phase == SP_DRIVE) {
      sp_phase = (vf_f > v_brake_lo) ? SP_BRAKE : SP_SETTLE;
      sp_t0 = boot_t; rec_pref = pitch;
    } else if (sp_phase == SP_BRAKE) {
      if (vf_f <= v_brake_lo || pitch < -0.38f) {   // 速度归零或后仰过深 -> 回正
        sp_phase = SP_RECOVER; sp_t0 = boot_t;
        rec_pref = std::max(pitch, -0.30f);         // 参考从当前俯仰开始斜坡回零
        x_stop = (float)g_data->qpos[0]; pos_int = 0.0f;   // 锚定实际停车位置
      }
    } else if (sp_phase == SP_RECOVER) {
      if ((std::fabs(pitch) < 0.012f && std::fabs(pw_f) < 0.12f && std::fabs(vf_f) < 0.06f) ||
          (sp_el > 2.0f && std::fabs(vf_f) < 0.06f)) {
        sp_phase = SP_SETTLE; sp_t0 = boot_t;
        x_stop = (float)g_data->qpos[0]; pos_int = 0.0f;
      }
    } else {  // SP_SETTLE
      const float pos_x = (float)g_data->qpos[0];
      const float pos_rel = pos_x - x_stop;
      if (std::fabs(pitch) > 2.0f * st_pitch || std::fabs(pw_f) > 0.25f ||
          std::fabs(vf_f) > 0.10f || std::fabs(pos_rel) > 3.0f * st_pos) {
        sp_phase = SP_RECOVER; sp_t0 = boot_t;
        rec_pref = pitch;
      }
    }
  }

  // ---------- 高度指令: 目标腿长 (跳跃时由状态机接管) ----------
  float L_target = l_stand;
  if (jump_active) {
    if (jp_phase == JP_SQUAT)      L_target = l_squat;   // 下蹲到理论极限
    else if (jp_phase == JP_JUMP)  L_target = l_jump;    // 蹬地伸直
    else if (jp_phase == JP_FLY)   L_target = 0.118f;    // 腾空: 收腿到 0.118 (落地平衡强)
    else                           L_target = l_stand;   // 落地: 恢复站姿腿长
  } else {
    const float leg_cmd = g_cmd_leg.load() + GetEnvFloat("WHEELLEG_LEG_REF", 0.0f);
    L_target = std::clamp(l_stand + leg_cmd, l_squat_min, 0.135f);  // 身高键范围: 下限到理论极限
  }
  if (!L_inited) { L_cur = l_stand; L_inited = true; }
  const float kLegRate = GetEnvFloat("WHEELLEG_LEG_RATE", 0.025f);
  const float leg_rate = jump_active ? (jp_phase == JP_JUMP ? 0.35f : 0.20f) : kLegRate;
  if (L_cur < L_target) L_cur = std::min(L_target, L_cur + leg_rate * dt);
  else                  L_cur = std::max(L_target, L_cur - leg_rate * dt);

  // ---------- 腿长控制 + 横滚补偿 ----------
  const float dL = std::clamp(kp_r * roll + kd_r * rw_f, -0.035f, 0.035f);
  const float hgt = leg_len * std::cos(theta);
  float F[2] = {0.0f, 0.0f};
  if (jp_phase == JP_SQUAT) {
    // 下蹲: 仅重力支撑 (下蹲由强关节位置环拉向 ik(l_squat), 避免 F 力矩把腿推反向折叠)
    F[0] = F[1] = grav;
  } else if (jp_phase == JP_JUMP) {
    // 位置环蹬地: 深蹲极限位奇异推不动, 腿脱离奇异区后猛推; 腿伸直后自然归零 (不过冲)
    F[0] = std::clamp(800.0f * (l_jump - hgt) + dL, -80.0f, 80.0f);
    F[1] = std::clamp(800.0f * (l_jump - hgt) - dL, -80.0f, 80.0f);
  } else if (jp_phase == JP_FLY) {
    F[0] = F[1] = grav;   // 腾空: 不收腿 (kp_q 保持角度, 防角动量交换后仰)
  } else if (jp_phase == JP_LAND) {
    // 落地: 压缩阻尼吸收冲击 (防回弹) + 渐进取力缓撑
    const float f_land_lim = GetEnvFloat("WHEELLEG_F_LIM_LAND", 80.0f);
    const float legd_comp = std::max(-legd_f, 0.0f) * std::cos(theta);  // 压缩速度 (legd<0=腿缩)
    if (jp_el < 0.08f) {
      F[0] = F[1] = grav + 50.0f * legd_comp;   // 落地瞬间: 阻尼吸收冲击, 不撑
    } else {
      // 渐进取力撑起 (追站姿腿长; 缓撑防回弹); 双向阻尼
      const float ramp = std::min(1.0f, (jp_el - 0.08f) / 0.25f);
      F[0] = std::clamp(250.0f * ramp * (l_stand + dL - hgt) - 40.0f * legd_f * std::cos(theta) + grav, -f_land_lim, f_land_lim);
      F[1] = std::clamp(250.0f * ramp * (l_stand - dL - hgt) - 40.0f * legd_f * std::cos(theta) + grav, -f_land_lim, f_land_lim);
    }
  } else {
    F[0] = std::clamp(kp_leg * (L_cur + dL - hgt) - kd_leg * legd_f * std::cos(theta) + grav, -f_lim, f_lim);
    F[1] = std::clamp(kp_leg * (L_cur - dL - hgt) - kd_leg * legd_f * std::cos(theta) + grav, -f_lim, f_lim);
  }
  // 髋回中目标 = ik(L_cur) (跟随腿长); FLY 锁当前腿长 (空中不伸腿, 防倾斜注入)
  float qa_ref = 0.0f, qb_ref = 0.0f;
  if (jp_phase == JP_FLY) LegIk(std::max(leg_len, l_squat), qa_ref, qb_ref);
  else                    LegIk(L_cur, qa_ref, qb_ref);

  // ---------- 速度摆: 腿角参考 th_cmd (加速腿后摆 / 刹车腿前摆) ----------
  if (jump_active && jp_phase != JP_SQUAT && jp_phase != JP_JUMP) vel_cmd = 0.0f;  // 腾空/落地不驱动
  const float ve = vf_f - vel_cmd;
  const float vel_dead = GetEnvFloat("WHEELLEG_VEL_DEAD", 0.08f);   // 速度死区: 站立/停车不摆腿
  if (vel_cmd < 0.01f) {
    vel_int = 0.0f;                                    // 松开按键即清积分: 停在原地, 不倒车
  } else if (std::fabs(ve) > vel_dead) {
    vel_int = std::clamp(vel_int + ve * dt, -0.3f, 0.3f);
  }
  // 前进速度上限: 超限强摆前 (刹车), 防持续加速
  const float v_max = GetEnvFloat("WHEELLEG_V_MAX", 1.0f);
  const float k_vmax = GetEnvFloat("WHEELLEG_K_VMAX", 1.0f);
  const float vmax_term = vf_f > v_max ? k_vmax * (vf_f - v_max) : 0.0f;
  const float tc_ramp = GetEnvFloat("WHEELLEG_TC_RAMP", 0.5f);      // th_cmd 起步斜坡 (防腿长耦合)
  // 松开按键 (vel_cmd≈0) -> 急停: 刹车摆增益加倍
  const float k_tha_eff = vel_cmd < 0.01f ? -0.6f : k_tha;
  const float th_cmd_base = (k_tha_eff * (vel_cmd - vf_f) + ki_a * vel_int + kp_p2 * pitch)
                          * std::min(1.0f, boot_t / tc_ramp);
  th_cmd = std::clamp(th_cmd_base + vmax_term, -0.20f, 0.20f);      // 上限项在 clip 外, 优先生效
  if (boot_t < 1.0f) th_cmd = 0.0f;   // 起步保护: 前 1s 腿纯回正 (防起步扰动放大)
  // 停车阶段覆盖: BRAKE 腿前伸急停 (质心前移, 制动力后仰力矩抵消惯性前倾, ∝ 速度, 参照落地急停)
  // RECOVER/SETTLE 腿回中 (0 = 速度摆参考, 站姿); RECOVER 初期渐减 (防腿突回扰动)
  if (!jump_active && vel_cmd < 0.01f) {
    if (sp_phase == SP_BRAKE) {
      const float brake_k = std::min(1.0f, std::max(vf_f, 0.0f) / 1.0f);
      const float leg_lean = (0.15f + 0.45f * brake_k) * std::min(1.0f, (boot_t - sp_t0) / 0.12f);
      th_cmd = leg_lean;
    } else if (sp_phase == SP_RECOVER) {
      th_cmd = 0.0f;   // 回正: 腿回中 (髋力矩限幅正常, 回中自然缓慢)
    } else if (sp_phase == SP_SETTLE) {
      th_cmd = 0.0f;
    }
  }
  const bool skip_th = jp_phase == JP_SQUAT || jp_phase == JP_JUMP;  // 跳跃阶段: 速度摆让位
  if (skip_th) th_cmd = 0.0f;

  // ---------- 腿角控制 (轮毂力矩: th -> th_cmd) ----------
  float T_hub = 0.0f;
  if (jp_phase == JP_SQUAT) {
    // 深蹲: 回正腿角 (θ→0) — T_hub 正(前摆)压前倾, 勿加负姿态项 (会变后摆加剧前倾)
    T_hub = kp_t * (0.0f - theta) - kd_t * theta_dot_f;
  } else if (jp_phase == JP_JUMP) {
    // 蹬地: 姿态压 (轮子着地: 负号); 弹起后空中: 反号 (角动量交换)
    if (airborne) T_hub = 12.0f * pitch + 3.0f * pw_f;
    else          T_hub = -12.0f * pitch - 3.0f * pw_f;
  } else if (jp_phase == JP_FLY) {
    // 空中: 姿态温和 + 落地前腿预前伸 (前伸就位, 落地瞬间即可压前摆)
    T_hub = kp_t * (0.35f - theta) - kd_t * theta_dot_f;
  } else if (jp_phase == JP_LAND) {
    // 腿前伸急停: 前 0.6s 腿前伸 (质心前移, 制动力产生后仰力矩抵消惯性前倾; ∝ 落地速度)
    const float brake_k = std::min(1.0f, std::fabs(jp_v_to) / 1.0f);
    // 腿前伸急停: 质心前移 -> 制动力后仰力矩抵消惯性前倾 (∝ 落地速度)
    // 末期渐减回中 (防回 DRIVE 时腿角突变扰动)
    const float lean_ramp = std::min(1.0f, (0.7f - jp_el) / 0.3f);
    const float lean = (0.15f + 0.45f * brake_k) * std::max(0.0f, lean_ramp);
    T_hub = kp_t * (lean - theta) - kd_t * theta_dot_f;
  } else {
    T_hub = kp_t * (th_cmd - theta) - kd_t * theta_dot_f;
  }

  // ---------- VMC -> 髋力矩 + 髋位置回中 (直膝支) ----------
  float tau_a[2] = {0.0f, 0.0f}, tau_b[2] = {0.0f, 0.0f};
  const float kp_q_eff = (jp_phase == JP_JUMP || jp_phase == JP_SQUAT) ? kp_q_jump
                       : (jp_phase == JP_FLY)  ? 0.5f : (jp_phase == JP_LAND) ? 2.0f : kp_q;
  const float kd_q_eff = jump_active ? 1.0f : kd_q;
  const bool vert_push = (jp_phase == JP_SQUAT || jp_phase == JP_JUMP);
  for (int i = 0; i < 2; ++i) {
    Vmc(qa[i], qb[i], F[i], T_hub, tau_a[i], tau_b[i], vert_push);
    const float qda = i == 0 ? alpha_Lv : alpha_Rv;
    const float qdb = i == 0 ? beta_Lv  : beta_Rv;
    tau_a[i] += kp_q_eff * (qa_ref - qa[i]) - kd_q_eff * qda;
    tau_b[i] += kp_q_eff * (qb_ref - qb[i]) - kd_q_eff * qdb;
  }

  // ---------- 平台恒平 (轮力矩 pitch 负反馈; 速度由腿摆驱动, 无需倾斜参考) ----------
  float T_w = 0.0f;
  if (jp_phase == JP_FLY) {
    T_w = 0.0f;                                        // 腾空无力矩
  } else if (jp_phase == JP_SQUAT) {
    T_w = std::clamp(kp_p * pitch + kd_p * pw_f, -tw_lim, tw_lim);   // 下蹲: 姿态正常
  } else if (jp_phase == JP_LAND) {
    // 落地: 前 0.15s 轮子随前冲压前摆; 之后姿态+刹车 (前倾时平衡车维持, vf 大时刹)
    if (jp_el < 0.15f) T_w = 0.35f;
    else               T_w = std::clamp(kp_p * pitch + kd_p * pw_f, -0.35f, 0.35f);
  } else {
    // 行驶 (vel_cmd>0): 纯姿态 (速度摆驱动)
    // 站立/停车 (vel_cmd≈0): 停车状态机 SP_BRAKE(急停) / SP_SETTLE(静态保持+死区平衡)
    if (vel_cmd < 0.01f) {
      // 位置锚点: BRAKE 期间保持进入时锚定 (不拉回); RECOVER/SETTLE 锚定在停车位置
      pos_int = 0.0f;
      if (sp_phase == SP_BRAKE) {
        // 急停: 速度刹车 (pref=-k_brake·vf 后仰参考 -> 轮子前推 -> 地面摩擦制动; 腿前伸配合)
        // 后仰参考限幅 -0.22 + 强俯仰率阻尼: 防深后仰/摆振 (急停后回正困难)
        const float pref = std::clamp(-k_brake * vf_f, -0.22f, 0.22f);
        T_w = std::clamp(kp_p * (pitch - pref) + kd_p_stop * pw_f, -tw_brake, tw_brake);
      } else if (sp_phase == SP_RECOVER) {
        // 回正: 俯仰参考斜坡回零 (rec_pref -> 0) + 位置环速度阻尼 (防倒车/回正时后滑)
        // 轮力矩受 rec_tw 限幅, 缓速回正, 防急停后俯仰摆振
        const float pos_x = (float)g_data->qpos[0];
        const float pos_rel = pos_x - x_stop;
        pos_int = std::clamp(pos_int + pos_rel * dt, -0.5f, 0.5f);
        const float v_t = -0.5f * pos_rel - 0.15f * pos_int;
        const float dref = std::clamp(0.0f - rec_pref, -rec_rate * dt, rec_rate * dt);
        rec_pref += dref;
        const float pref = rec_pref + std::clamp(0.8f * (v_t - vf_f), -0.12f, 0.12f);
        T_w = std::clamp(kp_p * (pitch - pref) + kd_p_stop * pw_f, -rec_tw, rec_tw);
      } else {
        // SETTLE: 死区平衡 + 速度阻尼, 无位置拉回 (停哪算哪, 防回正后位置环摆振)
        // 俯仰比例项死区 (Schmitt 迟滞): 平衡偏置力矩落入死区 -> 该项归零, 轮子不被持续驱动;
        // 车近静止时锚点跟随; 漂移/扰动超限由状态机转 RECOVER 回正后再静置
        const float pos_x = (float)g_data->qpos[0];
        const bool calm = std::fabs(vf_f) < 0.04f && std::fabs(pitch) < 0.02f &&
                          std::fabs(pw_f) < 0.2f;
        if (calm) { x_stop = pos_x; pos_int = 0.0f; }   // 近静止: 锚点跟随
        const float pref = std::clamp(-0.8f * vf_f, -0.06f, 0.06f);  // 纯速度阻尼参考
        // 俯仰角死区 (Schmitt, 阈 0.05/0.02 rad): 残余前倾 <0.05 不驱动轮子 -> 不爬行;
        // 超过才修正 (防倾倒); 速度阻尼 + 俯仰率阻尼压残余摆动
        const float t_bias = kp_p * pitch;
        if (db_on && std::fabs(pitch) < 0.03f) db_on = false;
        else if (!db_on && std::fabs(pitch) > 0.08f) db_on = true;
        const float t_bal = (db_on ? t_bias : 0.0f) - kp_p * pref + kd_p_stop * pw_f;
        T_w = std::clamp(t_bal, -rec_tw, rec_tw);
      }
    } else {
      pos_int = 0.0f;
      T_w = std::clamp(kp_p * pitch + kd_p * pw_f, -tw_lim, tw_lim);
    }
  }
  // yaw: 转向指令 -> 速度环; 松开 -> 只停转不回正 (保持当前朝向)
  const double* q = g_data->qpos + 3;
  const float yaw = (float)std::atan2(2.0*(q[3]*q[0] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]));
  yaw_f += (yaw - yaw_f) * (dt / 0.05f);
  yw_f += (body_gyaw - yw_f) * (dt / 0.02f);
  const float turn_cmd = g_cmd_turn.load() + g_hold_turn.load() * 1.5f; // 键盘按住: ±1.5 rad/s
  float T_y = 0.0f;
  if (std::fabs(turn_cmd) > 0.01f) {
    const float kp_wz = GetEnvFloat("WHEELLEG_KP_WZ", 0.05f);
    T_y = kp_wz * (turn_cmd - yw_f);          // 按住: 跟踪目标偏航角速度
  } else {
    T_y = -kd_wy * yw_f;                      // 松开: 只停转, 不回正 (保持当前朝向)
  }
  T_y = std::clamp(T_y, -0.03f, 0.03f);
  // 轮速阻尼 (每轮单独; 仅低速生效抑制粘滑极限环, 高速衰减不阻碍行驶)
  // SETTLE 锁轮: 车完全静止时轮角伺服 (kp_wh·Δθ + kd_wh·ω) 把轮子钉在当前角度 -> 轮子完全静止
  const float kd_w_eff = kd_w;
  auto wheel_damp = [kd_w_eff](float ws) {
    const float w = std::fabs(ws);
    if (w < 0.5f) return -kd_w_eff * ws;
    if (w < 2.5f) return -kd_w_eff * ws * (2.5f - w) / 2.0f;
    return 0.0f;
  };
  const float T_damp1 = wheel_damp(wheel_speed1);
  const float T_damp2 = wheel_damp(wheel_speed2);

  // ---------- 输出 ----------
  // 髋力矩限幅: 跳跃全阶段 与 急停(腿前伸需要大力矩摆腿) 提高
  const float th_lim_eff = (jump_active || sp_phase == SP_BRAKE) ? 3.0f : th_lim;
  float ctrl[6] = {
      std::clamp(tau_a[0], -th_lim_eff, th_lim_eff), std::clamp(tau_b[0], -th_lim_eff, th_lim_eff), T_w + T_y + T_damp1,
      std::clamp(tau_a[1], -th_lim_eff, th_lim_eff), std::clamp(tau_b[1], -th_lim_eff, th_lim_eff), T_w - T_y + T_damp2};
  // 防御: 过滤非有限值 (垃圾 -> 0); 跳跃阶段允许大力矩
  const float ctrl_lim = jump_active ? 10.0f : 1.0f;
  for (float& c : ctrl) if (!std::isfinite(c) || std::fabs(c) > ctrl_lim) c = 0.0f;
  CommitCtrls(ctrl);

  // ---------- 调试打印 (起步每 0.01s, 之后每 0.1s) ----------
  if (GetEnvFlag("WHEELLEG_DEBUG")) {
    static size_t dbg_cnt = 0;
    const bool early = boot_t < 0.3f && (dbg_cnt % 20 == 0);
    if (++dbg_cnt % 200 == 0 || early) {
      const char* ph = jp_phase == JP_DRIVE ? "DRIVE" : jp_phase == JP_SQUAT ? "SQUAT"
                       : jp_phase == JP_JUMP ? "JUMP" : jp_phase == JP_FLY ? "FLY" : "LAND";
      const char* sp = sp_phase == SP_DRIVE ? "drv" : sp_phase == SP_BRAKE ? "brk"
                       : sp_phase == SP_RECOVER ? "rec" : "set";
      printf("dbg t=%.3f [%s/%s] pitch=%.4f th=%.3f L=%.4f z=%.4f vx=%.3f qa=%.3f qaR=%.3f "
             "T_w=%.2f T_hub=%.2f F_L=%.1f air=%d wz=%.3f\n",
             boot_t, ph, sp, pitch, theta, leg_len, (float)g_data->qpos[2],
             vf_f, qa[0], qa_ref, T_w, T_hub, F[0], airborne ? 1 : 0, wz);
    }
  }
}

}  // namespace wheelleg
