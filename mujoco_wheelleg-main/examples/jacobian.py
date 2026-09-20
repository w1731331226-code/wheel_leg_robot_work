import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==========================================
# 1. 核心数学模型 (公式法)
# ==========================================

def get_lod_and_derivative(phi, L_OA, L_AD):
    """
    计算 L_OD 长度及其对 phi 的导数
    phi = (theta_A - theta_B) / 2
    """
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    # 根号内的项: h^2 = L_AD^2 - (L_OA * sin(phi))^2
    term_sqrt_sq = L_AD**2 - (L_OA * sin_phi)**2
    if term_sqrt_sq < 0:
        return None, None
        
    sqrt_term = np.sqrt(term_sqrt_sq)
    
    # L_OD 公式
    L_OD = L_OA * cos_phi + sqrt_term
    
    # L_OD 对 phi 的导数 (dL_OD / dphi)
    # d/dphi (A cos) = -A sin
    # d/dphi (sqrt(D^2 - A^2 sin^2)) = (1/2sqrt) * (-2 A^2 sin cos)
    dLOD_dphi = -L_OA * sin_phi - (L_OA**2 * sin_phi * cos_phi) / sqrt_term
    
    return L_OD, dLOD_dphi

def calculate_analytical_kinematics(theta_A_deg, theta_B_deg, params):
    """
    使用解析公式计算 C 点位置和雅可比矩阵
    """
    # 参数提取
    L_OA = params['L_OA']
    L_OB = params['L_OB']
    L_AD = params['L_AD']
    L_BC = params['L_BC']
    k = L_BC / L_AD
    
    # 角度转弧度
    thA = np.radians(theta_A_deg)
    thB = np.radians(theta_B_deg)
    
    # 中间变量
    phi = (thA - thB) / 2.0
    thD = (thA + thB) / 2.0
    
    # 1. 计算 L_OD 及其导数
    L_OD, dLOD_dphi = get_lod_and_derivative(phi, L_OA, L_AD)
    
    if L_OD is None: 
        return None, None
    
    # 2. 计算 C 点笛卡尔坐标 (Cx, Cy)
    # C = (L_OB - k*L_OA) * u_B + k * L_OD * u_D
    # u_B = [cos(thB), sin(thB)]
    # u_D = [cos(thD), sin(thD)]
    
    L_base = L_OB - k * L_OA
    
    Cx = L_base * np.cos(thB) + k * L_OD * np.cos(thD)
    Cy = L_base * np.sin(thB) + k * L_OD * np.sin(thD)
    
    # 计算极坐标输出
    L_OC = np.sqrt(Cx**2 + Cy**2)
    theta_C = np.arctan2(Cy, Cx)
    
    # 3. 计算雅可比矩阵
    # 我们需要 J_xy = [dC/dthA, dC/dthB]
    
    # 链式法则准备:
    # phi 对 thA, thB 的偏导
    dphi_dthA = 0.5
    dphi_dthB = -0.5
    # thD 对 thA, thB 的偏导
    dthD_dthA = 0.5
    dthD_dthB = 0.5
    
    # L_OD 对 thA, thB 的偏导
    dLOD_dthA = dLOD_dphi * dphi_dthA
    dLOD_dthB = dLOD_dphi * dphi_dthB
    
    # 向量项的导数
    # Term 1: V1 = L_base * [cos(thB), sin(thB)]
    # dV1/dthA = 0
    # dV1/dthB = L_base * [-sin(thB), cos(thB)]
    
    dV1_x_dthB = L_base * (-np.sin(thB))
    dV1_y_dthB = L_base * (np.cos(thB))
    
    # Term 2: V2 = k * L_OD * [cos(thD), sin(thD)]
    # 这是一个乘积求导: k * (dL_OD * u_D + L_OD * du_D)
    
    # u_D 对 thA (通过 thD)
    duD_x_dthD = -np.sin(thD)
    duD_y_dthD = np.cos(thD)
    
    # -- 对 thA 的偏导 --
    # dCx_dthA = k * [ (dLOD/dthA) * cos(thD) + L_OD * (-sin(thD)) * dthD/dthA ]
    dCx_dthA = k * (dLOD_dthA * np.cos(thD) + L_OD * (-np.sin(thD)) * dthD_dthA)
    dCy_dthA = k * (dLOD_dthA * np.sin(thD) + L_OD * (np.cos(thD)) * dthD_dthA)
    
    # -- 对 thB 的偏导 --
    # dCx_dthB = dV1/dthB + k * [ (dLOD/dthB) * cos(thD) + L_OD * (-sin(thD)) * dthD/dthB ]
    dCx_dthB = dV1_x_dthB + k * (dLOD_dthB * np.cos(thD) + L_OD * (-np.sin(thD)) * dthD_dthB)
    dCy_dthB = dV1_y_dthB + k * (dLOD_dthB * np.sin(thD) + L_OD * (np.cos(thD)) * dthD_dthB)
    
    J_xy = np.array([
        [dCx_dthA, dCx_dthB],
        [dCy_dthA, dCy_dthB]
    ])
    
    # 4. 转换到极坐标雅可比 J_polar
    # [dL/dt]   [ cos   sin ] [ dx/dt ]
    # [dth/dt]  [-sin/L cos/L] [ dy/dt ]
    
    c_out = np.cos(theta_C)
    s_out = np.sin(theta_C)
    
    J_transform = np.array([
        [c_out, s_out],
        [-s_out/L_OC, c_out/L_OC]
    ])
    
    J_final = J_transform @ J_xy
    
    return (Cx, Cy), J_final

# ==========================================
# 2. 几何解算 
# ==========================================
def calculate_geometry_points(theta_A_deg, theta_B_deg, params):
    # (此函数保留用于绘制连杆，逻辑同上一版，简略重写)
    L_OA, L_OB, L_AD, L_BC = params['L_OA'], params['L_OB'], params['L_AD'], params['L_BC']
    rad_A, rad_B = np.radians(theta_A_deg), np.radians(theta_B_deg)
    
    O = np.array([0., 0.])
    A = np.array([L_OA*np.cos(rad_A), L_OA*np.sin(rad_A)])
    B = np.array([L_OB*np.cos(rad_B), L_OB*np.sin(rad_B)])
    E = np.array([L_OA*np.cos(rad_B), L_OA*np.sin(rad_B)])
    
    rad_D = (rad_A + rad_B)/2.0
    half_angle = (rad_A - rad_B)/2.0
    half_chord_AE = L_OA * np.abs(np.sin(half_angle))
    
    if L_AD < half_chord_AE: return None
    
    dist_OM = L_OA * np.cos(half_angle)
    dist_MD = np.sqrt(L_AD**2 - half_chord_AE**2)
    dist_OD = dist_OM + dist_MD
    D = np.array([dist_OD*np.cos(rad_D), dist_OD*np.sin(rad_D)])
    
    vec_BC = (D - E) * (L_BC / L_AD)
    C = B + vec_BC
    return {'O':O, 'A':A, 'B':B, 'C':C, 'D':D, 'E':E}

# ==========================================
# 3. 主程序：绘图与交互
# ==========================================
params = {
    'L_OA': 0.12,  # 也是 OE
    'L_OB': 0.2,
    'L_AD': 0.12,   # 也是 DE
    'L_BC': 0.2
}
init_theta_A, init_theta_B = 120, 30

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.3)

# 绘图对象
line_OA, = ax.plot([], [], 'o-', lw=3, color='blue', alpha=0.3, label='Linkages')
line_OB, = ax.plot([], [], 'o-', lw=3, color='green', alpha=0.3)
line_other, = ax.plot([], [], 'o-', lw=2, color='purple', alpha=0.3)
line_BC, = ax.plot([], [], 'o-', lw=3, color='red', alpha=0.3)

# *** 验证点 ***
# 几何计算的 C (红点)
geom_C_plot, = ax.plot([], [], 'o', color='red', markersize=10, label='Geometric C')
# 公式计算的 C (黄叉)
formula_C_plot, = ax.plot([], [], 'x', color='yellow', markersize=10, markeredgewidth=2, label='Formula C (Verify)')

# 文本信息
txt_jacobian = ax.text(-0.45, -0.45, '', fontsize=9, fontfamily='monospace', 
                       bbox=dict(facecolor='white', alpha=0.8))

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_aspect('equal')
ax.legend(loc='upper right')
ax.grid(True)
ax.set_title("Jacobian Calculation & Formula Verification")

def update(val):
    tA, tB = s_thetaA.val, s_thetaB.val
    
    # 1. 几何计算 (用于画线)
    pts = calculate_geometry_points(tA, tB, params)
    if pts is None:
        ax.set_title("Invalid Geometry")
        return
        
    line_OA.set_data([pts['O'][0], pts['A'][0]], [pts['O'][1], pts['A'][1]])
    line_OB.set_data([pts['O'][0], pts['B'][0]], [pts['O'][1], pts['B'][1]])
    line_other.set_data([pts['A'][0], pts['D'][0], pts['E'][0]], [pts['A'][1], pts['D'][1], pts['E'][1]])
    line_BC.set_data([pts['B'][0], pts['C'][0]], [pts['B'][1], pts['C'][1]])
    geom_C_plot.set_data([pts['C'][0]], [pts['C'][1]])
    
    # 2. 公式计算与雅可比 (核心部分)
    (fCx, fCy), Jacobian = calculate_analytical_kinematics(tA, tB, params)
    
    if fCx is not None:
        # 绘制公式计算的点 -> 应该与红点完美重合
        formula_C_plot.set_data([fCx], [fCy])
        
        # 显示雅可比矩阵
        # J 0,0: dL/dThetaA  J 0,1: dL/dThetaB
        # J 1,0: dThC/dThetaA J 1,1: dThC/dThetaB
        txt_jacobian.set_text(
            f"Jacobian J (Polar Output):\n"
            f"dL/dA : {Jacobian[0,0]:.3f}  dL/dB : {Jacobian[0,1]:.3f}\n"
            f"dC/dA : {Jacobian[1,0]:.3f}  dC/dB : {Jacobian[1,1]:.3f}\n\n"
            f"Current L_OC: {np.sqrt(fCx**2+fCy**2):.2f}\n"
            f"Current Th_C: {np.degrees(np.arctan2(fCy, fCx)):.2f}°"
        )
    
    fig.canvas.draw_idle()

ax_thetaA = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_thetaB = plt.axes([0.2, 0.1, 0.65, 0.03])
s_thetaA = Slider(ax_thetaA, 'Theta A', -1080, 1080, valinit=init_theta_A)
s_thetaB = Slider(ax_thetaB, 'Theta B', -1080, 1080, valinit=init_theta_B)

s_thetaA.on_changed(update)
s_thetaB.on_changed(update)

update(None)
plt.show()