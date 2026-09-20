#!/usr/bin/env python3
"""平衡 LQR v2 —— 正确的轮式倒立摆模型:
- 质心在轮地接触点之上 h=0.0993m (实测), 整体质量 0.932kg, pitch 惯量 ~2.1e-3
- 轮: R=0.025, 有效质量 m_e = m_w + I_w/R^2
- 状态 [s, ds, phi, phid], 控制 u = 轮毂力矩(正=驱动前进)
- LQR: Q=diag(1,1,50,5), R=1, dt=0.0005 (与仿真步长一致)
"""
import numpy as np

m  = 0.932    # 全机质量
h  = 0.0993   # 质心离地高度 (接触点之上)
Ip = 2.1e-3   # 绕质心 pitch 惯量
mw = 0.12
R  = 0.025
Iw = 0.5 * mw * R * R
me = mw + Iw / (R * R)

# 拉格朗日: CoM=(s + h sinφ, h cosφ), 车轮滚动 s
# T = 0.5 m ((ds + h cosφ dφ)^2 + (h sinφ dφ)^2) + 0.5 Ip dφ^2 + 0.5 me ds^2
# V = m g h cosφ ; Q_s = u/R
def dyn(x, u):
    s, ds, phi, phid = x
    M = np.array([[m + me, m*h*np.cos(phi)], [m*h*np.cos(phi), m*h*h + Ip]])
    # [M11 dds + M12 ddphi  - m h sinφ phid^2 = u/R]
    # [M12 dds + M22 ddphi  - m g h sinφ      = 0 ]
    rhs = np.array([u/R + m*h*np.sin(phi)*phid*phid, m*9.81*h*np.sin(phi)])
    acc = np.linalg.solve(M, rhs)
    return np.array([ds, acc[0], phid, acc[1]])

eps = 1e-6
A = np.zeros((4,4)); B = np.zeros((4,1))
for i in range(4):
    xp = np.zeros(4); xp[i] = eps
    xm = np.zeros(4); xm[i] = -eps
    A[:, i] = (dyn(xp, 0.0) - dyn(xm, 0.0)) / (2*eps)
B = ((dyn(np.zeros(4), eps) - dyn(np.zeros(4), -eps)) / (2*eps)).reshape(-1,1)
print("A="); print(np.round(A, 3))
print("B=", np.round(B[:,0], 3))
print("不稳定极点: ", np.round(np.sqrt(max(0, A[3,2])), 2), " rad/s (实测~10)")

Q = np.diag([1e-6, 1e-6, 50.0, 5.0])
Rq = np.array([[1.0]])
dt = 0.0005
Ad = np.eye(4) + A*dt + 0.5*(A@A)*dt*dt
Bd = B*dt
P = np.eye(4)*10
for _ in range(4000):
    K = np.linalg.solve(Rq + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    Pn = Ad.T @ P @ (Ad - Bd @ K) + Q
    if np.max(np.abs(Pn-P)) < 1e-14: break
    P = Pn
K = np.linalg.solve(Rq + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
print("\nK =", np.round(K, 4))
print("闭环极点:", np.round(np.linalg.eigvals(Ad - Bd@K), 3))
