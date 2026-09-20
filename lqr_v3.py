#!/usr/bin/env python3
"""倒立摆 LQR v3: 当前模型参数 (总重0.982kg, 质心高0.1097m)"""
import numpy as np
m  = 0.982
h  = 0.1097
Ip = 2.5e-3     # 绕质心 pitch 惯量 (箱体0.62@z0 + 腿 + 轮, 估算)
mw = 0.20
R  = 0.025
Iw = 0.5*mw*R*R
me = mw + Iw/(R*R)
def dyn(x, u):
    s, ds, phi, phid = x
    M = np.array([[m+me, m*h*np.cos(phi)],[m*h*np.cos(phi), m*h*h+Ip]])
    rhs = np.array([u/R + m*h*np.sin(phi)*phid*phid, m*9.81*h*np.sin(phi)])
    acc = np.linalg.solve(M, rhs)
    return np.array([ds, acc[0], phid, acc[1]])
eps=1e-6
A=np.zeros((4,4)); B=np.zeros((4,1))
for i in range(4):
    xp=np.zeros(4); xp[i]=eps; xm=np.zeros(4); xm[i]=-eps
    A[:,i]=(dyn(xp,0)-dyn(xm,0))/(2*eps)
B=((dyn(np.zeros(4),eps)-dyn(np.zeros(4),-eps))/(2*eps)).reshape(-1,1)
print("A[3,2]=", round(A[3,2],2), " (不稳定极点=%.1f rad/s)" % np.sqrt(A[3,2]))
Q = np.diag([2.0, 1.0, 60.0, 6.0])
Rq = np.array([[1.0]])
dt = 0.0005
Ad = np.eye(4)+A*dt+0.5*(A@A)*dt*dt
Bd = B*dt
P = np.eye(4)*10
for _ in range(5000):
    K = np.linalg.solve(Rq+Bd.T@P@Bd, Bd.T@P@Ad)
    Pn = Ad.T@P@(Ad-Bd@K)+Q
    if np.max(np.abs(Pn-P))<1e-14: break
    P = Pn
K = np.linalg.solve(Rq+Bd.T@P@Bd, Bd.T@P@Ad)
print("K =", np.round(K,4))
print("闭环极点:", np.round(np.linalg.eigvals(Ad-Bd@K),3))
