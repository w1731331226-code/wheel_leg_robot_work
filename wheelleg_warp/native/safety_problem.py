"""实验性双精度预测/屏障/LP装配核；不接入默认控制。"""
import warp as wp
from native.controller import D, V3, fk, polar_jac, allowed


@wp.func
def barrier_row(w:int,r:int,h:D,rate:D,acceleration:D,g:V3,
                A:wp.array3d[D],b:wp.array2d[D],info:wp.array3d[D]):
    info[w,r,0]=h;info[w,r,1]=rate;info[w,r,2]=acceleration
    flag=int(0)
    if h<=D(0):flag=1
    if not wp.isfinite(h) or not wp.isfinite(rate) or not wp.isfinite(acceleration):flag=2
    for j in range(3):
        if not wp.isfinite(g[j]):flag=2
    if h>D(0) and rate<D(0):
        for j in range(3):A[w,r,j]=-g[j]
        b[w,r]=acceleration-rate*rate/(D(2)*h)
    return flag


@wp.kernel
def assemble(q:wp.array2d[D],v:wp.array2d[D],a0:wp.array2d[D],speeds:wp.array2d[D],nominal:wp.array2d[D],
             gain:wp.array2d[D],reserve:wp.array[D],ctrl_limits:wp.array[D],lower:D,gamma_length:D,gamma_joint:D,
             A:wp.array3d[D],b:wp.array2d[D],B:wp.array3d[D],box:wp.array2d[D],info:wp.array3d[D],
             diagnostics:wp.array2d[D],flags:wp.array[int]):
    w=wp.tid();flag=int(0)
    for r in range(34):
        b[w,r]=D(0)
        for j in range(4):A[w,r,j]=D(0)
    for r in range(6):
        for j in range(3):B[w,r,j]=D(0)
    hessian_error=D(0)
    for side in range(2):
        k=2*side;x=q[w,k];y=q[w,k+1];vx=v[w,k];vy=v[w,k+1]
        jac=polar_jac(x,y);geometry=fk(x,y)
        for j in range(2):
            B[w,k,j]=jac[0,j];B[w,k+1,j]=jac[1,j]
        full=D(0);half=D(0)
        for trial in range(2):
            eps=D(1.e-5)
            if trial==1:eps=D(5.e-6)
            plus=polar_jac(x+eps*vx,y+eps*vy);minus=polar_jac(x-eps*vx,y-eps*vy)
            term=((plus[0,0]-minus[0,0])*vx+(plus[1,0]-minus[1,0])*vy)/(D(2)*eps)
            if trial==0:full=term
            else:half=term
        hessian_error=wp.max(hessian_error,wp.abs(full-half))
        g=V3()
        for j in range(3):g[j]=jac[0,0]*gain[k,j]+jac[1,0]*gain[k+1,j]
        flag=wp.max(flag,barrier_row(w,side,geometry[3]-lower-reserve[0]-gamma_length,
            jac[0,0]*vx+jac[1,0]*vy,jac[0,0]*a0[w,k]+jac[1,0]*a0[w,k+1]+half,g,A,b,info))
    for k in range(4):
        for bound_side in range(2):
            sign=D(-1)
            if bound_side==1:sign=D(1)
            g=V3(sign*gain[k,0],sign*gain[k,1],sign*gain[k,2])
            flag=wp.max(flag,barrier_row(w,2+2*k+bound_side,D(1.5)-reserve[1]-gamma_joint+sign*q[w,k],
                sign*v[w,k],sign*a0[w,k],g,A,b,info))
    B[w,4,2]=D(1);B[w,5,2]=D(1)
    for motor in range(6):
        bound=wp.min(allowed(speeds[w,motor],motor<4),ctrl_limits[motor]);box[w,motor]=bound
        r=10+4*motor
        for j in range(3):
            A[w,r,j]=B[w,motor,j];A[w,r+1,j]=-B[w,motor,j]
            A[w,r+2,j]=B[w,motor,j];A[w,r+3,j]=-B[w,motor,j]
        A[w,r,3]=D(-1);A[w,r+1,3]=D(-1)
        b[w,r+2]=bound-nominal[w,motor];b[w,r+3]=bound+nominal[w,motor]
    predicted_length=D(1.e30);predicted_joint=D(1.e30)
    for step in range(1,11):
        time=D(step)*D(.0005)
        for side in range(2):
            k=2*side
            x=q[w,k]+v[w,k]*time+D(.5)*a0[w,k]*time*time
            y=q[w,k+1]+v[w,k+1]*time+D(.5)*a0[w,k+1]*time*time
            geometry=fk(x,y);predicted_length=wp.min(predicted_length,geometry[3])
            predicted_joint=wp.min(predicted_joint,wp.min(D(1.5)-wp.abs(x),D(1.5)-wp.abs(y)))
    diagnostics[w,0]=hessian_error
    diagnostics[w,1]=predicted_length-lower-reserve[0]-gamma_length
    diagnostics[w,2]=predicted_joint-reserve[1]-gamma_joint
    if hessian_error>=D(1.e-4):flag=2
    flags[w]=flag
