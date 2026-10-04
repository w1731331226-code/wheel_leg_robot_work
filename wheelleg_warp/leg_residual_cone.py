"""Experimental2D actuator-metric residual projection, not a state safety proof.

At a20ms kinematic boundary forecast, avoid opposing an inward nominal joint
torque. The homogeneous sign inequalities imply nonpositive residual joint
power only when that joint is moving outward. Coupled acceleration/contact
and invariant joint bounds do not follow from these algebraic conditions.
"""
import warp as wp
from native.controller import D,V2,M2,polar_jac,allowed

A16=wp.types.matrix(shape=(16,2),dtype=D)
B16=wp.types.vector(length=16,dtype=D)


@wp.func
def feasible(x:V2,a:A16,b:B16):
    ok=True
    for k in range(16):
        if a[k,0]*x[0]+a[k,1]*x[1]>b[k]+D(1.e-10):ok=False
    return ok


@wp.func
def project(x:V2,a:A16,b:B16,q:M2):
    # Exact active-face/vertex enumeration in two dimensions.0 is feasible.
    best=V2();cost=wp.dot(x,q*x)
    det=q[0,0]*q[1,1]-q[0,1]*q[1,0]
    if det<=D(1.e-14):return best
    inverse=M2(q[1,1],-q[0,1],-q[1,0],q[0,0])/det
    if feasible(x,a,b):return x
    for i in range(16):
        row=V2(a[i,0],a[i,1]);direction=inverse*row;den=wp.dot(row,direction)
        if den>D(1.e-15):
            candidate=x-direction*((wp.dot(row,x)-b[i])/den)
            if feasible(candidate,a,b):
                d=candidate-x;c=wp.dot(d,q*d)
                if c<cost:best=candidate;cost=c
        for j in range(i+1,16):
            cross=a[i,0]*a[j,1]-a[i,1]*a[j,0]
            if wp.abs(cross)>D(1.e-12):
                candidate=V2((b[i]*a[j,1]-a[i,1]*b[j])/cross,(a[i,0]*b[j]-b[i]*a[j,0])/cross)
                if feasible(candidate,a,b):
                    d=candidate-x;c=wp.dot(d,q*d)
                    if c<cost:best=candidate;cost=c
    return best


@wp.kernel
def supervise(mode:int,qpos:wp.array2d[float],qvel:wp.array2d[float],ids:wp.array[int],active:wp.array[int],
              memory:wp.array2d[D],diag:wp.array2d[D],ctrl:wp.array2d[float],gains:wp.array2d[D],stats:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0 or diag[w,14]!=D(0):return
    jl=polar_jac(D(qpos[w,ids[0]]),D(qpos[w,ids[1]]));jr=polar_jac(D(qpos[w,ids[2]]),D(qpos[w,ids[3]]))
    a=A16();b=B16();metric=M2();guard=0
    u=V2(diag[w,12]*memory[w,16],diag[w,12]*memory[w,17])
    for j in range(4):
        side=j//2;row=j%2;matrix=jl;s=D(1)
        if side==1:matrix=jr;s=D(-1)
        m=V2(s*matrix[row,0]*D(3.4335),s*matrix[row,1])
        metric[0,0]+=m[0]*m[0];metric[0,1]+=m[0]*m[1];metric[1,0]+=m[1]*m[0];metric[1,1]+=m[1]*m[1]
        position=D(qpos[w,ids[j]]);velocity=D(qvel[w,ids[4+j]]);sign=wp.sign(position)
        near=wp.abs(position)+wp.max(D(0),sign*velocity)*D(.02)>=D(1.4)
        if near and sign*diag[w,j]<D(0):
            a[j,0]=sign*m[0];a[j,1]=sign*m[1];guard+=1
        bound=allowed(velocity,True)/wp.max(D(1),gains[w,j])
        a[4+2*j,0]=m[0];a[4+2*j,1]=m[1];b[4+2*j]=wp.max(D(0),bound-diag[w,j])
        a[5+2*j,0]=-m[0];a[5+2*j,1]=-m[1];b[5+2*j]=wp.max(D(0),bound+diag[w,j])
    a[12,0]=D(1);a[13,0]=D(-1);a[14,1]=D(1);a[15,1]=D(-1)
    for j in range(12,16):b[j]=D(1)
    selected=u
    if mode==1 and guard>0:selected=project(u,a,b,metric)
    elif mode==2:selected=V2()
    delta=selected-u;distance=wp.sqrt(wp.max(D(0),wp.dot(delta,metric*delta)))
    stats[w,0]+=D(int(guard>0));stats[w,1]+=D(int(distance>D(1.e-10)));stats[w,2]=wp.max(stats[w,2],distance)
    stats[w,5]=selected[0];stats[w,6]=selected[1]
    if mode==1 and (guard==0 or distance<=D(1.e-10)):return
    for j in range(4):
        matrix=jl;s=D(1)
        if j>=2:matrix=jr;s=D(-1)
        row=j%2;value=s*(matrix[row,0]*D(3.4335)*selected[0]+matrix[row,1]*selected[1])
        if mode!=0:
            diag[w,6+j]=value
            ctrl[w,j]=float(diag[w,j]+value)
        if mode==1:
            stats[w,3]=wp.max(stats[w,3],a[j,0]*selected[0]+a[j,1]*selected[1])
            bound=allowed(D(qvel[w,ids[4+j]]),True)/wp.max(D(1),gains[w,j])
            stats[w,4]=wp.max(stats[w,4],wp.abs(D(ctrl[w,j]))-bound)
