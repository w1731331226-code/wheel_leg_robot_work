"""Experimental nominal incremental joint guard; prediction is not a safety certificate."""
import numpy as np
import warp as wp
from native.controller import D,V2,V4,V6,M2,fk,polar_jac,inverse2,command_bounds,sim
import parking_withdrawal_probe as parking

V8=wp.types.vector(length=8,dtype=D)
A8=wp.types.matrix(shape=(8,2),dtype=D)
MASS_UPPER=wp.constant(8.)
ROTOR=wp.constant(sim.hw.HIP_OUTPUT_INERTIA)
RATE=wp.constant(50.)  # One 20ms actor interval; fixed before regression, not fitted to failures.


@wp.func
def gain(qa:D,qb:D):
    j=polar_jac(qa,qb);r=fk(qa,qb)[3]
    # Conservative public mass at each leg; unmodelled contact/cross-leg effects remain.
    m=M2(D(MASS_UPPER),D(0),D(0),D(MASS_UPPER)*r*r)
    inertia=j*m*wp.transpose(j)+M2(D(ROTOR),D(0),D(0),D(ROTOR))
    return inverse2(inertia)


@wp.func
def inside(u:V2,a:A8,b:V8):
    valid=int(1)
    for i in range(8):
        if a[i,0]*u[0]+a[i,1]*u[1]>b[i]+D(1.e-9):valid=0
    return valid


@wp.func
def project(target:V2,a:A8,b:V8):
    # Existing residual-cone solver assumes feasible zero; nominal correction need not contain zero.
    best=target;cost=D(1.e30);found=int(0)
    if inside(target,a,b):best=target;cost=D(0);found=1
    for i in range(8):
        normal=V2(a[i,0],a[i,1]);square=wp.dot(normal,normal)
        if square>D(1.e-20):
            u=target+normal*((b[i]-wp.dot(normal,target))/square)
            if inside(u,a,b):
                delta=u-target;c=wp.dot(delta,delta)
                if c<cost:best=u;cost=c;found=1
        for k in range(i+1,8):
            det=a[i,0]*a[k,1]-a[i,1]*a[k,0]
            if wp.abs(det)>D(1.e-12):
                u=V2((b[i]*a[k,1]-a[i,1]*b[k])/det,(a[i,0]*b[k]-b[i]*a[k,0])/det)
                if inside(u,a,b):
                    delta=u-target;c=wp.dot(delta,delta)
                    if c<cost:best=u;cost=c;found=1
    return best,found


@wp.kernel
def clear(mask:wp.array[int],memory:wp.array2d[D]):
    w=wp.tid()
    if mask[w]:
        for j in range(16):memory[w,j]=D(0)


@wp.kernel
def guard(slot:int,q:wp.array2d[float],v:wp.array2d[float],active:wp.array[int],ids:wp.array[int],
          ctrl:wp.array2d[float],diag:wp.array2d[D],upper:V6,memory:wp.array2d[D],log:wp.array3d[D]):
    w=wp.tid()
    for j in range(62):log[slot,w,j]=D(0)
    if active[w]==0 or diag[w,14]!=D(0):return
    base=V6();res=V6();safe=V6();speeds=V6();previous=V6()
    for j in range(6):
        base[j]=diag[w,j];res[j]=diag[w,6+j];safe[j]=base[j]
        speeds[j]=D(v[w,ids[4+j]]);previous[j]=memory[w,5+j]
    bounds=command_bounds(speeds,upper);acc=V4()
    ready=memory[w,0]>D(0);memory[w,15]+=D(1)
    for j in range(4):
        if ready:acc[j]=(speeds[j]-memory[w,1+j])/D(.0005)
        log[slot,w,30+j]=acc[j];log[slot,w,50+j]=D(q[w,ids[j]]);log[slot,w,54+j]=speeds[j]
        if ready:log[slot,w,38+j]=acc[j]-memory[w,11+j]
    feasibility=V2();gleft=gain(D(q[w,ids[0]]),D(q[w,ids[1]]));gright=gain(D(q[w,ids[2]]),D(q[w,ids[3]]))
    for side in range(2):
        start=2*side;g=gleft
        if side==1:g=gright
        a=A8();b=V8();target=V2(base[start],base[start+1]);old=V2(previous[start],previous[start+1]);offset=V2(acc[start],acc[start+1])-g*old
        for j in range(2):
            qi=D(q[w,ids[start+j]]);vi=speeds[start+j]
            lo=-D(2)*D(RATE)*vi-D(RATE)*D(RATE)*(D(1.4)+qi)
            hi=-D(2)*D(RATE)*vi+D(RATE)*D(RATE)*(D(1.4)-qi)
            for k in range(2):a[2*j,k]=g[j,k];a[2*j+1,k]=-g[j,k]
            b[2*j]=hi-offset[j];b[2*j+1]=-lo+offset[j]
            a[4+2*j,j]=D(1);a[5+2*j,j]=D(-1);b[4+2*j]=bounds[start+j];b[5+2*j]=bounds[start+j]
        result,ok=project(target,a,b);feasibility[side]=D(ok)
        if ok==0:
            # Infeasible prediction is explicitly logged; saturation is not labelled safe.
            for j in range(2):result[j]=-wp.sign(D(q[w,ids[start+j]]))*bounds[start+j]
        safe[start]=result[0];safe[start+1]=result[1]
    lam=D(1)
    if feasibility[0]==D(0) or feasibility[1]==D(0):lam=D(0)
    for j in range(6):
        if res[j]>D(0):lam=wp.min(lam,(bounds[j]-safe[j])/res[j])
        elif res[j]<D(0):lam=wp.min(lam,(-bounds[j]-safe[j])/res[j])
    for side in range(2):
        start=2*side;g=gleft
        if side==1:g=gright
        predicted=V2(acc[start],acc[start+1])+g*(V2(safe[start],safe[start+1])-V2(previous[start],previous[start+1]))
        delta=g*V2(res[start],res[start+1])
        for j in range(2):
            qi=D(q[w,ids[start+j]]);vi=speeds[start+j]
            lo=-D(2)*D(RATE)*vi-D(RATE)*D(RATE)*(D(1.4)+qi)
            hi=-D(2)*D(RATE)*vi+D(RATE)*D(RATE)*(D(1.4)-qi)
            if delta[j]>D(0):lam=wp.min(lam,(hi-predicted[j])/delta[j])
            elif delta[j]<D(0):lam=wp.min(lam,(lo-predicted[j])/delta[j])
    lam=wp.clamp(lam,D(0),D(1));violation=D(-1.e30);psi=D(1.e30);cost=D(0);bound_error=D(0)
    final=V6()
    for j in range(6):
        final[j]=D(float(wp.clamp(safe[j]+lam*res[j],-bounds[j],bounds[j])))
        ctrl[w,j]=float(final[j]);cost+=(safe[j]-base[j])*(safe[j]-base[j])
        log[slot,w,2+j]=base[j];log[slot,w,8+j]=res[j];log[slot,w,14+j]=safe[j];log[slot,w,20+j]=final[j]
        bound_error=wp.max(bound_error,wp.abs(final[j])-bounds[j]);diag[w,j]=safe[j];diag[w,6+j]=final[j]-safe[j]
    for side in range(2):
        start=2*side;g=gleft
        if side==1:g=gright
        pred=V2(acc[start],acc[start+1])+g*(V2(final[start],final[start+1])-V2(previous[start],previous[start+1]))
        for j in range(2):
            k=start+j;qi=D(q[w,ids[k]]);vi=speeds[k]
            hplus=D(1.4)-qi;hminus=D(1.4)+qi
            low=-D(2)*D(RATE)*vi-D(RATE)*D(RATE)*hminus;high=-D(2)*D(RATE)*vi+D(RATE)*D(RATE)*hplus
            violation=wp.max(violation,wp.max(low-pred[j],pred[j]-high));psi=wp.min(psi,wp.min(-vi+D(RATE)*hplus,vi+D(RATE)*hminus))
            log[slot,w,34+k]=pred[j];log[slot,w,42+k]=wp.min(hplus,hminus);log[slot,w,46+k]=-wp.sign(qi)*vi
            memory[w,11+k]=pred[j]
    log[slot,w,0]=D(1);log[slot,w,1]=memory[w,15];log[slot,w,26]=diag[w,12];log[slot,w,27]=lam
    log[slot,w,28]=feasibility[0];log[slot,w,29]=feasibility[1];log[slot,w,58]=violation;log[slot,w,59]=psi;log[slot,w,60]=wp.sqrt(cost);log[slot,w,61]=bound_error
    diag[w,12]*=lam;memory[w,0]=D(1)
    for j in range(4):memory[w,1+j]=speeds[j]
    for j in range(6):memory[w,5+j]=final[j]


def instrument(factory,cases,directory):
    if not cases or len(cases)>20 or directory is None:raise ValueError('Guard diagnostic requires1…20 cases andoutput directory')
    n=len(cases);memory=wp.zeros((n,16),dtype=D);log=wp.zeros((40,n,62),dtype=D);mask=wp.ones(n,dtype=int)
    launch=wp.launch;calls=0
    def intercept(kernel,dim,inputs=None,**kwargs):
        nonlocal calls
        result=launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
        if kernel is parking.phase.roles.experimental.control_physical_nominal:
            slot=calls%40;calls+=1
            launch(guard,dim,[slot,inputs[0],inputs[1],inputs[5],inputs[7],inputs[14],inputs[15],V6(1.,1.,1.,1.,1.05,1.05),memory,log])
        return result
    wp.launch=intercept
    try:result=factory()
    finally:wp.launch=launch
    raw=result[0];assert calls==80 and raw.num_envs==n
    reset,wait=raw.reset,raw.step_wait;chunks=[[] for _ in cases];frozen=set()
    def reset_all():
        value=reset();memory.zero_();log.zero_();frozen.clear()
        for part in chunks:part.clear()
        return value
    def step_wait():
        value=wait();frames=log.numpy()
        for w in range(n):
            if w in frozen:continue
            keep=frames[:,w,0]==1
            if keep.any():chunks[w].append(frames[keep,w].copy())
            if value[2][w]:
                data=np.concatenate(chunks[w]);assert len(data)==value[3][w]['physical_steps']
                np.testing.assert_array_equal(data[:,1],np.arange(1,len(data)+1));assert np.isfinite(data).all()
                file=directory/f'joint_guard_{cases[w]["seed"]}.npz';assert not file.exists();np.savez_compressed(file,trace=data)
                value[3][w]['joint_guard_trace']=dict(path=file.name,sha256=parking.phase.rec.sha(file),rows=len(data))
                frozen.add(w);chunks[w].clear()
        if value[2].any():mask.assign(value[2].astype(np.int32));wp.launch(clear,n,[mask,memory])
        return value
    raw.reset,raw.step_wait=reset_all,step_wait;raw._joint_guard_buffers=(memory,log,mask)
    return result
