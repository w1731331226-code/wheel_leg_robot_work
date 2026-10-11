"""V2 of the frozen failed residual kernel: project against already accepted Nom.

V1 remains unchanged for reproduction. Only the effective projection mode differs.
"""
from cartesian_pair_kernel import wp,D,V2,V3,V6,fk,polar_jac,command_bounds,project_bounds


@wp.kernel
def deliver_accepted_nominal(slot:int,mode:int,q:wp.array2d[float],v:wp.array2d[float],ids:wp.array[int],
            effective:wp.array2d[float],active:wp.array[int],latent:wp.array2d[D],reference:wp.array2d[D],
            state:wp.array2d[D],shadow:wp.array2d[D],ctrl:wp.array2d[float],diag:wp.array2d[D],trace:wp.array3d[D]):
    w=wp.tid()
    for j in range(38):trace[slot,w,j]=D(0)
    if active[w]==0:return
    trace[slot,w,0]=D(1)
    for j in range(3):trace[slot,w,1+j]=latent[w,j]
    if diag[w,14]!=D(0):
        trace[slot,w,33]=diag[w,14]
        return
    qa=D(q[w,ids[0]]);qb=D(q[w,ids[1]]);qc=D(q[w,ids[2]]);qd=D(q[w,ids[3]])
    left=fk(qa,qb);right=fk(qc,qd)
    rl=V2(left[0]-D(.075),left[1]);rr=V2(right[0]-D(.075),right[1])
    if mode==1:rl=V2(reference[w,0],reference[w,1]);rr=V2(reference[w,2],reference[w,3])
    valid=(mode==0 or mode==1) and wp.isfinite(qa) and wp.isfinite(qb) and wp.isfinite(qc) and wp.isfinite(qd)
    for j in range(2):valid=valid and wp.isfinite(rl[j]) and wp.isfinite(rr[j])
    valid=valid and wp.length(rl)>D(1.e-6) and wp.length(rr)>D(1.e-6)
    for j in range(3):
        valid=valid and wp.isfinite(D(effective[w,2*j])) and wp.abs(D(effective[w,2*j]))<=D(1)
        valid=valid and effective[w,2*j+1]==-effective[w,2*j]
        valid=valid and wp.isfinite(latent[w,j]) and wp.abs(latent[w,j])<=D(1)
    if not valid:
        diag[w,14]=D(2);trace[slot,w,33]=D(2)
        for j in range(6):ctrl[w,j]=0.;diag[w,6+j]=D(0)
        return
    a=V3()
    for j in range(3):a[j]=latent[w,j]+wp.clamp(D(effective[w,2*j])-latent[w,j],D(-.01),D(.01))
    force=V2(D(3.4335)*a[0],D(3.4335)*a[1])
    u=V6(wp.dot(rl,force)/(wp.length(rl)*D(3.4335)),-wp.dot(rr,force)/(wp.length(rr)*D(3.4335)),
        rl[0]*force[1]-rl[1]*force[0],-(rr[0]*force[1]-rr[1]*force[0]),a[2],-a[2])
    denominator=wp.max(D(1),wp.abs(u[2]+u[3])/D(.1))
    for j in range(6):denominator=wp.max(denominator,wp.abs(u[j]))
    alpha=D(1)/denominator
    for j in range(6):u[j]*=alpha
    jl=polar_jac(qa,qb);jr=polar_jac(qc,qd)
    tl=jl*V2(D(3.4335)*u[0],u[2]);tr=jr*V2(D(3.4335)*u[1],u[3])
    residual=V6(tl[0],tl[1],tr[0],tr[1],D(.3)*u[4],D(.3)*u[5]);base=V6();speed=V6()
    nonzero=bool(False);bad_map=bool(False)
    for side in range(2):
        matrix=jl
        if side==1:matrix=jr
        square=matrix[0,0]*matrix[0,0]+matrix[0,1]*matrix[0,1]+matrix[1,0]*matrix[1,0]+matrix[1,1]*matrix[1,1]
        determinant=wp.abs(matrix[0,0]*matrix[1,1]-matrix[0,1]*matrix[1,0])
        maximum=(square+wp.sqrt(wp.max(D(0),square*square-D(4)*determinant*determinant)))/D(2)
        if determinant==D(0) or maximum>determinant*D(1.e6):bad_map=True
    for j in range(6):
        base[j]=diag[w,j];speed[j]=D(v[w,ids[4+j]])
        nonzero=nonzero or u[j]!=D(0)
        valid=valid and wp.isfinite(base[j]) and wp.isfinite(speed[j]) and wp.isfinite(residual[j])
    if not valid:
        diag[w,14]=D(2);trace[slot,w,33]=D(2)
        for j in range(6):ctrl[w,j]=0.;diag[w,6+j]=D(0)
        return
    bounds=command_bounds(speed,V6(D(1),D(1),D(1),D(1),D(1.05),D(1.05)))
    lam,error=project_bounds(base,residual,speed,bounds,int(diag[w,13]),bad_map,nonzero,False,1)
    if error:
        diag[w,14]=D(error);trace[slot,w,33]=D(error)
        for j in range(6):ctrl[w,j]=0.;diag[w,6+j]=D(0)
        return
    for j in range(3):
        latent[w,j]=a[j];shadow[w,16+2*j]=a[j];shadow[w,17+2*j]=-a[j];trace[slot,w,4+j]=a[j]
    for j in range(6):
        state[w,16+j]=u[j]
        ctrl[w,j]=float(wp.clamp(base[j]+lam*residual[j],-bounds[j],bounds[j]))
        diag[w,6+j]=lam*residual[j]
        trace[slot,w,7+j]=u[j];trace[slot,w,15+j]=residual[j]
        trace[slot,w,21+j]=D(ctrl[w,j]);trace[slot,w,27+j]=bounds[j]
    diag[w,12]*=lam;trace[slot,w,13]=alpha;trace[slot,w,14]=lam
    trace[slot,w,34]=rl[0];trace[slot,w,35]=rl[1];trace[slot,w,36]=rr[0];trace[slot,w,37]=rr[1]
