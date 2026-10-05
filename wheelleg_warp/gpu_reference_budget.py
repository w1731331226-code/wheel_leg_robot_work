"""Optional post-physical-projection command contraction, no safety theorem."""
import warp as wp
from native.controller import D

WIDTH=18


@wp.kernel
def apply_budget(mode:int,q:wp.array2d[float],ids:wp.array[int],reference:wp.array2d[D],active:wp.array[int],
                 diagnostic:wp.array2d[D],ctrl:wp.array2d[float],tracking:wp.array[int],stats:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    log=tracking[w]!=0
    if log:stats[w,0]+=D(1)
    if diagnostic[w,14]!=D(0):
        if log:stats[w,9]+=D(1)
        return
    rho=D(1);valid=True
    for j in range(4):
        ref=reference[w,j%2];position=D(q[w,ids[j]]);room=D(1.4)-wp.abs(ref)
        if not wp.isfinite(position) or not wp.isfinite(ref) or room<=D(0):valid=False
        else:rho=wp.min(rho,(D(1.4)-wp.abs(position))/room)
    if not valid:
        if log:stats[w,9]+=D(1)
        return
    rho=wp.clamp(rho,D(0),D(1));applied=D(1)
    if mode==1:applied=rho
    elif mode==2:applied=D(0)
    old_energy=D(0);new_energy=D(0);changed=0;zero=True;zero_error=0;excess=D(0)
    for j in range(4):
        old=ctrl[w,j];nominal=D(float(diagnostic[w,j]));delta=D(old)-nominal
        if delta!=D(0):zero=False
        old_energy+=delta*delta;selected=old
        if mode!=0 and applied!=D(1) and delta!=D(0):
            selected=float(nominal+applied*delta)
            ctrl[w,j]=selected
            diagnostic[w,6+j]=applied*delta
        elif mode==2:diagnostic[w,6+j]=D(0)
        increment=D(selected)-nominal;new_energy+=increment*increment
        if selected!=old:changed=1
        if delta==D(0) and selected!=old:zero_error+=1
        excess=wp.max(excess,wp.max(wp.min(nominal,D(old))-D(selected),D(selected)-wp.max(nominal,D(old))))
    if log:
        stats[w,1]+=applied;stats[w,2]=wp.min(stats[w,2],applied);stats[w,3]+=D(changed)
        stats[w,4]+=old_energy;stats[w,5]+=new_energy;stats[w,6]=wp.max(stats[w,6],excess)
        stats[w,8]+=D(zero_error);stats[w,10]=wp.max(stats[w,10],applied);stats[w,11]+=D(int(zero))
        stats[w,13]+=D(1);stats[w,14]+=rho;stats[w,15]+=diagnostic[w,12]
