"""Single prespecified constant0.9 leg command control, not an extra search."""
import warp as wp
from native.controller import D


@wp.kernel
def apply_fixed(alpha:D,active:wp.array[int],diagnostic:wp.array2d[D],ctrl:wp.array2d[float],tracking:wp.array[int],stats:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    log=tracking[w]!=0
    if log:stats[w,0]+=D(1)
    if diagnostic[w,14]!=D(0):
        if log:stats[w,9]+=D(1)
        return
    old_energy=D(0);new_energy=D(0);changed=0;zero=True;excess=D(0)
    for j in range(4):
        old=ctrl[w,j];nominal=D(float(diagnostic[w,j]));delta=D(old)-nominal
        if delta!=D(0):zero=False
        old_energy+=delta*delta;selected=old
        if delta!=D(0):
            selected=float(nominal+alpha*delta);ctrl[w,j]=selected;diagnostic[w,6+j]=alpha*delta
        increment=D(selected)-nominal;new_energy+=increment*increment
        if selected!=old:changed=1
        excess=wp.max(excess,wp.max(wp.min(nominal,D(old))-D(selected),D(selected)-wp.max(nominal,D(old))))
    if log:
        stats[w,1]+=alpha;stats[w,2]=wp.min(stats[w,2],alpha);stats[w,3]+=D(changed)
        stats[w,4]+=old_energy;stats[w,5]+=new_energy;stats[w,6]=wp.max(stats[w,6],excess)
        stats[w,10]=wp.max(stats[w,10],alpha);stats[w,11]+=D(int(zero));stats[w,13]+=D(1);stats[w,14]+=alpha;stats[w,15]+=diagnostic[w,12]
