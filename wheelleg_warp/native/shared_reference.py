"""Frozen, public-operating-region reference shared by all residual methods."""
from pathlib import Path
import numpy as np
import warp as wp
from native.controller import D,V6,fk,polar_jac


def load():
    with np.load(Path(__file__).with_suffix('.npz'),allow_pickle=False) as saved:
        states,efforts,ends,metric,thresholds=[saved[k].copy() for k in ('states','efforts','ends','metric','thresholds')]
    if states.shape!=(2,400,6) or efforts.shape!=(2,400,3) or metric.shape!=(6,6) or ends.shape!=(2,) or thresholds.shape!=(2,):
        raise ValueError('共同参考资源格式不正确')
    if not all(np.isfinite(a).all() for a in (states,efforts,metric,thresholds)) or np.any((ends<0)|(ends>=400)) or np.linalg.eigvalsh(metric).min()<=0:
        raise ValueError('共同参考资源非有限或二次型无效')
    return states,efforts,ends.astype(np.int32),metric,thresholds


@wp.kernel
def update(q:wp.array2d[float],v:wp.array2d[float],sensor:wp.array2d[float],ids:wp.array[int],
           heights:wp.array[D],angles:wp.array[D],memory:wp.array2d[D],offset:int,
           task:wp.array2d[D],param:wp.array2d[D],active:wp.array[int],reference:wp.array3d[D],
           effort:wp.array3d[D],ends:wp.array[int],metric:wp.array2d[D],thresholds:wp.array[D],request:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    if task[w,1]<D(0):
        for j in range(4):memory[w,offset+j]=D(0)
        for j in range(6):request[w,j]=D(0)
        return
    target=V6(D(0),D(0),D(0),D(0),D(0),D(0))
    use=param[w,13]<thresholds[0] and wp.abs(param[w,0])>thresholds[1]
    if use:
        qw=D(q[w,3]);qx=D(q[w,4]);qy=D(q[w,5]);qz=D(q[w,6])
        pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
        yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
        vx=wp.cos(yaw)*D(v[w,0])+wp.sin(yaw)*D(v[w,1])
        direction=int(0)
        if param[w,0]<D(0):direction=1
        if memory[w,offset+3]==D(0):
            memory[w,offset+1]=vx;memory[w,offset+3]=D(1)
        if memory[w,13]>D(0):memory[w,offset+2]=D(1)
        if memory[w,offset+2]==D(0) and wp.abs(memory[w,offset+1])>D(1.e-6):
            al=D(q[w,ids[0]]);bl=D(q[w,ids[1]]);ar=D(q[w,ids[2]]);br=D(q[w,ids[3]])
            left=fk(al,bl);right=fk(ar,br);jl=polar_jac(al,bl);jr=polar_jac(ar,br)
            length=(left[3]+right[3])/D(2);index=int(0)
            for knot in range(1,heights.shape[0]-1):
                if length>heights[knot]:index=knot
            ratio=wp.clamp((length-heights[index])/(heights[index+1]-heights[index]),D(0),D(1))
            eq=(D(1)-ratio)*angles[index]+ratio*angles[index+1]
            rate=(jl[0,1]*D(v[w,ids[4]])+jl[1,1]*D(v[w,ids[5]])+jr[0,1]*D(v[w,ids[6]])+jr[1,1]*D(v[w,ids[7]]))/D(2)
            gyro=D(sensor[w,ids[10]+1])
            x=V6((left[2]+right[2])/D(2)+D(1.5707963267948966)-pitch-eq,rate-gyro,D(0),
                 vx*reference[direction,0,3]/memory[w,offset+1],pitch,gyro)
            phase=int(memory[w,offset]);best=phase;minimum=D(1.e30)
            # ponytail: three adjacent phases, original frozen metric; no
            # predictor/optimizer online and no new tuning coefficients.
            for step in range(3):
                candidate=phase+step
                if candidate<=ends[direction]:
                    delta=V6(D(0),D(0),D(0),D(0),D(0),D(0))
                    for j in range(6):delta[j]=reference[direction,candidate,j]-x[j]
                    cost=D(0)
                    for i in range(6):
                        for j in range(6):cost+=delta[i]*metric[i,j]*delta[j]
                    if cost<minimum:minimum=cost;best=candidate
            memory[w,offset]=D(best)
            force=effort[direction,best,0];hub=effort[direction,best,1];wheel=effort[direction,best,2]
            target=V6(jl[0,0]*force+jl[0,1]*hub,jl[1,0]*force+jl[1,1]*hub,
                      jr[0,0]*force+jr[0,1]*hub,jr[1,0]*force+jr[1,1]*hub,wheel,wheel)
            peak=D(1)
            for j in range(6):peak=wp.max(peak,wp.abs(target[j]))
            for j in range(6):target[j]=target[j]/peak
    norm=D(0)
    for j in range(6):norm+=wp.abs(target[j]-request[w,j])
    scale=wp.min(D(1),D(.1)/wp.max(norm,D(1.e-30)))
    for j in range(6):request[w,j]=wp.clamp(request[w,j]+scale*(target[j]-request[w,j]),D(-1),D(1))
