"""Frozen shared request; public height/speed interpolation needs domain checks."""
from pathlib import Path
import numpy as np
import warp as wp
from native.controller import D


def load_reference():
    with np.load(Path(__file__).with_suffix('.npz'),allow_pickle=False) as saved:
        plan=saved['schedule'].copy()
    if plan.shape!=(400,2,6) or not np.isfinite(plan).all() or np.max(abs(plan))>1.+1e-12:
        raise ValueError('冻结共同参考须为400×2×6、有限且每电机不超过1Nm')
    delta=np.diff(np.concatenate([np.zeros_like(plan[:1]),plan]),axis=0)
    if np.max(abs(delta).sum(axis=2))>.1+1e-12:raise ValueError('冻结参考超过5ms L1变化域')
    return plan


@wp.kernel
def update_reference(state:wp.array2d[D],param:wp.array2d[D],active:wp.array[int],
                     plan:wp.array3d[D],request:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    weight=wp.clamp((wp.abs(param[w,0])-D(.75))/D(.25),D(0),D(1))
    weight*=wp.clamp((D(.12)-param[w,13])/(D(.12)-D(.115)),D(0),D(1))
    if state[w,1]<D(0) or weight==D(0):
        for j in range(6):request[w,j]=D(0)
        return
    arrival_step=int(state[w,1]*D(2000)+D(.5))
    start=((arrival_step+9)/10)*10
    elapsed=int(state[w,0])-start
    if elapsed<0:return
    direction=int(0)
    if param[w,0]<D(0):direction=1
    slot=wp.min(elapsed/10,399)
    for j in range(6):request[w,j]=weight*plan[slot,direction,j]
