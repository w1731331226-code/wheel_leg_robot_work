"""One shared Nom authority pool;not additive independent correction budgets."""
import numpy as np
import warp as wp
from native.controller import D


def bounded_total(previous,original,delta,damping):
    arrays=[np.asarray(x,dtype=float) for x in (previous,original,delta,damping)]
    if any(x.shape!=arrays[0].shape or x.ndim!=2 or x.shape[1]!=6 or not np.isfinite(x).all() for x in arrays) or np.any(np.abs(arrays[0])>1):
        raise ValueError('Finite sixmotor request and valid previous budget required')
    previous,original,delta,damping=arrays
    target=np.clip(original+delta+damping,-1,1);change=target-previous
    norm=np.sum(abs(change),axis=1);factor=np.minimum(1,.1/np.maximum(norm,1e-30))
    return np.clip(previous+factor[:,None]*change,-1,1)


@wp.kernel
def compose(previous:wp.array2d[D],original:wp.array2d[D],delta:wp.array2d[D],damping:wp.array2d[D],
            active:wp.array[int],out:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    norm=D(0)
    for j in range(6):norm+=wp.abs(wp.clamp(original[w,j]+delta[w,j]+damping[w,j],D(-1),D(1))-previous[w,j])
    factor=wp.min(D(1),D(.1)/wp.max(norm,D(1e-30)))
    for j in range(6):out[w,j]=wp.clamp(previous[w,j]+factor*(wp.clamp(original[w,j]+delta[w,j]+damping[w,j],D(-1),D(1))-previous[w,j]),D(-1),D(1))


@wp.kernel
def reference_copy(original:wp.array2d[D],motion:wp.array[D],shadow:wp.array2d[D]):
    w=wp.tid()
    for j in range(16):shadow[w,j]=original[w,j]
    shadow[w,16]=motion[w]


def unit():
    rng=np.random.default_rng(218);previous=rng.uniform(-1,1,(32,6));original=rng.uniform(-1,1,(32,6));delta=rng.normal(0,3,(32,6));damping=rng.normal(0,3,(32,6))
    expected=bounded_total(previous,original,delta,damping)
    assert np.all(np.sum(abs(expected-previous),axis=1)<=.1+1e-14) and np.all(abs(expected)<=1)
    for device in ('cpu','cuda'):
        arrays=[wp.array(x,dtype=D,device=device) for x in (previous,original,delta,damping)];out=wp.array(previous,dtype=D,device=device)
        wp.launch(compose,32,[*arrays,wp.ones(32,dtype=wp.int32,device=device),out],device=device)
        np.testing.assert_allclose(out.numpy(),expected,atol=1e-14,rtol=0)
    target=bounded_total(np.zeros((1,6)),np.zeros((1,6)),np.ones((1,6))*10,np.ones((1,6))*10)
    assert np.sum(abs(target))<=.1+1e-14  # Two proposals still get one shared budget.
    np.testing.assert_array_equal(bounded_total(expected,expected,np.zeros_like(expected),np.zeros_like(expected)),expected)


if __name__=='__main__':unit();print('PASS one-pool1Nm/L1.1Nm5ms CPU-CUDA budget;no doubled authority')
