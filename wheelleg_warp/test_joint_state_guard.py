"""GPU projection vs independent SciPy oracle, plus guarded synthetic states."""
import numpy as np
import warp as wp
from scipy.optimize import minimize
import joint_state_guard as guard


@wp.kernel
def batch(target:wp.array2d[guard.D],a:wp.array3d[guard.D],b:wp.array2d[guard.D],out:wp.array2d[guard.D]):
    w=wp.tid();matrix=guard.A8();rhs=guard.V8()
    for j in range(8):
        matrix[j,0]=a[w,j,0];matrix[j,1]=a[w,j,1];rhs[j]=b[w,j]
    value,ok=guard.project(guard.V2(target[w,0],target[w,1]),matrix,rhs)
    out[w,0]=value[0];out[w,1]=value[1];out[w,2]=guard.D(ok)


def run():
    rng=np.random.default_rng(32831);n=128;a=rng.normal(size=(n,8,2));center=rng.uniform(-.5,.5,(n,2))
    b=np.einsum('nij,nj->ni',a,center)+rng.uniform(.01,1,(n,8));target=rng.normal(size=(n,2))*2
    a[-1]=0;b[-1]=1;a[-1,0]=[1,0];b[-1,0]=-1;a[-1,1]=[-1,0];b[-1,1]=-1
    result=wp.zeros((n,3),dtype=guard.D,device='cuda:0')
    wp.launch(batch,n,[wp.array(target,dtype=guard.D,device='cuda:0'),wp.array(a,dtype=guard.D,device='cuda:0'),wp.array(b,dtype=guard.D,device='cuda:0'),result])
    values=result.numpy();assert values[-1,2]==0
    for i in range(n-1):
        oracle=minimize(lambda u:.5*np.sum((u-target[i])**2),center[i],jac=lambda u:u-target[i],
            constraints=[dict(type='ineq',fun=lambda u,i=i:b[i]-a[i]@u,jac=lambda u,i=i:-a[i])],method='SLSQP',options=dict(ftol=1e-12,maxiter=100))
        assert oracle.success and values[i,2]==1
        np.testing.assert_allclose(values[i,:2],oracle.x,rtol=0,atol=2e-7)
        assert np.max(a[i]@values[i,:2]-b[i])<=1e-8
    # Entire controller module compiles; synthetic q/v is not robot integration.
    n=3;q=np.zeros((n,4),np.float32);q[:]=[1.,-1.,1.,-1.];q[1,1]=-1.399
    v=np.zeros((n,6),np.float32);v[1,1]=-.2
    ids=wp.array([0,1,2,3,0,1,2,3,4,5],dtype=int,device='cuda:0')
    ctrl=wp.zeros((n,6),device='cuda:0');diag=wp.zeros((n,38),dtype=guard.D,device='cuda:0');memory=wp.zeros((n,16),dtype=guard.D,device='cuda:0');log=wp.zeros((40,n,62),dtype=guard.D,device='cuda:0')
    wp.launch(guard.guard,n,[0,wp.array(q,device='cuda:0'),wp.array(v,device='cuda:0'),wp.array([1,1,0],dtype=int,device='cuda:0'),ids,ctrl,diag,guard.V6(1.,1.,1.,1.,1.05,1.05),memory,log])
    t=log.numpy()[0];assert np.isfinite(t).all();assert (t[:2,28:30]==1).all() and t[2,0]==0
    assert np.max(t[:2,58])<=1e-5 and np.max(t[:2,61])<=1e-5
    assert np.max(abs(t[1,14:20]))>0,'Unsafe nominal prediction must be corrected before residual projection'
    before=memory.numpy().copy();wp.launch(guard.clear,n,[wp.array([0,1,0],dtype=int,device='cuda:0'),memory]);after=memory.numpy()
    np.testing.assert_array_equal(after[1],0);np.testing.assert_array_equal(after[[0,2]],before[[0,2]])
    print('PASS328 GPU127convexQP vsSciPy/infeasible,syntheticNom correction/constraints/inactive/maskedreset;0robot/physics/learning')


if __name__=='__main__':run()
