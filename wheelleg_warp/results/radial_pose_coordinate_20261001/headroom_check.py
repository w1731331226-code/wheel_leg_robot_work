"""Independent fixed-bound radial headroom check; no physical simulation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
from native.controller import D,V2,M2,inverse2,outward_force_headroom
@wp.kernel
def check(J:wp.array3d[D],out:wp.array2d[D]):
    w=wp.tid();j=M2(J[w,0,0],J[w,0,1],J[w,1,0],J[w,1,1])
    base=j*V2(D(20),D(1));room=outward_force_headroom(V2(j[0,0],j[1,0]),base,V2(D(1),D(1)))
    tau=base+V2(j[0,0],j[1,0])*wp.min(D(100),room);virtual=inverse2(j)*tau
    out[w,0]=room;out[w,1]=tau[0];out[w,2]=tau[1];out[w,3]=virtual[1]
wp.init();wp.set_device('cuda:0')
j=np.array([[[.03,.2],[-.04,.2]],[[-.03,-.2],[.04,-.2]],[[0,.2],[.04,.2]]])
out=wp.zeros((3,4),dtype=D);wp.launch(check,3,[wp.array(j,dtype=D),out]);r=out.numpy()
np.testing.assert_allclose(r[:,0],[20/3,20/3,0],atol=1e-12,rtol=0)
assert np.max(abs(r[:,1:3]))<=1+1e-12
np.testing.assert_allclose(r[:,3],1,atol=1e-12,rtol=0)
p=Path(__file__).with_name('headroom_check.json');p.write_text(json.dumps(dict(passed=True,cases=3,radial_limits_and_angular_preservation_verified=True,results=r.tolist()),indent=2)+'\n')
print('PASS radial headroom and angular preservation')
