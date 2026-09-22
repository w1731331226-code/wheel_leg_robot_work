"""Original-direction projection: torque bounds, guard cases and real controller integration."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
from native.controller import D,V6,residual_projection,control
from native.terrain_env import TerrainEnv
from native.terrain import TerrainScenario
from dashboard.live_env import atomic_json as write


@wp.kernel
def project(base:wp.array2d[D],residual:wp.array2d[D],speed:wp.array2d[D],flags:wp.array2d[int],mode:int,out:wp.array2d[D]):
    w=wp.tid();b=V6();r=V6();v=V6()
    for k in range(6):b[k]=base[w,k];r[k]=residual[w,k];v[k]=speed[w,k]
    lam,error=residual_projection(b,r,v,flags[w,0],flags[w,1]!=0,True,flags[w,2]!=0,mode)
    out[w,0]=lam;out[w,1]=D(error)


def check(output):
    wp.init();wp.set_device('cuda:0');rng=np.random.default_rng(76301);n=1006
    speed=rng.uniform(-1,1,(n,6))*[35,35,35,35,85,85]
    rpm=abs(speed)*60/(2*np.pi);rated=np.array([175]*4+[490]*2);no_load=np.array([280]*4+[710]*2)
    bound=np.array([40]*4+[4.5]*2)*np.clip((no_load-rpm)/(no_load-rated),0,1)
    base=rng.uniform(-1,1,(n,6))*bound;residual=rng.uniform(-1,1,(n,6));flags=np.zeros((n,3),np.int32)
    flags[:,0]=rng.integers(0,2,n);flags[:6]=0
    speed[:6]=0;bound[:6]=[40]*4+[4.5]*2;base[:6]=0;residual[:6]=.1
    base[1:4,4]=4.5;flags[1:4,0]=1;residual[1,4]=-.1
    base[3,5]=4.5;residual[3,4:]=[-.1,.1]  # Diff3 blocked by the other saturated wheel.
    flags[4,1]=1;flags[5,:2]=1  # Singular map with/without base infeasibility.
    gb=wp.array(base,dtype=D);gr=wp.array(residual,dtype=D);gv=wp.array(speed,dtype=D);gf=wp.array(flags,dtype=wp.int32);out=wp.zeros((n,2),dtype=D)
    modes=[]
    for mode in (0,1):
        wp.launch(project,n,[gb,gr,gv,gf,mode,out]);got=out.numpy()
        limits=np.ones_like(residual);np.divide(bound-base,residual,out=limits,where=residual>0);np.divide(-bound-base,residual,out=limits,where=residual<0)
        expected=np.clip(limits.min(axis=1),0,1)
        expected[(flags[:,1]!=0)&((flags[:,0]==0)|(mode==1))]=0
        if mode==0:expected[flags[:,0]!=0]=0
        np.testing.assert_allclose(got[:,0],expected,atol=1e-12,rtol=0)
        assert np.all(abs(base+got[:,0,None]*residual)<=bound+1e-10)
        modes.append(got)
    np.testing.assert_array_equal(modes[0][:6,0],[1,0,0,0,0,0])
    np.testing.assert_array_equal(modes[1][:6,0],[1,1,0,0,0,0])
    assert modes[0][5,1]==0 and modes[1][5,1]==1  # Preserve old semantics, protect new mode.
    for target in (gb,gr,gv):
        saved=target.numpy();bad=saved.copy();bad[0,0]=np.nan;bad[1,0]=np.inf;target.assign(bad)
        wp.launch(project,n,[gb,gr,gv,gf,1,out]);np.testing.assert_array_equal(out.numpy()[:2],[[0,2],[0,2]])
        target.assign(saved)
    flags[0,2]=1;gf.assign(flags);wp.launch(project,n,[gb,gr,gv,gf,1,out]);np.testing.assert_array_equal(out.numpy()[0],[0,2])
    for bad in ('true',1,None):
        try:TerrainEnv(1,project_clipped_base=bad)
        except ValueError:pass
        else:raise AssertionError('Invalid switch accepted')
    a=TerrainEnv(1,scenario=TerrainScenario());b=TerrainEnv(1,scenario=TerrainScenario(),project_clipped_base=True)
    try:
        a.reset();b.reset()
        for _ in range(50):
            action=rng.uniform(-.1,.1,(1,3)).astype(np.float32);x=a.step(action);y=b.step(action)
            assert a.diag.numpy()[0,13]==b.diag.numpy()[0,13]==0
            np.testing.assert_allclose(x[0],y[0],atol=1e-6,rtol=0);np.testing.assert_allclose(x[1],y[1],atol=1e-6,rtol=0)
        b.reset();assert np.all(b.k['state'].numpy()[:,16:]==0)
        # Actual controller, without stepping nonfinite physics: guard emits finite zero torque.
        velocity=b.data.qvel.numpy();velocity[0,0]=np.nan;b.data.qvel.assign(velocity)
        wp.launch(control,1,[b.data.qpos,b.data.qvel,b.data.sensordata,b.targets,b.command,b.active,b.k['state'],b.ids,b.k['heights'],b.k['gains'],b.k['feed'],b.k['angles'],b.k['reference'],b.k['yaw'],b.data.ctrl,b.diag,1])
        assert b.diag.numpy()[0,14]==2;np.testing.assert_array_equal(b.data.ctrl.numpy(),0)
    finally:a.close();b.close()
    result=dict(passed=True,projection_cases=n,torque_bounds_verified=True,singular_guard_verified=True,
        nonfinite_zero_control_verified=True,nonbinding_paired_seconds=1.,nonbinding_modes_equivalent=True)
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refusing to overwrite existing evidence')
    check(a.output)
