"""GPU task-contract regressions: real wheel centers, whole-terrain exit, delay/reset."""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
import warp as wp
from native.environment import after,wheel_center,D
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from terrain_eval import validate_terrain_rows,summarize_terrain
from dashboard.live_env import atomic_json as write


@wp.kernel
def centers(q:wp.array2d[float],ids:wp.array[int],offsets:wp.array3d[wp.vec3d],out:wp.array3d[D]):
    w,side=wp.tid();point=wheel_center(q,ids,offsets,w,side)
    for k in range(3):out[w,side,k]=point[k]


def check(output):
    mixed=TerrainScenario(terrain='mixed',roughness_m=.003,speed=.7,terrain_seed=81500)
    cases=[replace(mixed,speed=direction*.7,delay_ms=delay,terrain_seed=81500+i) for i,(direction,delay) in enumerate(((1,0),(1,10),(-1,15),(-1,20)))]
    cases += [TerrainScenario(terrain='single_side_ramp',grade_deg=sign*3,speed=.7,terrain_seed=81510+i) for i,sign in enumerate((1,-1))]
    cases += [TerrainScenario(terrain_seed=81520),mixed]
    env=TerrainEnv(len(cases),scenario=cases);n=env.num_envs
    try:
        env.reset();assert env.history.shape[1]==41
        np.testing.assert_allclose(env.required_terrain_end[:4],1.021,atol=2e-7)
        assert all(env.task_goals[i]>s.center+env.required_terrain_end[i] for i,s in enumerate(cases) if s.terrain!='legacy')
        args=[env.data.qpos,env.data.qvel,env.data.sensordata,env.data.qacc_warmstart,env.data.time,env.contact_flags,env.ids,env.param,env.command,
              env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets]
        # Actual alpha/passive chain against CPU kinematics, not ideal closed-chain FK.
        out=wp.zeros((n,2,3),dtype=D);rng=np.random.default_rng(81500);peak=0.;data=mujoco.MjData(env.cpu)
        for _ in range(5):
            q=env.data.qpos.numpy();q[:,:3]=rng.normal(size=(n,3));rotation=rng.normal(size=(n,4));rotation/=np.linalg.norm(rotation,axis=1,keepdims=True);q[:,3:7]=rotation
            for side in (0,1):
                q[:,env.ids.numpy()[2*side]]=rng.uniform(-.7,.7,n);q[:,env.ids.numpy()[17+side]]=rng.uniform(-.7,.7,n)
            env.data.qpos.assign(q);wp.launch(centers,(n,2),[env.data.qpos,env.ids,env.wheel_offsets,out]);actual=out.numpy()
            for i in range(n):
                data.qpos[:]=q[i];mujoco.mj_kinematics(env.cpu,data)
                expected=data.geom_xpos[env.wheel_geom_ids];peak=max(peak,float(np.max(abs(actual[i]-expected))))
                np.testing.assert_allclose(actual[i],expected,atol=3e-7,rtol=0)
        env.reset();q=env.data.qpos.numpy();state=env.state.numpy();flags=env.contact_flags.numpy()
        for i,s in enumerate(cases):
            if s.terrain!='legacy':q[i,0]=np.sign(s.speed)*(s.center+env.required_terrain_end[i]+.025)-.075
            state[i,0]=3999;state[i,1]=0;state[i,2:4]=q[i,:2];state[i,5]=1
            flags[i,1]=env.required_terrain_contact_masks[i]
        q[0,env.ids.numpy()[17]]+=.2;q[2,env.ids.numpy()[17]]-=.2  # One wheel before exit, both directions.
        flags[1,1]=0;flags[5,1]=8  # Missing mixed contact; wrong side on a single-side ramp.
        env.data.qpos.assign(q);env.stopped_q.assign(q);env.state.assign(state);env.contact_flags.assign(flags)
        wp.launch(after,n,args,block_dim=32);terminal=env.state.numpy().copy();rewards=env.reward.numpy().copy()
        expected=[False,False,False,True,True,False,True,True]
        np.testing.assert_array_equal(terminal[:,19].astype(bool),expected)
        np.testing.assert_allclose(rewards,np.where(expected,10.,-10.)+.0005,atol=1e-9,rtol=0)
        # Late geometry changes cannot corrupt evidence frozen at the terminal physical step.
        env.data.geom_xpos.fill_(wp.vec3(99.,99.,99.))
        for _ in range(4):wp.launch(after,n,args,block_dim=32)
        np.testing.assert_array_equal(env.state.numpy()[:,24:28],terminal[:,24:28])
        _,_,done,infos=env.step_wait();assert done.all();validate_terrain_rows(infos)
        for i,row in enumerate(infos):np.testing.assert_array_equal(row['wheel_progress_m'],terminal[i,24:26])
        assert np.all(env.state.numpy()[:,24:28]==0)
        # A body reaching an old/short goal cannot trigger parking while wheels remain before exit.
        env.reset();q=env.data.qpos.numpy();param=env.param.numpy();q[0,0]=cases[0].center+.7;param[0,2]=q[0,0]
        env.data.qpos.assign(q);env.param.assign(param);env.contact_flags.fill_(0);wp.launch(after,n,args,block_dim=32)
        assert env.state.numpy()[0,1]<0
        # Real ring writes/reads through the same production after() kernel, including startup/wrap/reset.
        env.reset();env.contact_flags.fill_(0);gyro=env.ids.numpy()[10];sensor=env.data.sensordata.numpy()
        checks=[]
        for step in range(1,81):
            sensor[:,gyro]=step;env.data.sensordata.assign(sensor);wp.launch(after,n,args,block_dim=32)
            expected_obs=np.array([max(0,step-round(s.delay_ms*2)) for s in cases])
            np.testing.assert_array_equal(env.obs.numpy()[:,3],expected_obs)
            if step in (1,20,40,41,80):checks.append(dict(step=step,observed=env.obs.numpy()[:4,3].tolist()))
        env.reset();sensor[:,gyro]=999;env.data.sensordata.assign(sensor);wp.launch(after,n,args,block_dim=32)
        np.testing.assert_array_equal(env.obs.numpy()[1:4,3],0)
        result=dict(passed=True,worlds=n,wheel_center_cpu_max_error=peak,mixed_end=env.required_terrain_end[:4],
            terminal_reward_and_evidence_consistent=True,terminal_evidence_frozen=True,parking_requires_exit=True,delay_samples=checks,reset_clears_history=True)
        output.parent.mkdir(parents=True,exist_ok=True);write(output,result);print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refusing to overwrite existing evidence')
    check(a.output)
