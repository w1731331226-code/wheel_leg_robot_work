"""GPU checks for irreversible task termination; no policy updates or gate relaxation."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
from native.environment import after
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from terrain_eval import evaluate_terrain
from dashboard.live_env import atomic_json as write


def check(output):
    env=TerrainEnv(9,scenario=TerrainScenario(),terminate_on_attitude_failure=True)
    try:
        env.reset();q=env.data.qpos.numpy();state=env.state.numpy();param=env.param.numpy()
        for i,degrees in ((0,4.9),(1,5.1),(3,6.),(8,5.1)):
            angle=np.deg2rad(degrees)/2;q[i,3:7]=[np.cos(angle),np.sin(angle),0,0]
        state[2,8]=np.deg2rad(5.1)  # A recovered pose cannot erase a previous failure.
        state[0,8]=np.deg2rad(5.)  # The exact existing <= boundary remains admissible.
        param[3,7:11]=[1,2,np.deg2rad(3),q[3,0]]
        param[4,7]=1;state[4,8]=np.deg2rad(10.1)  # Relative target still has world safety boundary.
        param[5,11]=0;state[5,8]=np.deg2rad(5.1)  # Default evaluation continues.
        param[6,3]=.0005  # Task deadline, not an external TimeLimit.
        for i in (7,8):state[i,0]=3999;state[i,1]=0;state[i,2:4]=q[i,:2];state[i,5]=1
        env.data.qpos.assign(q);env.stopped_q.assign(q);env.state.assign(state);env.param.assign(param)
        args=[env.data.qpos,env.data.qvel,env.data.sensordata,env.data.qacc_warmstart,env.data.time,
              env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,env.residual,
              env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w]
        wp.launch(after,9,args,block_dim=32)
        np.testing.assert_array_equal(env.done.numpy(),[0,7,7,0,7,0,6,5,5])
        first=env.reward.numpy();terminal=env.done.numpy()!=0
        np.testing.assert_allclose(first[[2,7]],[-9.9995,10.0005],atol=1e-10,rtol=0)
        # Remaining substeps must neither duplicate terminal penalty nor advance a stopped state.
        for _ in range(39):wp.launch(after,9,args,block_dim=32)
        np.testing.assert_array_equal(env.reward.numpy()[terminal],first[terminal])
        states=env.state.numpy();np.testing.assert_array_equal(states[[1,2,4,6],0],1)
        assert states[7,19]==1 and states[8,19]==0
        _,_,done,infos=env.step_wait()
        assert all(not infos[i]['TimeLimit.truncated'] for i in np.flatnonzero(done))
        assert all(not infos[i]['success'] for i in (1,2,4,6,8))
        assert all(not infos[i]['terrain_passed'] for i in (1,2,4,6))
        assert infos[6]['reason']=='timeout' and infos[1]['reason']=='attitude_failure'
        reset=env.state.numpy()[done];np.testing.assert_array_equal(reset[:,8:11],0);np.testing.assert_array_equal(reset[:,21:24],0)
        np.testing.assert_array_equal(env.done.numpy()[done],0)
        np.testing.assert_array_equal(env.active.numpy()[done],1)
        try:evaluate_terrain(None,None,[],{'terminate_on_attitude_failure':True})
        except ValueError:pass
        else:raise AssertionError('Training termination leaked into full-trajectory evaluation')
        result=dict(passed=True,gpu_worlds=9,checks=['threshold_below_above','irreversible_history','relative_target','world_safety_boundary',
            'default_full_trajectory','single_terminal_penalty_in_40_substeps','task_deadline_no_bootstrap',
            'simultaneous_completion_failure','autoreset_clears_history','evaluation_rejects_early_termination'])
        write(output,result);print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refusing to overwrite existing evidence')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    check(a.output)
