"""Native history-sensitive design success, physical distinction and reset."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import warp as wp
from native.environment import NativeEnv,collect_physical,after,reset_rows
from probe_height_115_margin import cases


def run(output):
    assert not output.exists();output.mkdir(parents=True)
    env=NativeEnv.height115_candidate(n=2,scenario=cases()[4:6],residual_scale=0)
    try:
        assert env.design_joint_gate and env.state.shape[1]==39 and env.observation_space.shape==(38,)
        env.reset();q=env.data.qpos.numpy();s=env.state.numpy()
        # Same valid terminal pose; only one history contains a design breach.
        s[:,0]=3999;s[:,1]=0;s[:,2:4]=q[:,:2];s[:,5]=1;s[:,37]=3999
        s[:,38]=[.1,-.001];env.state.assign(s)
        param=env.param.numpy();param[:,5:8]=0;env.param.assign(param)
        env.data.actuator_force.zero_();env.data.ctrl.zero_();env.command.zero_()
        wp.launch(collect_physical,2,env.physical_args)
        d=env.data
        wp.launch(after,2,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,
            env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,env.history,
            env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        terminal=env.state.numpy();reward=env.reward.numpy()
        assert terminal[:,19].tolist()==[1.,0.] and reward[0]-reward[1]==20.
        _,_,done,infos=env.step_wait()
        assert done.all() and all(r['physical_safety_passed'] for r in infos)
        assert [r['design_joint_passed'] for r in infos]==[True,False]
        assert [r['success'] for r in infos]==[True,False]
        assert all(r['design_joint_contract']=='active-1p4-v1' for r in infos)
        assert np.all(env.state.numpy()[:,38]>1)
        # Actual collector uses active joints and retains the worst substep.
        env.reset();q=env.data.qpos.numpy();q[0,env.ids.numpy()[0]]=np.float32(1.401)
        env.data.qpos.assign(q);wp.launch(collect_physical,2,env.physical_args)
        observed=env.state.numpy()[:,38]
        assert abs(observed[0]-(1.4-float(np.float32(1.401))))<1e-12 and observed[1]>0
        env.data.qpos.assign(env.q0);wp.launch(collect_physical,2,env.physical_args)
        np.testing.assert_array_equal(env.state.numpy()[:,38],observed)
        env.mask.assign(np.array([1,0],np.int32));wp.launch(reset_rows,2,env.reset_args)
        assert env.state.numpy()[0,38]>1 and env.state.numpy()[1,38]==observed[1]
        env.reset();env.step(np.zeros((2,3),np.float32));state=env.state.numpy()
        assert np.all(state[:,0]==40) and np.all(state[:,37]==40) and np.all(state[:,38]>0)
        result=dict(passed=True,baseline_version=env.baseline_version,terminal_success=[True,False],
            physical_pass=[True,True],past_design_violation_rejected=True,terminal_reward_difference=20.,
            collector_peak_history_preserved=True,masked_and_auto_reset_passed=True,native_physical_steps=40,
            observation_dimension=38,learning=False,
            source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__),Path(__file__).parent/'native/environment.py')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PASS native full design contract: past violation rejects success/+10, physical1.5 remains distinct, full/masked reset and40 native steps')
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
