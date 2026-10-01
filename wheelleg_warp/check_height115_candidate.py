"""Shared candidate version, fixed-design isolation and one native policy-step/reset check."""
import argparse,json,sys
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
from native.environment import NativeEnv
from native.design import current_vmc_table
from probe_current_vmc_design import current_vmc_table as compatibility_table
from probe_height_115_margin import cases
from probe_height_115_action_predict_loow import sha


def run(output):
    output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    assert current_vmc_table is compatibility_table
    archived=np.load(ROOT/'wheelleg_warp/results/current_vmc_braking_events_20261002/controller_inputs.npz',allow_pickle=False)
    scenarios=[replace(cases()[0],mass=7.,drive_difference=.05),replace(cases()[0],mass=8.,drive_difference=0.)]
    env=NativeEnv.height115_candidate(n=2,scenario=scenarios)
    try:
        for name in ('gains','feed','angles'):np.testing.assert_allclose(env.k[name].numpy(),archived[name],rtol=1e-10,atol=1e-10)
        assert env.height_safety=='physical_v1' and env.feasible_reference and env.coordinated_reference and env.radial_guard
        assert env.observation_space.shape==(32,) and env.action_space.shape==(3,)
        assert not np.array_equal(env.model.body_mass.numpy()[0],env.model.body_mass.numpy()[1])
        assert not np.array_equal(env.model.actuator_gainprm.numpy()[0],env.model.actuator_gainprm.numpy()[1])
        env.reset();q=env.data.qpos.numpy();q[1]=q[0];env.data.qpos.assign(q);env.data.qvel.zero_();env.data.sensordata.zero_()
        state=env.k['state'].numpy();state[:,0]=2.;env.k['state'].assign(state);env.targets.assign(np.tile([.1,-.1,.1],(2,1)))
        d=env.data;wp.launch(env.control_kernel,2,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
            env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
        commands=d.ctrl.numpy();np.testing.assert_array_equal(commands[0],commands[1])
        env.reset();assert np.all(env.k['state'].numpy()[:,16:19]==0)
        obs,reward,done,_=env.step(np.zeros((2,3),np.float32));assert obs.shape==(2,32) and np.isfinite(reward).all() and not done.any()
        physical=env.state.numpy();np.testing.assert_array_equal(physical[:,37],physical[:,0]);assert np.all(physical[:,0]==40)
        env.reset();assert np.all(env.state.numpy()[:,37]==0) and np.all(env.k['state'].numpy()[:,16:19]==0)
        result=dict(role='shared_height115_candidate_engineering_check_not_training_admission',passed=True,baseline_version=env.baseline_version,
            original_design_arrays_match=True,random_physics_does_not_change_design_or_same_input_commands=True,
            same_input_command=commands[0].tolist(),observation_dim=32,action_dim=3,physical_evidence_steps=physical[:,37].tolist(),reset_clears_evidence_and_action_filter=True,
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/design.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/controller.py')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS shared height115 candidate: archived design, hidden-parameter isolation, native40 steps and reset')
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
