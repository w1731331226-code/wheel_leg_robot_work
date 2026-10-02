"""Shared candidate version, fixed-design isolation and one native policy-step/reset check."""
import argparse,json,sys
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
from tempfile import TemporaryDirectory
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from native.environment import NativeEnv,after,reset_rows
from native.design import current_vmc_table
from training_contract import observation_spec
from probe_current_vmc_design import current_vmc_table as compatibility_table
from probe_height_115_margin import cases
from probe_height_115_action_predict_loow import sha


def run(output):
    output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    assert current_vmc_table is compatibility_table
    archived=np.load(ROOT/'wheelleg_warp/results/current_vmc_braking_events_20261002/controller_inputs.npz',allow_pickle=False)
    scenarios=[replace(cases()[0],mass=7.,drive_difference=.05),replace(cases()[0],mass=8.,drive_difference=0.,delay_ms=20.)]
    env=NativeEnv.height115_candidate(n=2,scenario=scenarios)
    legacy=None
    try:
        for name in ('gains','feed','angles'):np.testing.assert_allclose(env.k[name].numpy(),archived[name],rtol=1e-10,atol=1e-10)
        assert env.height_safety=='physical_v1' and env.feasible_reference and env.coordinated_reference and env.radial_guard
        assert env.observation_space.shape==(38,) and env.action_space.shape==(3,)
        assert not np.array_equal(env.model.body_mass.numpy()[0],env.model.body_mass.numpy()[1])
        assert not np.array_equal(env.model.actuator_gainprm.numpy()[0],env.model.actuator_gainprm.numpy()[1])
        env.reset();q=env.data.qpos.numpy();q[1]=q[0];env.data.qpos.assign(q);env.data.qvel.zero_();env.data.sensordata.zero_()
        state=env.k['state'].numpy();state[:,0]=2.;env.k['state'].assign(state);env.targets.assign(np.tile([.1,-.1,.1],(2,1)))
        d=env.data;wp.launch(env.control_kernel,2,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
            env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
        commands=d.ctrl.numpy();np.testing.assert_array_equal(commands[0],commands[1])
        env.reset();assert np.all(env.k['state'].numpy()[:,16:19]==0)
        obs,reward,done,_=env.step(np.zeros((2,3),np.float32));assert obs.shape==(2,38) and np.isfinite(reward).all() and not done.any()
        physical=env.state.numpy();np.testing.assert_array_equal(physical[:,37],physical[:,0]);assert np.all(physical[:,0]==40)
        env.reset();assert np.all(env.state.numpy()[:,37]==0) and np.all(env.k['state'].numpy()[:,16:19]==0)
        def control_now():
            wp.launch(env.control_kernel,2,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
                env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)
        # Controller fixture only: equal physical packet, lambda0, distinct queued requests.
        original_param=env.param.numpy();p=original_param.copy();p[:,4]=0;env.param.assign(p)
        ids=env.ids.numpy();v=env.data.qvel.numpy();v[:,ids[4:10]]=100.;env.data.qvel.assign(v)
        state=env.k['state'].numpy();state[:,0]=2.;state[:,16:19]=[[.4,-.3,.2],[-.4,.3,-.2]];env.k['state'].assign(state)
        control_now();assert np.all(env.diag.numpy()[:,12]==0)
        wp.launch(after,2,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,
            env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        alias=env.obs.numpy();np.testing.assert_array_equal(alias[0,:32],alias[1,:32]);assert not np.array_equal(alias[0,32:],alias[1,32:])
        pending=env.k['state'].numpy()[:,16:19].copy()
        env.reset();state=env.k['state'].numpy();state[:,0]=2.;state[:,16:19]=pending;env.k['state'].assign(state)
        control_now();assert np.all(env.diag.numpy()[:,12]>0)
        future_command_difference=float(np.max(abs(d.ctrl.numpy()[0]-d.ctrl.numpy()[1])));assert future_command_difference>1e-4
        env.param.assign(original_param);env.reset()
        action=np.tile([.8,-.6,.2],(2,1)).astype(np.float32)
        obs,_,done,_=env.step(action);assert not done.any()
        np.testing.assert_allclose(obs[:,32:],[[.4,-.4,-.4,.4,.2,-.2]]*2,rtol=0,atol=1e-7)
        np.testing.assert_allclose(obs[:,32::2],env.k['state'].numpy()[:,16:19],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(obs[:,33::2],-obs[:,32::2])
        # Full20ms physical delay still returns the reset packet at the first20ms policy epoch.
        np.testing.assert_array_equal(obs[1,:32],env.obs0.numpy()[1]);assert env.history.shape[1]==41
        physical=env.state.numpy();np.testing.assert_array_equal(physical[:,37],physical[:,0]);assert np.all(physical[:,0]==40)
        mask=env.mask.numpy();mask[:]=[1,0];env.mask.assign(mask);wp.launch(reset_rows,2,env.reset_args)
        np.testing.assert_array_equal(env.obs.numpy()[0,32:],0);np.testing.assert_array_equal(env.obs.numpy()[1,32:],obs[1,32:])
        env.reset();assert np.all(env.obs.numpy()[:,32:]==0)
        legacy=NativeEnv.height115_candidate(n=2,scenario=scenarios,observation_contract='legacy32')
        assert legacy.observation_space.shape==(32,) and legacy.baseline_version=='height115-current-vmc-v4-damping-preserved-legacy32-candidate'
        legacy.reset();new_obs,new_reward,new_done,_=env.step(action);old_obs,old_reward,old_done,_=legacy.step(action)
        np.testing.assert_allclose(new_obs[:,:32],old_obs,rtol=0,atol=1e-5)
        np.testing.assert_allclose(new_reward,old_reward,rtol=0,atol=1e-5);np.testing.assert_array_equal(new_done,old_done)
        q_error=float(np.max(abs(env.data.qpos.numpy()-legacy.data.qpos.numpy())))
        v_error=float(np.max(abs(env.data.qvel.numpy()-legacy.data.qvel.numpy())))
        assert q_error<=2e-6 and v_error<=1e-3
        # No learning: exercise actual SB3 normalization/checkpoint shape guards and38D policy roundtrip.
        with TemporaryDirectory(prefix='height115-observation-') as temporary:
            folder=Path(temporary);old_norm=VecNormalize(legacy);new_norm=VecNormalize(env)
            old_policy=PPO('MlpPolicy',old_norm,n_steps=8,batch_size=16,n_epochs=1,device='cpu',seed=8)
            old_policy.save(folder/'legacy');old_norm.save(folder/'legacy.pkl')
            try:PPO.load(folder/'legacy',env=new_norm,device='cpu')
            except ValueError:pass
            else:raise AssertionError('32D checkpoint accepted for38D observation')
            try:VecNormalize.load(folder/'legacy.pkl',env)
            except (AssertionError,ValueError):pass
            else:raise AssertionError('32D normalization accepted for38D observation')
            policy=PPO('MlpPolicy',new_norm,n_steps=8,batch_size=16,n_epochs=1,device='cpu',seed=8)
            encoded=new_norm.reset();before,_=policy.predict(encoded,deterministic=True)
            policy.save(folder/'current');new_norm.save(folder/'current.pkl')
            restored_norm=VecNormalize.load(folder/'current.pkl',env)
            restored=PPO.load(folder/'current',env=restored_norm,device='cpu');after_action,_=restored.predict(encoded,deterministic=True)
            np.testing.assert_array_equal(before,after_action);np.testing.assert_array_equal(restored_norm.obs_rms.mean,new_norm.obs_rms.mean)
            assert policy.num_timesteps==restored.num_timesteps==0 and policy._n_updates==restored._n_updates==0
        assert observation_spec('request_state_v1',6)['dimension']==38
        result=dict(role='shared_height115_candidate_engineering_check_not_training_admission',passed=True,baseline_version=env.baseline_version,
            original_design_arrays_match=True,random_physics_does_not_change_design_or_same_input_commands=True,
            same_input_command=commands[0].tolist(),observation_dim=38,observation_spec=env.observation_spec,action_dim=3,physical_evidence_steps=physical[:,37].tolist(),reset_clears_evidence_and_action_filter=True,
            lambda_zero_fixture_old_packet_equal=True,distinct_current_request_observed=True,future_same_input_command_difference_Nm=future_command_difference,
            physical20ms_delay_preserved=True,current_own_request_not_sensor_delayed=True,masked_reset_preserved_other_world=True,
            legacy32_first_packet_and_short_physics_preserved=dict(max_qpos_difference=q_error,max_qvel_difference=v_error),
            old_checkpoint_and_normalization_rejected=True,current_policy_normalization_roundtrip_passed=True,learning_performed=False,
            limitations='Controller alias fixture is not a physical safety rollout. Short native40-step sampling/shape checks only; no whole-episode performance, robustness, full Markov-state or training-admission claim. New and all comparison policies must freeze the same request-state observation contract.',
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/design.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/training_contract.py')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS shared height115 candidate: design isolation, lambda0 alias,20ms delay/reset, legacy packet/physics and SB3 shape roundtrip; no learning')
    finally:
        env.close()
        if legacy is not None:legacy.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
