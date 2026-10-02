"""Shared Nom/Actor separation, original correction domain, limits, reward and reset."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import warp as wp
from native.environment import NativeEnv,after,reset_rows
from native.controller import D
from probe_height_115_margin import cases
from probe_braking_feedback import execute_extra,execution_graph,Forecaster
from training_contract import digest


def launch(env):
    d=env.data
    wp.launch(env.control_kernel,env.num_envs,[d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,env.k['state'],env.ids,
        env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],d.ctrl,env.diag,0,0]+env.control_extra,block_dim=32)


def reject(call):
    try:call()
    except ValueError:return
    raise AssertionError('Invalid correction/interface accepted')


def run(output):
    assert not output.exists();output.mkdir(parents=True)
    scenes=[replace(cases()[0],mass=7.,drive_difference=.05),replace(cases()[0],mass=8.,drive_difference=0.)]
    old=NativeEnv.height115_candidate(n=2,scenario=scenes,residual_scale=0)
    new=NativeEnv.height115_candidate(n=2,scenario=scenes,nominal_correction=True)
    correction=np.tile([.02,-.01,.02,-.01,.02,.02],(2,1));extra=wp.array(correction,dtype=D)
    max_error=0.;actor_peak=0.
    try:
        assert new.observation_space.shape==(38,) and new.action_space.shape==(3,)
        reject(lambda:old.set_nominal_correction(correction))
        reject(lambda:execution_graph(new,extra));reject(lambda:Forecaster(new))
        for bad in (np.full((2,6),1.01),np.full((2,6),np.nan),np.zeros((1,6))):reject(lambda:new.set_nominal_correction(bad))
        reject(lambda:new.set_nominal_correction(correction))  # No physical interval accrued.
        for rpm in (0.,600.,800.,-600.):
            for env in (old,new):
                env.reset();v=env.data.qvel.numpy();v[:,env.ids.numpy()[8:10]]=rpm*np.pi/30;env.data.qvel.assign(v)
                env.data.sensordata.zero_();state=env.k['state'].numpy();state[:,0]=2.;env.k['state'].assign(state)
                s=env.state.numpy();s[:,0]=10;env.state.assign(s);env.command.zero_()
            new.set_nominal_correction(correction);reject(lambda:new.set_nominal_correction(correction*0))
            launch(old);wp.launch(execute_extra,2,[old.data.qvel,old.ids,old.control_extra[0],extra,old.active,old.data.ctrl,old.diag])
            launch(new)
            difference=float(np.max(abs(old.data.ctrl.numpy()-new.data.ctrl.numpy())));max_error=max(max_error,difference);assert difference==0
            assert np.all(new.diag.numpy()[:,6:12]==0)
            # Both worlds share q/v/state; hidden model calibration must not enter allocation.
            np.testing.assert_array_equal(new.data.ctrl.numpy()[0],new.data.ctrl.numpy()[1])
            state=new.k['state'].numpy();state[:,16:19]=[[.2,-.1,.2]]*2;new.k['state'].assign(state);launch(new)
            diag=new.diag.numpy();control=new.data.ctrl.numpy()
            np.testing.assert_allclose(control,diag[:,:6]+diag[:,6:12],rtol=0,atol=3e-7)
            actor_peak=max(actor_peak,float(abs(diag[:,6:12]).max()))
            new.reset();assert np.all(new.nominal_correction.numpy()==0)
        # Reward fixture: same physical state and accepted total, classic amount is not Actor penalty.
        for env in (old,new):
            env.reset();s=env.state.numpy();s[:,0]=10;env.state.assign(s);state=env.k['state'].numpy();state[:,0]=2.;env.k['state'].assign(state)
        new.set_nominal_correction(correction);launch(old)
        wp.launch(execute_extra,2,[old.data.qvel,old.ids,old.control_extra[0],extra,old.active,old.data.ctrl,old.diag]);launch(new)
        np.testing.assert_array_equal(old.data.ctrl.numpy(),new.data.ctrl.numpy())
        for env in (old,new):
            d=env.data;wp.launch(after,2,[d.qpos,d.qvel,d.sensordata,d.qacc_warmstart,d.time,env.contact_flags,env.ids,env.param,env.command,env.state,env.k['state'],env.diag,
                env.residual,env.active,env.done,env.reward,env.obs,env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
        reward_gain=new.reward.numpy()-old.reward.numpy();assert np.all(reward_gain>0)
        np.testing.assert_array_equal(new.obs.numpy()[:,26:32],0.)
        assert np.any(old.obs.numpy()[:,26:32]!=0)
        # Captured native40-step path, masked reset and elapsed-time budget.
        new.reset();new.step(np.zeros((2,3),np.float32));new.set_nominal_correction(correction)
        _,_,done,_=new.step(np.zeros((2,3),np.float32));assert not done.any()
        np.testing.assert_array_equal(new.state.numpy()[:,37],new.state.numpy()[:,0]);assert np.all(new.state.numpy()[:,0]==80)
        np.testing.assert_array_equal(new.diag.numpy()[:,6:12],0)
        mask=new.mask.numpy();mask[:]=[1,0];new.mask.assign(mask);wp.launch(reset_rows,2,new.reset_args)
        np.testing.assert_array_equal(new.nominal_correction.numpy()[0],0);np.testing.assert_array_equal(new.nominal_correction.numpy()[1],correction[1])
        new.reset();assert np.all(new.nominal_correction.numpy()==0)
        # A bypassed invalid GPU buffer still cannot issue NaN/out-of-domain commands.
        new.nominal_correction.assign(np.full((2,6),np.nan));launch(new);assert np.all(new.data.ctrl.numpy()==0) and np.all(new.diag.numpy()[:,14]==2)
        result=dict(role='shared_nominal_actor_boundary_check_not_admission',passed=True,zero_actor_external_command_max_difference=max_error,
            actor_actual_contribution_peak_Nm=actor_peak,reward_fixture_classic_penalty_removed=reward_gain.tolist(),
            original_request_box_and_elapsed_slew_guarded=True,invalid_gpu_buffer_fails_closed=True,hidden_parameters_isolated=True,native_physical_steps=80,
            masked_reset_clears_only_selected_correction=True,old_external_graph_and_forecaster_rejected=True,default_factory_unchanged=True,
            limitations='Controller/reward fixtures and short native path only; no joint invariance, full task, robust/all-height, learning or default-promotion claim.',
            source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):digest(p) for p in (Path(__file__),Path(__file__).parent/'native/controller.py',Path(__file__).parent/'native/environment.py',Path(__file__).parent/'probe_braking_feedback.py')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS shared Nom boundary: exact zero-Actor external commands, Actor-only residual/reward, limits/reset and old-predictor rejection')
    finally:old.close();new.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
