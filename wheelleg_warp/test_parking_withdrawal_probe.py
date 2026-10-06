"""Non-integrating control/packet and synthetic terminal admission checks."""
import json
import tempfile
from pathlib import Path
import numpy as np
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
import parking_withdrawal_probe as p
import nominal_mean_main as main
from nominal_reference_env import RawReferenceCache,NormalizedReferencePair
from route_state import RouteState
from test_nom_yaw_filter_probe import control_args


def query(raw,condition,command,seen_motion):
    raw.reset();raw.command.fill_(command)
    requested=np.tile([.6,-.6,.4,-.4,.2,-.2],(raw.num_envs,1)).astype(np.float32)
    raw.targets.assign(requested)
    state=raw.k['state'].numpy();state[:,0]=2.;state[:,16:22]=[.25,-.25,.15,-.15,.1,-.1];raw.k['state'].assign(state)
    seen,effective,log=raw._parking_buffers;seen.fill_(seen_motion)
    p.phase.prepare_step(raw,0,'phase_support');args=control_args(raw);args[12]=raw._role_buffers[0]
    arrays=[a for a in args if isinstance(a,wp.array)];saved=[a.numpy().copy() for a in arrays]
    wp.launch(p.phase.roles.experimental.control_physical_nominal,raw.num_envs,args,block_dim=32)
    baseline=[a.numpy().copy() for a in (raw.k['state'],raw.diag,raw.data.ctrl)]
    for a,b in zip(arrays,saved):a.assign(b)
    wp.launch(p.prepare,raw.num_envs,[0,int(condition!='original'),raw.command,raw.active,raw.state,raw.k['state'],raw.targets,seen,effective,log])
    for a,b in zip(arrays,saved):np.testing.assert_array_equal(a.numpy(),b)
    altered=list(args);altered[3]=effective
    wp.launch(p.phase.roles.experimental.control_physical_nominal,raw.num_envs,altered,block_dim=32)
    wp.launch(p.finish,raw.num_envs,[0,raw.k['state'],raw.diag,log])
    gated=condition!='original' and seen_motion==1 and command==0
    np.testing.assert_array_equal(effective.numpy(),0 if gated else requested)
    np.testing.assert_array_equal(raw.targets.numpy(),requested)
    if not gated:
        for a,b in zip((raw.k['state'],raw.diag,raw.data.ctrl),baseline):np.testing.assert_array_equal(a.numpy(),b)
    else:
        previous=state[:,16:22];expected=previous+np.clip(-previous,-.01,.01)
        np.testing.assert_array_equal(raw.k['state'].numpy()[:,16:22],expected)
        assert np.any(expected!=0)  # Withdrawal decays through the existing filter.
    np.testing.assert_array_equal(raw.data.time.numpy(),0)


def packet_models(raw,models):
    cache=RawReferenceCache(RouteState(raw,'route'));model_records=[]
    for ref in models:
        norm=VecNormalize.load(main.ROOT/ref['normalization'],VecCheckNan(cache,raise_exception=True));norm.training=False;norm.norm_reward=False
        env=NormalizedReferencePair(norm,cache);obs=env.reset()
        assert obs.shape==(raw.num_envs,78) and norm.obs_rms.mean.shape==(39,)
        model=PPO.load(main.ROOT/ref['model'],device='cuda');before=main.training.weight_digest(model)
        stats=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count,norm.ret_rms.mean.copy(),norm.ret_rms.var.copy(),norm.ret_rms.count)
        action=model.predict(obs,deterministic=True)[0]
        assert action.shape==(raw.num_envs,6) and np.isfinite(action).all()
        assert before==main.training.weight_digest(model) and model.num_timesteps==200000 and model._n_updates==400
        main.training.equal(stats,(norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count,norm.ret_rms.mean,norm.ret_rms.var,norm.ret_rms.count))
        assert main.sha(main.ROOT/ref['model'])==ref['model_sha256'] and main.sha(main.ROOT/ref['normalization'])==ref['normalization_sha256']
        model_records.append(ref['seed'])
    return model_records


def lifecycle(case):
    with tempfile.TemporaryDirectory() as tmp:
        directory=Path(tmp);raw=p.instrument([case],'virtual6','parking_request_withdrawal',directory)
        try:
            raw.reset();seen,effective,log=raw._parking_buffers
            ref,role=raw._role_buffers;table,gyro,error,_=raw._gyro_buffers
            trace,_,count,*_=raw._complete_buffers
            for step,command in enumerate((0.,.5,0.),1):
                state=raw.state.numpy();state[0,0]=step-1;raw.state.assign(state)
                raw.command.fill_(command);raw.targets.fill_(.5);p.phase.prepare_step(raw,0,'phase_support')
                args=control_args(raw);args[12]=ref
                wp.launch(p.phase.roles.noise.before_control,1,[0,args[2],args[7],raw.state,args[6],args[5],0,table.shape[1],gyro,error])
                wp.launch(p.prepare,1,[0,1,raw.command,raw.active,raw.state,raw.k['state'],raw.targets,seen,effective,log]);args[3]=effective
                wp.launch(p.phase.roles.experimental.control_physical_nominal,1,args,block_dim=32)
                wp.launch(p.finish,1,[0,raw.k['state'],raw.diag,log])
                wp.launch(p.phase.roles.finish,1,[0,raw.diag,role])
                wp.launch(p.phase.roles.noise.filtered_measurement,1,[0,raw.k['state'],gyro])
                wp.launch(p.phase.roles.noise.fresh_measurement,1,[0,raw.data.sensordata,raw.ids,raw.state,raw.active,table,0,gyro,error])
                values=np.zeros(trace.shape);values[0,0,:4]=[1,step,(step-1)*.0005,step*.0005];values[0,0,43]=command;trace.assign(values)
                count.assign(np.array([[step,step]],dtype=float));state[0,0]=step;raw.state.assign(state)
                if step==3:raw.done.fill_(5)
                result=raw.step_wait()
                if step==2:
                    saved=p.preserve(raw,directory);assert saved['complete_worlds']==[]
            assert raw._parking_frozen=={0} and not raw._parking_chunks[0] and not seen.numpy().any()
            f=directory/result[3][0]['parking_trace']['path'];digest=p.phase.rec.sha(f)
            with np.load(f,allow_pickle=False) as z:
                p.check_log(z['trace'],'parking_request_withdrawal');assert len(z['trace'])==3
            raw.done.fill_(5);raw.step_wait();assert p.phase.rec.sha(f)==digest
            seen.fill_(1);raw.reset();assert not seen.numpy().any() and not log.numpy().any() and not raw._parking_frozen
            np.testing.assert_array_equal(raw.data.time.numpy(),0)
        finally:raw.close()


def run():
    main.verify();file=main.OUT/'round210_direction_review.json';review=json.loads(file.read_text());protocol=review['diagnostic']
    assert protocol['total_first_episode_evaluation_budget']==78 and review['new_training_or_evaluations']==0
    worlds=queries=0;models=set()
    for condition in protocol['conditions']:
        for group in p.phase.broad.batches(protocol['cases']):
            raw=p.instrument([c for _,c in group],'virtual6',condition)
            try:
                assert raw.obs.shape[1]==38 and raw._parking_contract['current_command_owner_verified']
                for command,seen in ((0.,0),(.8,1),(-.8,1),(0.,1)):
                    query(raw,condition,command,seen);queries+=raw.num_envs
                models.update(packet_models(raw,protocol['models']));worlds+=raw.num_envs
                raw.active.zero_();raw._parking_buffers[0].fill_(1)
                wp.launch(p.prepare,raw.num_envs,[0,1,raw.command,raw.active,raw.state,raw.k['state'],raw.targets,*raw._parking_buffers])
                assert not raw._parking_buffers[2].numpy()[0,:,0].any()
                np.testing.assert_array_equal(raw._parking_buffers[0].numpy(),1)
            finally:raw.close()
    lifecycle(protocol['cases'][0])
    main.write(main.OUT/'round211_parking_source_unit.json',dict(verified=True,static_worlds=worlds,static_control_queries=queries,model_seeds=sorted(models),
        startup_moving_original_exact_noop=True,parking_original_noop=True,withdrawal_before_unchanged_slew_filter=True,raw_requests_preserved=True,
        command_owner_and_capture_signatures_verified=True,actor_raw38_route39_pair78_unchanged=True,model_RMS_inference_immutable=True,
        inactive_latch_untouched=True,synthetic_first_terminal_freeze_autoreset_explicit_reset_and_partial_preservation=True,
        proposal_sha256=main.sha(file),source_sha256={f'wheelleg_warp/{n}':main.sha(main.ROOT/'wheelleg_warp'/n) for n in ('parking_withdrawal_probe.py','test_parking_withdrawal_probe.py')},
        trainer_contract_sha256=main.sha(main.OUT/'trainer_contract.json'),new_training_or_evaluations=0,
        status='Source and static interface admission only;78 scientific evaluation queue not frozen or run. Synthetic terminal is not a real task result.'))
    print('PASS parking source',worlds,'staticworlds',queries,'non-integrating queries,3CUDA model/RMS inference and synthetic lifecycle;0 evaluations',flush=True)


if __name__=='__main__':run()
