"""Separate frozen24k CUDA engineering;never a main-study warmstart."""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
import train_height_comparison as base
import execution_input_collector as collector
import light_parking_request as parking
from execution_history_env import ExecutionHistory
from route_state import RouteState
from route_pilot import RouteLedger
from smoke_reward_training import weight_digest, check_agent
from dashboard.live_env import atomic_json
from review_yaw_sector import ROOT, sha

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/execution_input_history_v1'
PARENT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_mean_matched_learning_v1/proposal.json'


def make_env(p, arm):
    factory = base.raw_env
    base.raw_env = lambda rows, mode: collector.instrument(lambda a,b: parking.instrument(factory,a,b), rows, mode)
    try: curriculum = base.CurriculumEnv(p, 'virtual6', 1991, 10, milestones=[4000,8000])
    finally: base.raw_env = factory
    raw = curriculum.venv; route = RouteState(curriculum); history = ExecutionHistory(route, raw, arm)
    norm = VecNormalize(VecCheckNan(history, raise_exception=True), **p['normalization'])
    return curriculum, raw, route, history, norm


class HistoryLedger(RouteLedger):
    def __init__(self, curriculum, route, history, norm, directory):
        super().__init__(curriculum, route, directory, 2000)
        self.history, self.norm = history, norm
    def save(self):
        native = self.raw.venv
        buffers = list(native._execution_buffers)+list(native._light_parking_buffers)
        before = ([b.numpy() for b in buffers], [v.copy() for v in
                  (self.history.frames,self.history.inputs,self.history.elapsed,self.history.valid)],
                  self.norm.obs_rms.mean.copy(),self.norm.obs_rms.var.copy(),self.norm.obs_rms.count,
                  self.model._last_obs.copy())
        super().save()
        base.equal(before, ([b.numpy() for b in buffers], [self.history.frames,self.history.inputs,self.history.elapsed,self.history.valid],
                   self.norm.obs_rms.mean,self.norm.obs_rms.var,self.norm.obs_rms.count,self.model._last_obs))
        self.saved[-1]['history_481RMS_command_parking_buffers_preserved'] = True
        atomic_json(self.directory/f'step_{self.model.num_timesteps}.json', self.saved[-1])


def freeze():
    assert not (OUT/'engineering_contract.json').exists()
    for name in ('round230_direction_review.json','round231_lifecycle.json','round231_delivery.json'):
        assert json.loads((OUT/name).read_text())['verified']
    p = json.loads(PARENT.read_text())
    sources = {str(Path(m.__file__).resolve().relative_to(ROOT)): sha(m.__file__)
               for m in list(sys.modules.values()) if getattr(m, '__file__', None)
               and Path(m.__file__).resolve().is_relative_to(ROOT) and Path(m.__file__).is_file()
               and not Path(m.__file__).resolve().is_relative_to(ROOT/'.venv') and str(m.__file__).endswith('.py')}
    from ppo_env import XML
    sources[str(Path(XML).relative_to(ROOT))] = sha(XML)
    atomic_json(OUT/'engineering_contract.json', dict(arms=['H0','H1'], samples_each=12000,total_samples=24000,
       worlds=10,engineering_seed=1991,source_sha256=sources,parent_sha256=sha(PARENT),
       lifecycle_sha256=sha(OUT/'round231_lifecycle.json'),direction_sha256=sha(OUT/'round230_direction_review.json'),
       ppo=p['ppo'],normalization=p['normalization'],initial_log_std=p['initial_log_std'],
       milestones=[4000,8000],checkpoints=list(range(2000,12001,2000)),device='cuda',
       parking='Same qualified seen_nonzero/publiccmd0 withdrawal;originalslew unchanged.',
       rule='No scientific evaluation/promotion/warmstart;onecontinuouslearn perarm,stopqueueonerror;preservepartial,never implicitrestart.',
       main_training_admitted=False,formal5_admitted=False))
    print('FROZEN engineering24k only;mainnotadmitted',flush=True)


def verify():
    c=json.loads((OUT/'engineering_contract.json').read_text())
    assert sha(PARENT)==c['parent_sha256'] and all(sha(ROOT/n)==v for n,v in c['source_sha256'].items())
    assert c['total_samples']==24000 and not c['main_training_admitted']
    return c,json.loads(PARENT.read_text())


def run():
    torch.set_num_threads(1); contract,p=verify(); start_all=time.perf_counter(); reports={}
    assert not (OUT/'engineering_completion.json').exists()
    for arm in contract['arms']:
        verify(); directory=OUT/'engineering'/arm; directory.mkdir(parents=True,exist_ok=False)
        curriculum,raw,route,history,norm=make_env(p,arm)
        kwargs=dict(p['ppo']); policy=dict(kwargs.pop('policy_kwargs'))
        model=PPO('MlpPolicy',norm,seed=1991,device='cuda',policy_kwargs=policy,**kwargs)
        with torch.no_grad():
            model.policy.action_net.weight.zero_(); model.policy.action_net.bias.zero_()
            model.policy.log_std.copy_(torch.tensor(p['initial_log_std'],device='cuda'))
        initial=weight_digest(model); h=hashlib.sha256()
        for b in (raw.q0,raw.param,raw.k['gains'],raw.k['reference']):h.update(b.numpy().tobytes())
        if reports:
            peer=reports['H0'];assert initial==peer['initial_weights'] and h.hexdigest()==peer['initial_world']
        atomic_json(directory/'initialization.json',dict(arm=arm,initial_weights=initial,initial_world=h.hexdigest(),
            observation_dimensions=481,mean_zero=True,only_information_mask_differs=True,engineering_only=True))
        callback=HistoryLedger(curriculum,route,history,norm,directory)
        try:
            start=time.perf_counter();model.learn(total_timesteps=12000,callback=callback);callback.save()
            wall=time.perf_counter()-start
            assert model.num_timesteps==12000 and model._n_updates==240 and weight_digest(model)!=initial
            check_agent(model,480)
            assert [r['policy_steps'] for r in callback.saved]==contract['checkpoints']
            assert {r['stage'] for r in curriculum.transition_log}=={2,3} and np.any(curriculum.stages==3)
            assert callback.rows and raw._light_parking_stats['withdraw_samples']>0
            assert norm.obs_rms.mean.shape==(481,) and np.isfinite(norm.obs_rms.var).all()
            np.testing.assert_allclose(norm.obs_rms.count,12010.0001,atol=1e-7,rtol=0)
            last=directory/'step_12000';loaded=PPO.load(last.with_suffix('.zip'),device='cuda')
            base.equal(model.policy.state_dict(),loaded.policy.state_dict());base.equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict());check_agent(loaded,480)
            with last.with_suffix('.pkl').open('rb') as f: restored=pickle.load(f)
            for rms in ('obs_rms','ret_rms'):
                for field in ('mean','var','count'):base.equal(getattr(getattr(norm,rms),field),getattr(getattr(restored,rms),field))
            obs=model._last_obs.copy();base.equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
            collector.preserve(raw,directory/'final_command_buffers.npz')
            np.savez_compressed(directory/'final_history.npz',frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)
            atomic_json(directory/'episodes.json',dict(episodes=callback.rows,transitions=curriculum.transition_log))
            report=dict(verified=True,arm=arm,policy_samples=12000,epochs=240,adam_updates=480,
                initial_weights=initial,initial_world=h.hexdigest(),completed_episodes=len(callback.rows),
                checkpoints=contract['checkpoints'],curriculum_transitions=curriculum.transition_log,
                parking=raw._light_parking_stats.copy(),RMS_weights_nonemptyAdam_prediction_reload_exact=True,
                learn_and_checkpoint_wall_s=wall,engineering_only=True,main_training_admitted=False)
            atomic_json(directory/'verification.json',report);reports[arm]=report
            print('PASS232',arm,'12k/240epochs/480Adam',wall,'seconds',flush=True)
        except BaseException as e:
            model.save(directory/'interrupted_model.zip');norm.save(str(directory/'interrupted_rms.pkl'))
            collector.preserve(raw,directory/'interrupted_command_buffers.npz')
            np.savez_compressed(directory/'interrupted_history.npz',frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)
            atomic_json(directory/'interruption.json',dict(error=repr(e),sampled=model.num_timesteps,
                confirmed_trained=callback.trained,checkpoints=callback.saved,implicit_resume=False));raise
        finally:norm.close()
    verify()
    atomic_json(OUT/'engineering_completion.json',dict(verified=True,reports=reports,total_policy_samples=24000,
        queue_wall_s=time.perf_counter()-start_all,contract_sha256=sha(OUT/'engineering_contract.json'),
        main_training_admitted=False,formal5_admitted=False,scientific_evaluations=0))
    print('COMPLETE232 engineering24k;mainstillnotadmitted',flush=True)


if __name__=='__main__': {'freeze':freeze,'run':run}[sys.argv[1]]()
