"""Actual terminal/curriculum qualification, static CUDA PPO/RMS reload;no learn."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize, VecCheckNan
from execution_history_env import ExecutionHistory, COMMAND_INDICES
from review_yaw_sector import ROOT, sha

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/execution_input_history_v1'
PARENT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_mean_matched_learning_v1/proposal.json'


class Scripted(VecEnv):
    def __init__(self):
        self._execution_intervals = None; self.tick = 0; self.actions = None
        super().__init__(2, Box(-np.inf, np.inf, (39,), dtype=np.float32), Box(-1, 1, (6,), dtype=np.float32))
    def reset(self): self.tick = 0; return np.zeros((2, 39), np.float32)
    def step_async(self, actions): self.actions = actions
    def step_wait(self):
        self.tick += 1; obs = np.zeros((2, 39), np.float32); obs[:, 6] = self.tick
        done = np.array([self.tick == 2, False]); dt = np.array([.0065 if done[0] else .02, .02])
        self._execution_intervals = dict(command_integral=np.ones((2, 6))*dt[:, None], sensor_elapsed_s=dt, terminal_worlds=done.copy())
        infos = [{}, {}]
        if done[0]: infos[0]['terminal_observation'] = obs[0].copy(); obs[0, 6] = 99
        return obs, np.array([1., 2.]), done, infos
    def close(self): pass
    def get_attr(self, name, indices=None): return [getattr(self, name, None)]*2
    def set_attr(self, name, value, indices=None): setattr(self, name, value)
    def env_method(self, *args, **kwargs): raise NotImplementedError
    def env_is_wrapped(self, *args, **kwargs): return [False, False]


def static():
    raw = Scripted(); env = ExecutionHistory(raw, raw, 'H1'); obs = env.reset()
    assert obs.shape == (2, 481) and np.all(obs[:, -1] == 1) and not obs[:, 471:-1].any()
    action = np.zeros((2, 6), np.float32); env.step(action); assert raw.actions is action
    obs, reward, done, info = env.step(action)
    assert done.tolist() == [True, False]; np.testing.assert_array_equal(reward, [1, 2])
    assert info[0]['terminal_observation'][357] == 2 and obs[0, 357] == 99
    assert info[0]['terminal_observation'][470] == np.float32(.0065/.02)
    assert not env.inputs[0].any() and env.valid[0].sum() == 1 and env.valid[1].sum() == 3
    h0, h1 = env.encode('H0'), env.encode('H1')
    same = np.delete(np.arange(481), COMMAND_INDICES)
    np.testing.assert_array_equal(h0[:, same], h1[:, same]); assert not h0[:, COMMAND_INDICES].any()
    assert h1[1, COMMAND_INDICES].any()
    saved = info[0]['terminal_observation'].copy(); env.reset()
    np.testing.assert_array_equal(info[0]['terminal_observation'], saved)
    print('PASS231 scripted partialterminal/ownership/matched54coordinateablation', flush=True)


def gpu(p):
    import train_height_comparison as base
    import light_phase_reference as light
    import execution_input_collector as collector
    from route_state import RouteState
    factory = base.raw_env
    base.raw_env = lambda rows, mode: collector.instrument(lambda a, b: light.instrument(factory, a, b), rows, mode)
    try: curriculum = base.CurriculumEnv(p, 'virtual6', 1991, 4, milestones=[4, 4000])
    finally: base.raw_env = factory
    raw = curriculum.venv; route = RouteState(curriculum); history = ExecutionHistory(route, raw, 'H1')
    norm = VecNormalize(VecCheckNan(history, raise_exception=True), **p['normalization'])
    assert raw.param.numpy()[:, 3].max()+2 < 20
    counts = np.zeros(4, int); calls = 0; physical = 0; partial = 0; terminals = []; capture = {}
    wait = route.step_wait
    def observed():
        value = wait(); capture['packet'] = value[0].copy(); capture['reward'] = value[1].copy(); capture['done'] = value[2].copy()
        capture['terminal'] = {w: value[3][w]['terminal_observation'].copy() for w in np.flatnonzero(value[2])}
        capture['counts_after_reset'] = raw.state.numpy()[:, 0].astype(np.int64)
        return value
    route.step_wait = observed; before_count = np.zeros(4, np.int64)
    history_wait = history.step_wait
    def observed_history():
        value = history_wait()
        capture['history_terminal'] = {w: value[3][w]['terminal_observation'].copy() for w in np.flatnonzero(value[2])}
        return value
    history.step_wait = observed_history
    try:
        norm.reset()
        for calls in range(1, 2501):
            obs, reward, done, infos = norm.step(np.zeros((4, 6), np.float32))
            steps = capture['counts_after_reset']-before_count
            for w in np.flatnonzero(done): steps[w] = infos[w]['physical_steps']-before_count[w]
            assert np.all((steps>0)&(steps<=40)); physical += int(steps.sum())
            before_count = capture['counts_after_reset'].copy()
            np.testing.assert_array_equal(reward, capture['reward']); np.testing.assert_array_equal(done, capture['done'])
            np.testing.assert_array_equal(history.frames[:, -1], capture['packet'])
            for w in np.flatnonzero(done):
                np.testing.assert_array_equal(history.valid[w], [0]*9+[1])
                assert not history.inputs[w].any() and raw._execution_previous_sensor_steps[w] == 0
                assert infos[w]['terminal_observation'].shape == (481,)
                np.testing.assert_array_equal(capture['history_terminal'][w][351:390], capture['terminal'][w])
                np.testing.assert_array_equal(infos[w]['terminal_observation'],
                                              norm.normalize_obs(capture['history_terminal'][w][None])[0])
                partial += int(steps[w] < 40)
                terminals.append(dict(world=int(w), actor_call=calls, physical_steps=infos[w]['physical_steps'], last_interval_steps=int(steps[w]),
                                      curriculum_stage=infos[w]['curriculum_stage'], reason=infos[w]['reason']))
            h0, h1 = history.encode('H0'), history.encode('H1'); keep=np.delete(np.arange(481),COMMAND_INDICES)
            np.testing.assert_array_equal(h0[:,keep],h1[:,keep]); assert not h0[:,COMMAND_INDICES].any()
            counts += done
            if np.all(counts>=2) and np.all(curriculum.stages==3): break
        assert calls<=2500 and np.all(counts>=2) and {r['stage'] for r in curriculum.transition_log} == {2, 3}
        assert curriculum.transition_rows == 8 and partial > 0
        saved_history = [v.copy() for v in (history.frames, history.inputs, history.elapsed, history.valid)]
        norm.save(str(OUT/'unit_history_rms.pkl'))
        loaded = VecNormalize.load(str(OUT/'unit_history_rms.pkl'), VecCheckNan(history, raise_exception=True))
        loaded.training=False; loaded.norm_reward=False
        for name in ('mean','var','count'): np.testing.assert_array_equal(getattr(norm.obs_rms,name),getattr(loaded.obs_rms,name))
        np.testing.assert_array_equal(norm.normalize_obs(history.encode()),loaded.normalize_obs(history.encode()))
        model = PPO('MlpPolicy', norm, seed=2311, device='cuda', n_steps=50, batch_size=100,
                    policy_kwargs=dict(net_arch=[64,64]))
        with torch.no_grad(): model.policy.action_net.weight.zero_(); model.policy.action_net.bias.zero_()
        prediction = model.predict(obs, deterministic=True)[0]; model.save(OUT/'unit_history_model.zip')
        agent = PPO.load(OUT/'unit_history_model.zip', env=loaded, device='cuda')
        np.testing.assert_array_equal(prediction, agent.predict(obs, deterministic=True)[0])
        for name,value in model.policy.state_dict().items(): assert torch.equal(value,agent.policy.state_dict()[name])
        assert model.policy.optimizer.state_dict()==agent.policy.optimizer.state_dict()
        assert model.num_timesteps==agent.num_timesteps==0
        for before,after in zip(saved_history,(history.frames,history.inputs,history.elapsed,history.valid)):np.testing.assert_array_equal(before,after)
        collector.preserve(raw, OUT/'lifecycle_partial_buffers.npz')
        return dict(actor_calls=calls, actual_physical_steps=physical, per_world_terminal_count=counts.tolist(),
                    partial_real_terminal_count=partial, terminals=terminals, curriculum_transitions=curriculum.transition_log,
                    history_dimensions=481, normalization_dimensions=481, model_device=str(model.device),
                    model_training_steps=0, RMS_weights_optimizer_reload_exact=True, main_training_admitted=False)
    finally: norm.close()


def run():
    torch.set_num_threads(1)
    assert not any((OUT/n).exists() for n in ('lifecycle_registration.json','round231_lifecycle.json','lifecycle_failure.json'))
    p=json.loads(PARENT.read_text()); sources={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__)}
    for name in ('execution_history_env.py','execution_input_collector.py','execution_input_features.py','route_state.py',
                 'light_phase_reference.py','train_height_comparison.py','native/environment.py'):
        sources['wheelleg_warp/'+name]=sha(ROOT/'wheelleg_warp'/name)
    (OUT/'lifecycle_registration.json').write_text(json.dumps(dict(parent_sha256=sha(PARENT), source_sha256=sources,
        worlds=4,max_actor_calls=2500,max_physics_steps=400000,training_steps=0,engineering_seed=1991,
        milestones=[4,4000],banks={s:rows[:4] for s,rows in p['training_banks']['1991'].items()},
        scope='Lifecycle qualification on prior engineering banks;zero actions,not fresh scientific test orlearning.'),indent=2)+'\n')
    try:
        static(); result=gpu(p)
        assert all(sha(ROOT/n)==v for n,v in sources.items())
        (OUT/'round231_lifecycle.json').write_text(json.dumps(dict(verified=True,round=231,result=result,
            source_sha256=sources,registration_sha256=sha(OUT/'lifecycle_registration.json'),
            limits='OneH1sameexecutionengineeringstream with H0coordinate mask;no paired trainedpolicyrollouts or contributionproof. Public sensorclock age shared;before-gaincommandnot actualtorque. CUDAmodel/RMS/emptyoptimizer reload only,continuouslearning qualification pending.'),indent=2)+'\n')
        print('PASS231 actualterminal/curriculum/history/RMScudareload;training0',result['actor_calls'],result['actual_physical_steps'],flush=True)
    except BaseException as e:
        (OUT/'lifecycle_failure.json').write_text(json.dumps(dict(error=repr(e), implicit_retry=False),indent=2)+'\n');raise


if __name__=='__main__':run()
