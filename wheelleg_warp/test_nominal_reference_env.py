"""Paired packet/RMS lifecycle and lightweight static controller admission."""
import gc
import json
import tempfile
from pathlib import Path
import numpy as np
import warp as wp
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
import train_height_comparison as base
import light_phase_reference as light
from nominal_reference_env import RawReferenceCache, NormalizedReferencePair
from nominal_packet_reference import reference, unit as reference_unit
from route_state import RouteState
from test_route_state import ScriptedEnv
from test_nom_yaw_filter_probe import control_args
from dashboard.live_env import atomic_json
from review_yaw_sector import ROOT, sha

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_mean_matched_learning_v1'


class HeightSwitch(ScriptedEnv):
    def reset(self):
        self.packet[:, 11] = [-.185, -.14]
        self.packet[:, 6] = self.packet[:, 9] = .8
        self.packet[:, 26:38] = np.arange(12)[None]/40
        return super().reset()

    def step_wait(self):
        obs, reward, done, infos = super().step_wait()
        if done[0]:
            obs[0, 11] = self.packet[0, 11] = .08
        return obs, reward, done, infos


def pair_env(raw, normalization):
    cache = RawReferenceCache(RouteState(raw, 'route'))
    norm = VecNormalize(VecCheckNan(cache, raise_exception=True), **normalization)
    return cache, norm, NormalizedReferencePair(norm, cache)


def scripted_check(p):
    raw = HeightSwitch(); cache, norm, env = pair_env(raw, p['normalization'])
    expected = RunningMeanStd(shape=(39,))
    try:
        obs = env.reset(); expected.update(norm.get_original_obs())
        for _ in range(3):
            np.testing.assert_array_equal(expected.mean, norm.obs_rms.mean)
            np.testing.assert_array_equal(expected.var, norm.obs_rms.var)
            assert expected.count == norm.obs_rms.count and norm.obs_rms.mean.shape == (39,)
            np.testing.assert_array_equal(obs[:, :39], norm.normalize_obs(norm.get_original_obs()))
            np.testing.assert_array_equal(obs[:, 39:], norm.normalize_obs(cache.current_reference))
            np.testing.assert_array_equal(obs[:, 26:38], obs[:, 65:77])
            actions = np.zeros((2, 3), np.float32)
            obs, reward, done, infos = env.step(actions); expected.update(norm.get_original_obs())
            np.testing.assert_array_equal(reward, [1., 2.]); assert raw.actions is actions
            if done[0]:
                assert infos[0]['terminal_observation'].shape == (78,)
                assert cache.terminal_references[0][11] == np.float32(-.185)
                assert cache.current_reference[0, 11] == np.float32(.08)
                np.testing.assert_array_equal(infos[0]['terminal_observation'][39:],
                    norm.normalize_obs(cache.terminal_references[0][None])[0])
        # Clipping destroys rawheight information; the cached raw reference must survive it.
        norm.training = False; norm.obs_rms.mean[11] = 10.; norm.obs_rms.var[11] = 1e-16
        obs = env.reset(); assert np.all(obs[:, 11] == -10.)
        np.testing.assert_array_equal(obs[:, 50], obs[:, 11])
        assert np.all(cache.current_reference[:, 22] >= .115)
        before = (norm.obs_rms.mean.copy(), norm.obs_rms.var.copy(), norm.obs_rms.count)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'normalization.pkl'; norm.save(file)
            other_raw = HeightSwitch(); other_cache = RawReferenceCache(RouteState(other_raw, 'route'))
            other_norm = VecNormalize.load(file, VecCheckNan(other_cache, raise_exception=True))
            other = NormalizedReferencePair(other_norm, other_cache)
            try:
                np.testing.assert_array_equal(other.reset(), obs)
                np.testing.assert_array_equal(other_norm.obs_rms.mean, before[0])
                np.testing.assert_array_equal(other_norm.obs_rms.var, before[1]); assert other_norm.obs_rms.count == before[2]
            finally: other.close()
        return dict(actual_only_39_RMS_updates=True, clipped_raw_reference_preserved=True,
                    asynchronous_terminal_old_and_reset_new_reference=True, normalization_reload_exact=True)
    finally: env.close()


def control_identity(raw):
    comparisons = 0
    for degrees in (-5., 0., 5.):
        for command in (-.8, 0., .8):
            raw.reset(); q = raw.data.qpos.numpy()
            q[:, 3] = np.cos(np.deg2rad(degrees)/2); q[:, 4] = np.sin(np.deg2rad(degrees)/2)
            raw.data.qpos.assign(q); raw.command.fill_(command)
            raw.targets.assign(np.tile([.1, -.1, .05, -.05, .2, -.2], (raw.num_envs, 1)))
            args = control_args(raw)
            wp.launch(light.phase.roles.original.control_physical_nominal, raw.num_envs, args, block_dim=32)
            memory = raw.k['state'].numpy(); memory[:, 0] = 2.; memory[:, 3:9] = 0.; raw.k['state'].assign(memory)
            arrays = [a for a in args if isinstance(a, wp.array)]; saved = [a.numpy().copy() for a in arrays]
            light.prepare_step(raw)
            for a, b in zip(arrays, saved): np.testing.assert_array_equal(a.numpy(), b)
            ref = raw._light_phase_buffers[0]; altered = list(args); altered[12] = ref
            wp.launch(light.phase.roles.experimental.control_physical_nominal, raw.num_envs, altered, block_dim=32)
            outputs = [a.numpy().copy() for a in (raw.k['state'], raw.diag, raw.data.ctrl)]
            for a, b in zip(arrays, saved): a.assign(b)
            if command == 0:
                wp.launch(light.phase.roles.original.control_physical_nominal, raw.num_envs, args, block_dim=32)
            else:
                # Independent CPU preparation of the frozen optional anchor branch.
                expected = np.zeros((raw.num_envs, 16)); expected[:, :10] = raw.k['reference'].numpy(); expected[:, 15] = .115
                replacement = wp.array(expected, dtype=light.phase.D); altered[12] = replacement
                wp.launch(light.phase.roles.experimental.control_physical_nominal, raw.num_envs, altered, block_dim=32)
            for a, b in zip(outputs, (raw.k['state'].numpy(), raw.diag.numpy(), raw.data.ctrl.numpy())):
                np.testing.assert_array_equal(a, b)
            comparisons += raw.num_envs
    raw.reset()
    assert all(not a.numpy().any() for a in raw._light_phase_buffers)
    assert not raw.data.time.numpy().any() and not raw.state.numpy()[:, 0].any()
    assert '_complete_buffers' not in raw.__dict__ and '_force_buffers' not in raw.__dict__
    return comparisons


def synthetic_curriculum(p):
    current = light.curriculum(p, 'virtual6', p['seeds'][0], 10, milestones=[10, 20])
    raw = current.venv; cache, norm, env = pair_env(current, p['normalization'])
    try:
        env.reset(); transitions = []
        for stage in (2, 3):
            old_height = raw.obs.numpy()[3, 11]
            state = raw.state.numpy(); state[3, 0] = 1; state[3, 1] = 0.; raw.state.assign(state)
            done = raw.done.numpy(); done[3] = 5; raw.done.assign(done)
            obs, _, ended, infos = env.step_wait()
            assert ended[3] and current.stages[3] == stage
            assert cache.terminal_references[3][11] == old_height
            assert cache.current_reference[3, 11] == raw.obs.numpy()[3, 11]
            light.prepare_step(raw)
            np.testing.assert_array_equal(raw._light_phase_buffers[0].numpy()[:, 2], raw.k['reference'].numpy()[:, 2])
            np.testing.assert_array_equal(obs[:, 39:], norm.normalize_obs(cache.current_reference))
            assert infos[3]['terminal_observation'].shape == (78,)
            transitions.append(dict(stage=stage, old_height_field=float(old_height), new_height_field=float(raw.obs.numpy()[3, 11])))
        assert all(row['actual_episode_end'] for row in current.transition_log)
        return transitions
    finally: env.close()


def check():
    p = json.loads((OUT/'proposal.json').read_text()); reference_unit()
    scripted = scripted_check(p); owners = []; comparisons = 0
    defaults = (wp.launch, light.environment.control_physical_nominal, base.raw_env)
    for seed, stages in p['training_banks'].items():
        for stage, cases in stages.items():
            raw = light.instrument(base.raw_env, cases, 'virtual6')
            try:
                assert defaults == (wp.launch, light.environment.control_physical_nominal, base.raw_env)
                assert raw.num_envs == 100 and raw.action_space.shape == (6,) and raw.observation_space.shape == (38,)
                comparisons += control_identity(raw)
                owners.append(dict(seed=seed, stage=stage, **raw._light_phase_topology))
            finally: raw.close()
            del raw; gc.collect()
            print('CHECKED100', seed, stage, flush=True)
    transitions = synthetic_curriculum(p)
    assert comparisons == 8100
    atomic_json(OUT/'round201_environment_unit.json', dict(verified=True, source_stage_only=True,
        scripted_pair_lifecycle=scripted, registered_static_worlds=900, static_control_comparisons=comparisons,
        batch_owners=owners, synthetic_curriculum_transitions=transitions,
        lightweight_buffer_bytes_per100=owners[0]['small_owned_buffer_bytes'], no_dense_training_recorder=True,
        original_inputs_and_global_defaults_preserved=True, conditional_reference_refreshed_after_stage_reset=True,
        new_task_evaluations=0, training_updates=0,
        limits='Static controlled queries and synthetic reset/curriculum events; no recorded task trajectories. '
               'Actual continuous lifecycle and optimizer updates are reserved for203 engineering.201 is not policy/learning admission.'))
    print('PASS paired39RMS/78terminal/reload/clipping/900staticworlds/8100controlcomparisons;0 learning/task eval', flush=True)


if __name__ == '__main__':
    check()
