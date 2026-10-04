"""Registered fixed-policy, delayed-observation yaw-sector intervention.

This is a development mechanism probe, not a new algorithm or safety proof.
The filter acts on actor requests before the existing rate limiter/physics.
"""
from pathlib import Path
import argparse, json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
from train_height_comparison import raw_env, evaluate, summary, entry, protocol
from native.terrain import sample_height_terrain_115
from terrain_eval import validate_terrain_rows
from training_contract import digest
from smoke_reward_training import weight_digest
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
PILOT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/reward_pilot_v1'


def sector_request(action, packet):
    action = np.asarray(action, dtype=np.float32)
    packet = np.asarray(packet)
    if action.ndim != 2 or action.shape[1] != 3 or packet.shape != (len(action), 38):
        raise ValueError('Expected diff3 requests and request_state_v1 packets')
    if not np.isfinite(action).all() or not np.isfinite(packet).all() or np.max(abs(action)) > 1:
        raise ValueError('Nonfinite or unbounded request/packet')
    # Existing nominal gains; delayed raw gyro differs from the filtered
    # internal controller rate. This sign test is deliberately not a barrier.
    restoring = -.4 * packet[:, 2] - 2. * packet[:, 5]
    opposed = action[:, 2] * restoring < 0
    result = action.copy()
    result[opposed, 2] = 0
    return result, opposed


def self_check():
    o = np.zeros((5, 38), np.float32)
    o[:, 2] = [1, -1, 1, -1, 0]
    a = np.tile([.2, -.3, .5], (5, 1)).astype(np.float32)
    out, opposed = sector_request(a, o)
    np.testing.assert_array_equal(out[:, :2], a[:, :2])
    np.testing.assert_array_equal(out[:, 2], [0, .5, 0, .5, .5])
    mirrored = o.copy(); mirrored[:, 2] *= -1; mirrored[:, 5] *= -1
    reverse = a.copy(); reverse[:, 2] *= -1
    np.testing.assert_array_equal(sector_request(reverse, mirrored)[0][:, 2], -out[:, 2])
    assert opposed.tolist() == [True, False, True, False, False]


def collect(cases, prefix, condition):
    raw = raw_env(cases, 'diff3')
    env = VecNormalize.load(str(prefix) + '.pkl', VecCheckNan(raw, raise_exception=True))
    env.training = False; env.norm_reward = False
    agent = PPO.load(str(prefix) + '.zip', device='cuda')
    before = (agent.num_timesteps, agent._n_updates, weight_digest(agent))
    rms = (env.obs_rms.mean.copy(), env.obs_rms.var.copy(), env.obs_rms.count)
    rows = [None] * len(cases); pending = np.ones(len(cases), bool)
    counts = np.zeros((len(cases), 3), np.int64)
    clipping = np.zeros(len(cases), np.int64)
    try:
        obs = env.reset()
        ids = raw.ids.numpy()
        deadline = int(np.ceil((raw.param.numpy()[:, 3].max() + 2) / .02)) + 2
        for _ in range(deadline):
            action = agent.predict(obs, deterministic=True)[0]
            # Invert only the actual normalized/clipped actor packet. No
            # current-state, contact, ground-truth parameter or future input.
            packet = env.unnormalize_obs(obs)
            projected, opposed = sector_request(action, packet)
            counts[pending, 0] += 1
            counts[pending, 1] += opposed[pending]
            counts[pending, 2] += np.abs(action[pending, 2]) > 1e-6
            clipping[pending] += np.any(abs(obs[pending]) >= env.clip_obs, axis=1)
            if condition == 'sector': action = projected
            elif condition == 'legs_only': action[:, 2] = 0
            elif condition != 'all': raise ValueError(condition)
            obs, _, done, infos = env.step(action)
            stopped = raw.stopped_q.numpy() if done.any() else None
            for w in np.flatnonzero(done & pending):
                length = float(np.mean([sim.fk_joints(float(stopped[w, ids[2*s]]), float(stopped[w, ids[2*s+1]]))['leg_len'] for s in range(2)]))
                rows[w] = dict(**cases[w], **{k:v for k,v in infos[w].items() if k != 'terminal_observation'},
                               final_mean_fk_leg_m=length, request_sample_counts=counts[w].tolist(),
                               actor_packet_clipped_samples=int(clipping[w]))
                pending[w] = False
            if not pending.any(): break
        assert not pending.any()
        assert before == (agent.num_timesteps, agent._n_updates, weight_digest(agent))
        np.testing.assert_array_equal(rms[0], env.obs_rms.mean)
        np.testing.assert_array_equal(rms[1], env.obs_rms.var)
        assert rms[2] == env.obs_rms.count
        validate_terrain_rows(rows)
        return rows
    finally:
        env.close()


def run(out):
    self_check()
    base = protocol(BASE)
    old = json.loads((PILOT / 'proposal.json').read_text())
    contract = json.loads((PILOT / 'trainer_contract.json').read_text())
    assert all(digest(ROOT / name) == value for name, value in contract['source_sha256'].items())
    cases = [entry(i, sample_height_terrain_115(i, 3, 'development')) for i in range(4500000, 4500032)]
    models = []
    for seed in [1609, 1610, 1611]:
        prefix = PILOT / 'runs/original' / str(seed) / 'step_200000'
        record = json.loads(prefix.with_suffix('.json').read_text())
        assert digest(prefix.with_suffix('.zip')) == record['checkpoint']['checkpoint_sha256']
        assert digest(prefix.with_suffix('.pkl')) == record['checkpoint']['normalization_sha256']
        models.append(dict(seed=seed, prefix=str(prefix), checkpoint=record['checkpoint']))
    out.mkdir(parents=True, exist_ok=False)
    registration = dict(version='yaw-sector-development-v1', cases=cases, models=models,
        conditions=['all', 'sector', 'legs_only'], classical=old['fixed_classical']['B1'],
        hypothesis='Suppressing requests opposite to a delayed-observation PD direction improves fixed-policy task execution; distinguish from disabling wheel requests entirely.',
        sector='a_w=0 iff a_w*(-0.4*actor_packet_yaw-2*actor_packet_gyro_z)<0; other requests unchanged; no tunable coefficient or sweep',
        primary='Report all three per-model paired success changes sector-all and sector-legs_only; B1 ceiling and physical/design nondegradation separately.',
        next_step='Only consider sector training if all three sector-all success changes positive, no additional physical/design failures in any pair, and at least two models exceed legs_only; otherwise reject this specific filter and review evidence, not tune thresholds.',
        budget_episodes=320, policy_training_steps=0, independent_test=False,
        role='Fresh public development mechanism probe after channel audit; three original-arm final200k models fixed by protocol, no worst-seed or checkpoint selection.',
        limits='Not exact internal nominal feedback, not a 2kHz noninterference guarantee or novel PPO method. Original request slew limits, actuator projection and history remain; clamp changes closed-loop action subspace.',
        failure_policy='Preserve runtime failure and partial jobs; no silent retry/resume or replacement case.',
        old_gate_or_final_used=False, base_protocol_sha256=digest(BASE/'protocol.json'),
        source_sha256={**contract['source_sha256'], 'wheelleg_warp/probe_yaw_sector.py':digest(__file__)})
    atomic_json(out/'registration.json', registration)
    results = {}; records = []; completed = 0
    try:
        jobs = [('B1', None, None)] + [(f'{m["seed"]}_{c}', m, c) for m in models for c in registration['conditions']]
        for label, model, condition in jobs:
            atomic_json(out/'progress.json', dict(status='running', completed_episodes=completed, pending_job=label))
            assert all(digest(ROOT/name) == value for name,value in registration['source_sha256'].items())
            rows = evaluate(cases, 'diff3', candidate=registration['classical']) if model is None else collect(cases, Path(model['prefix']), condition)
            result = dict(summary=summary(rows), physical=sum(r['physical_safety_passed'] for r in rows),
                          design=sum(r['design_joint_passed'] for r in rows), runs=rows)
            path=out/(label+'.json'); atomic_json(path,result)
            results[label]={k:v for k,v in result.items() if k!='runs'}
            records.append(dict(path=path.name,sha256=digest(path),episodes=len(rows)))
            completed += len(rows)
            atomic_json(out/'completed_jobs.json',dict(completed_episodes=completed,records=records))
            print('COMPLETED', completed, label, results[label], flush=True)
        paired=[]
        for m in models:
            a,s,l=[results[f'{m["seed"]}_{c}'] for c in registration['conditions']]
            paired.append(dict(seed=m['seed'], sector_minus_all=s['summary']['success_count']-a['summary']['success_count'],
                sector_minus_legs=s['summary']['success_count']-l['summary']['success_count'],
                nondegradation=s['physical']>=a['physical'] and s['design']>=a['design']))
        gate=all(p['sector_minus_all']>0 and p['nondegradation'] for p in paired) and sum(p['sector_minus_legs']>0 for p in paired)>=2
        assert completed==320
        atomic_json(out/'result.json',dict(completed_episodes=completed,results=results,paired=paired,
            filter_training_candidate_gate=gate,training_steps=0,registration_sha256=digest(out/'registration.json'),
            inference='Finite development closed-loop intervention, not independent learned-method superiority or publication readiness.'))
        atomic_json(out/'progress.json',dict(status='complete',completed_episodes=completed))
    except BaseException as error:
        atomic_json(out/'interruption.json',dict(completed_episodes=completed,error_type=type(error).__name__,error=str(error),silently_resumable=False))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check: self_check(); print('PASS sector request sign, bounds, leg preservation and mirror symmetry')
    else:
        if args.output is None: parser.error('--output required')
        run(args.output)
