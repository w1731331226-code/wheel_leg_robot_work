"""One registered10 first-episode engineering pair; no learned or tuned policy."""
import json
import time
import mujoco
import numpy as np
import execution_history_evaluation as original
import joint_reference_adapter as adapter
from route_state import RouteState
from execution_history_env import ExecutionHistory
from train_height_comparison import raw_env, summary
import complete_contact_witness as witness
from native.controller import sim
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/zero_episode_pair_v1'


def candidate(cases, directory):
    raw = adapter.instrument(raw_env, cases, 'virtual6', directory)
    history = ExecutionHistory(RouteState(raw), raw, 'H1')
    return raw, adapter.JointReferenceActions(history, raw, 'M3')


def evaluate(cases, arm, directory, graph_budget):
    if arm == 'old_B0':
        raw, _, _, _, env = original.make_env(cases, 'B0', None, directory)
        dimension = 6
    else:
        raw, env = candidate(cases, directory); dimension = 3
    rows = [None]*len(cases); actors = [[] for _ in cases]
    try:
        obs = env.reset(); assert obs.shape == (5, 481)
        initial = dict(q=raw.data.qpos.numpy().copy(), v=raw.data.qvel.numpy().copy(),
            memory=raw.k['state'].numpy().copy(), param=raw.param.numpy().copy(), reference=raw.k['reference'].numpy().copy())
        np.savez_compressed(directory/'initial.npz', **initial)
        deadline = int(np.ceil((raw.param.numpy()[:, 3].max()+2)/.02))+2
        ids = raw.ids.numpy(); calls = 0
        for _ in range(deadline):
            assert (calls+1)*40*len(cases) <= graph_budget, 'Registered graph-step budget exceeded'
            for w in range(5):
                if rows[w] is None:
                    actors[w].append(np.r_[obs[w], np.zeros(dimension)].astype(np.float32))
            obs, _, done, infos = env.step(np.zeros((5, dimension), np.float32)); calls += 1
            stopped = raw.stopped_q.numpy() if done.any() else None
            for w in np.flatnonzero(done):
                if rows[w] is not None:
                    continue
                length = float(np.mean([sim.fk_joints(float(stopped[w, ids[2*s]]),float(stopped[w, ids[2*s+1]]))['leg_len'] for s in range(2)]))
                rows[w] = dict(**cases[w], **{k:v for k,v in infos[w].items() if k != 'terminal_observation'}, final_mean_fk_leg_m=length)
                if arm == 'joint_zero':
                    np.testing.assert_array_equal(raw._joint_requested.numpy()[w], 0)
                    np.testing.assert_array_equal(raw._joint_buffers[4].numpy()[w], 0)
                    assert raw._joint_buffers[5].numpy()[w] == 0
            if all(r is not None for r in rows):
                break
        assert all(r is not None for r in rows)
        for w, row in enumerate(rows):
            trace = np.array(actors[w]); assert trace.shape == (int(np.ceil(row['physical_steps']/40)),481+dimension)
            np.testing.assert_array_equal(trace[:, 481:], 0)
            file = directory/f'actor_{row["seed"]}.npz'; np.savez_compressed(file, trace=trace)
            row['zero_actor_trace'] = dict(path=file.name,sha256=sha(file),dimension=dimension)
        result = dict(runs=rows, summary=summary(rows), physics_calls=calls,
            physical=sum(r['physical_safety_passed'] for r in rows),design=sum(r['design_joint_passed'] for r in rows),
            first_episode_world_steps=sum(r['physical_steps'] for r in rows),
            actual_graph_world_steps=calls*40*len(cases), nominal_design_unchanged=True)
        atomic_json(directory/'result.json', result)
        checked = witness.check_files(directory, cases, {r['seed']:r for r in rows})
        assert checked['physics_steps'] == result['first_episode_world_steps']
        if arm == 'joint_zero':
            for row in rows:
                file = directory/row['joint_reference_trace']['path']; assert sha(file) == row['joint_reference_trace']['sha256']
                with np.load(file) as z:
                    adapter.check_log(z['trace'],z['role'],z['phase'])
                    np.testing.assert_array_equal(z['trace'][:, 10:16], 0)
        return result, checked
    except BaseException:
        witness.preserve_partial(raw,directory,cases)
        if arm == 'joint_zero':
            np.savez_compressed(directory/'partial_reference_buffers.npz',trace=raw._joint_buffers[11].numpy(),
                requested=raw._joint_requested.numpy(),filtered=raw._joint_buffers[4].numpy())
        raise
    finally:
        env.close()


def run():
    assert not any((OUT/f).exists() for f in ('started.json','completion.json','failure.json'))
    p = json.loads((OUT/'proposal.json').read_text())
    assert len(p['cases']) == 5 and p['evaluations'] == 10
    assert all(sha(ROOT/n) == h for n,h in p['source_sha256'].items())
    fd, fd_calls = mujoco.mjd_transitionFD, []
    def counted(*args, **kwargs):
        fd_calls.append(float(args[2])); assert len(fd_calls) <= p['constructor_FD_budget']
        return fd(*args,**kwargs)
    mujoco.mjd_transitionFD = counted; results = {}; start=time.perf_counter()
    atomic_json(OUT/'started.json',dict(proposal_sha256=sha(OUT/'proposal.json'),runner_sha256=sha(__file__),evaluations=10))
    try:
        for arm in p['arms']:
            directory=OUT/arm;directory.mkdir(exist_ok=False)
            remaining=p['physical_graph_world_step_budget']-sum(x['result']['actual_graph_world_steps'] for x in results.values())
            result, checked=evaluate(p['cases'],arm,directory,remaining);results[arm]=dict(result=result,checked=checked)
            atomic_json(OUT/'progress.json',dict(completed=len(results)*5,records=results,FD_calls=len(fd_calls)))
            print('COMPLETED304',arm,5,'success',result['summary']['success_count'],flush=True)
        old, new = [results[a]['result']['runs'] for a in p['arms']]
        differences = [{k:[a[k],b[k]] for k in ('success','reason','physical_safety_passed','design_joint_passed','terrain_passed','terrain_exit_passed') if a[k]!=b[k]} for a,b in zip(old,new)]
        passed = not any(differences) and all(r['physical_safety_passed'] and r['design_joint_passed'] for r in old+new)
        atomic_json(OUT/'completion.json',dict(verified=True,round=304,results=results,label_differences=differences,
            engineering_pair_gate_passed=passed,evaluations=10,unique_cases=5,
            baseline_constructor_transitionFD_calls=fd_calls,wall_seconds=time.perf_counter()-start,
            actual_graph_world_steps=sum(x['result']['actual_graph_world_steps'] for x in results.values()),
            first_episode_world_steps=sum(x['result']['first_episode_world_steps'] for x in results.values()),
            new_training_samples=0,formal_PPO_admitted=False,method_advantage_established=False,
            limits='Five existingzero-action development cases; realterminal/autoreset andlogging engineering only. Exacttrajectories/numeric historicalreproduction notassumed.305deepreview before nonzero scientificmechanism protocol.'))
        print('DONE304 engineeringpair gate',passed,flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),completed_arms=list(results),FD_calls=fd_calls,implicit_retry=False))
        raise
    finally:
        mujoco.mjd_transitionFD=fd


if __name__ == '__main__':
    run()
