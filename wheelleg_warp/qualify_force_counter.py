"""Real completed/partial/guard counter conservation with two zero-action worlds."""
import json
import numpy as np
import mujoco
import warp as wp
from fixed_reference_force import make_env
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/fixed_force_main_worker_v1'


def run():
    q=json.loads((OUT/'counter_proposal.json').read_text());assert not any((OUT/n).exists() for n in ('counter_started.json','counter_completion.json','counter_failure.json'))
    p=json.loads((ROOT/q['bank_protocol']).read_text());p['worlds']=2
    calls=0;fd_calls=[];graph=wp.capture_launch;fd=mujoco.mjd_transitionFD
    def capture(*args,**kwargs):
        nonlocal calls
        calls+=1;assert calls*40*2<=q['graph_world_step_budget'];return graph(*args,**kwargs)
    def counted(*args,**kwargs):fd_calls.append(float(args[2]));assert len(fd_calls)<=q['FD_budget'];return fd(*args,**kwargs)
    wp.capture_launch=capture;mujoco.mjd_transitionFD=counted;rows=[];witnesses=[];norm=None
    atomic_json(OUT/'counter_started.json',dict(proposal_sha256=sha(OUT/'counter_proposal.json')))
    try:
        curriculum,raw,history,norm=make_env(p,q['arm'],q['seed'],guarded=True);norm.training=False;norm.reset()
        for tick in range(q['max_actor_ticks']):
            _,_,done,infos=norm.step(np.zeros((2,3),np.float32))
            for w in np.flatnonzero(done):rows.append({**{k:v for k,v in infos[w].items() if k!='terminal_observation'},'world_index':int(w)})
            complete=sum(r['physical_steps'] for r in rows);partial=int(raw.state.numpy()[:,0].sum());valid=raw._joint_guard_stats['valid_substeps']
            assert complete+partial==valid
            if done.any():
                np.testing.assert_array_equal(raw._joint_guard_buffers[0].numpy()[done],0)
                witnesses.append(dict(tick=tick,done_worlds=np.flatnonzero(done).tolist(),completed_physical_steps=complete,partial_episode_steps=partial,actual_valid_guard_steps=valid,actual_graph_world_steps=calls*40*2))
                np.savez_compressed(OUT/'terminal_snapshot.npz',terminal_q=raw.stopped_q.numpy(),terminal_v=raw.stopped_v.numpy(),current_q=raw.data.qpos.numpy(),current_v=raw.data.qvel.numpy(),guard_memory=raw._joint_guard_buffers[0].numpy(),guard_last=raw._joint_guard_buffers[1].numpy())
                break
        assert witnesses and rows and all(r['physical_safety_passed'] and r['design_joint_passed'] for r in rows)
        atomic_json(OUT/'counter_completion.json',dict(verified=True,witnesses=witnesses,episodes=rows,actual_graph_world_steps=calls*40*2,FD_calls=fd_calls,new_learning_samples=0,scientific_evaluations=0))
        print('PASS334 actualcompleted+partial=guard valid onallsteps,real terminal/maskedreset andstoppedqv saved;0learning')
    except BaseException as error:
        atomic_json(OUT/'counter_failure.json',dict(error=repr(error),actual_graph_world_steps=calls*40*2,FD_calls=fd_calls,implicit_retry=False));raise
    finally:
        if norm is not None:norm.close()
        wp.capture_launch=graph;mujoco.mjd_transitionFD=fd


if __name__=='__main__':run()
