"""Recheck recorded full-deadline coverage, original domains and task evidence without GPU."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import wheelleg_sim as sim
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
from training_contract import STOP_DISTANCE_M,TAIL_START_S,TAIL_SPEED_M_S,POST_ARRIVAL_S


def check(source):
    d=json.loads((source/'verification.json').read_text());z=np.load(source/'prediction.npz',allow_pickle=False)
    a=z['schedule'];delta=np.diff(np.concatenate([np.zeros_like(a[:1]),a]),axis=0)
    assert abs(a).max()<=1. and np.sum(abs(delta),axis=2).max()<=.10000000001
    np.testing.assert_allclose(a,np.clip(np.arange(1,len(a)+1)[:,None,None]*z['direction'][None],-1,1),atol=1e-12)
    assert np.isfinite(z['trace']).all() and np.isfinite(z['task_trace']).all()
    assert z['trace'].shape[:2]==(len(a)*10,38) and z['task_trace'].shape==(*z['trace'].shape[:2],2)
    for row in d['rows']:
        w=row['world'];s=z['initial_task'][w];steps=int(z['end_steps'][w]);t=z['trace'][:steps,w*19:(w+1)*19];height=z['task_trace'][:steps,w*19:(w+1)*19,0]
        elapsed=(s[0]+np.arange(1,steps+1))*.0005-s[1]
        assert elapsed[-1]>=POST_ARRIVAL_S-1e-10 and elapsed[-1]-.0005<POST_ARRIVAL_S-1e-10
        # Original pre-arrival evidence is already irrevocable; include it too.
        assert s[37]==s[0]>0 and min(s[31:33])>=HEIGHT_115_GEOMETRIC_MIN and s[33]>=0 and max(s[35:37])<=1e-6
        assert max(s[8:11])<=np.deg2rad(5) and s[5]>0 and np.sqrt(s[4]/s[5])<=.2
        distance=np.maximum(s[6],np.linalg.norm(t[:,:,:2]-s[None,None,2:4],axis=2).max(axis=0))
        speed=np.maximum(s[7],np.linalg.norm(t[elapsed>=TAIL_START_S,:,17:19],axis=2).max(axis=0))
        np.testing.assert_allclose(distance,row['stop_distance_m'],atol=1e-12)
        np.testing.assert_allclose(speed,row['tail_speed_m_s'],atol=1e-12)
        np.testing.assert_array_equal((distance<=STOP_DISTANCE_M)&(speed<=TAIL_SPEED_M_S),row['task'])
        rmse=np.sqrt((s[28]+np.sum((height-.115)**2,axis=0)*.0005)/(s[29]+steps*.0005))
        np.testing.assert_allclose(rmse,row['height_rmse_m'],atol=1e-12)
        # Independent CPU FK at startup, midpoint and terminal for every arm.
        for sample in (0,steps//2,steps-1):
            for arm in range(19):
                q=t[sample,arm,:17]
                expected=(sim.fk_joints(q[7],q[10])['leg_len']+sim.fk_joints(q[12],q[15])['leg_len'])/2
                assert abs(expected-height[sample,arm])<1e-10
        valid=np.array(row['physical'])&np.array(row['task'])&np.array(row['height_pass'])&~np.array(row['nonwheel_contact'])
        np.testing.assert_array_equal(valid,row['valid'])
        if row['selected_arm'] is not None:assert valid[row['selected_arm']]
    assert not d['actual_future_used_for_selection']
    print('PASS38 complete backups: task deadline, prior evidence, action domain, recorded gates and independent CPU FK')


def check_graph():
    import warp as wp
    from native.environment import NativeEnv
    from native.terrain import bank_height_115
    from probe_height_115_margin import cases
    from probe_braking_feedback import execution_graph,D
    envs=[NativeEnv(n=2,scenario=cases()[4:6],bank_factory=bank_height_115,height_conditioned=True,height_design='range115',
        residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=True) for _ in range(2)]
    try:
        for env in envs:env.reset()
        extra=wp.zeros((2,6),dtype=D);graph=execution_graph(envs[1],extra)
        for _ in range(4):
            wp.capture_launch(envs[0].graph)
            for _ in range(4):wp.capture_launch(graph)
            np.testing.assert_allclose(envs[0].data.qpos.numpy(),envs[1].data.qpos.numpy(),atol=2e-6,rtol=0)
            np.testing.assert_allclose(envs[0].data.qvel.numpy(),envs[1].data.qvel.numpy(),atol=1e-3,rtol=0)
            np.testing.assert_allclose(envs[0].data.ctrl.numpy(),envs[1].data.ctrl.numpy(),atol=1e-5,rtol=0)
            for env in envs:
                state=env.state.numpy();np.testing.assert_array_equal(state[:,37],state[:,0])
                assert min(state[:,31:33].ravel())>=HEIGHT_115_GEOMETRIC_MIN and min(state[:,33])>=0
                assert max(state[:,35:37].ravel())<=1e-6
        print('PASS shared5ms execution graph versus native40-substep graph,160 steps per world')
    finally:
        for env in envs:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--gpu-graph-check',action='store_true');args=p.parse_args();check(args.source)
    if args.gpu_graph_check:check_graph()
