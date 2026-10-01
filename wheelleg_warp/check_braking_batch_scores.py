"""Recorded candidate equivalence and archived physical-failure rejection; no new physics."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.terrain import model,HeightTerrainScenario
from model_lqr import sagittal_basis
from select_braking_common_action import lqr_cost,batch_scores


def check():
    m=model(HeightTerrainScenario(stand_height_m=.115));ref,Q,R,P=lqr_cost(m);b,u=sagittal_basis(m)
    project=np.linalg.pinv(b);input_project=np.linalg.pinv(u)
    folder=ROOT/'wheelleg_warp/results/braking_nominal_rollout_common_20261001';raw=np.load(folder/'rollout.npz',allow_pickle=False)
    prior=np.load(ROOT/'wheelleg_warp/results/braking_candidate_selection_checked_20261001/selection.npz',allow_pickle=False)
    n=234;q=raw['qpos'][:,n:];v=raw['qvel'][:,n:];commands=raw['commands'][:,n:]
    costs,valid=batch_scores(m,q,v,commands,commands,raw['initial_qpos'][n:],raw['initial_qvel'][n:],raw['initial_controller'][n:],ref,Q,R,P,project,input_project)
    # Dataset uses known unit-gain direct motors; commands equal actuator force.
    np.testing.assert_allclose(costs,prior['candidate_costs'].ravel(),rtol=1e-10,atol=1e-7)
    assert np.array_equal(valid,prior['valid'].ravel())
    assert np.array_equal(np.argmin(np.where(valid,costs,np.inf).reshape(18,13),axis=1),prior['indices']%13)
    bad=np.load(ROOT/'wheelleg_warp/results/radial_braking_response_20261001/nonlinear_lp_replay.npz',allow_pickle=False)
    _,invalid=batch_scores(m,bad['qpos'][:,None],bad['qvel'][:,None],bad['ctrl'][:,None],bad['ctrl'][:,None],bad['qpos'][0:1],bad['qvel'][0:1],np.zeros((1,23)),ref,Q,R,P,project,input_project)
    assert not invalid[0]
    # Same local motion at two distances from a fixed arrival origin: the LQR
    # reference moves with the controller, so these task budgets are invisible.
    local_q=q[:,0:1].astype(float);local_initial=raw['initial_qpos'][n:n+1].astype(float);local_memory=raw['initial_controller'][n:n+1].astype(float)
    shift=.05-local_initial[0,0]
    local_q[:,:,0]+=shift;local_initial[:,0]+=shift;local_memory[:,14]+=shift
    args=(m,local_q,v[:,0:1],commands[:,0:1],commands[:,0:1],local_initial,raw['initial_qvel'][n:n+1],local_memory,ref,Q,R,P,project,input_project)
    near_cost,near_valid=batch_scores(*args)
    far_q=local_q.copy();far_initial=local_initial.copy();far_memory=local_memory.copy()
    far_q[:,:,0]+=.7;far_initial[:,0]+=.7;far_memory[:,14]+=.7
    far_cost,far_valid=batch_scores(m,far_q,v[:,0:1],commands[:,0:1],commands[:,0:1],far_initial,raw['initial_qvel'][n:n+1],far_memory,ref,Q,R,P,project,input_project)
    np.testing.assert_allclose(near_cost,far_cost,rtol=1e-10,atol=1e-7)
    assert near_valid[0] and far_valid[0]
    near_distance=np.max(np.linalg.norm(np.concatenate([local_initial[None,:,:2],local_q[:,:,:2]]),axis=2))
    far_distance=np.max(np.linalg.norm(np.concatenate([far_initial[None,:,:2],far_q[:,:,:2]]),axis=2))
    assert near_distance<=.6<far_distance
    print('PASS234 costs, validity,18 selections and archived failed-plan rejection')
    print(f'TASK GAP: identical local cost/validity, fixed-origin distances {near_distance:.6f}/{far_distance:.6f}m; arrival budget is absent')


if __name__=='__main__':check()
