"""Real parent curriculum logic with fake plant reset/forward and real GPU auxiliary reset."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import warp as wp
import train_height_comparison as base
from cartesian_curriculum import CartesianCurriculum,refresh_after_curriculum
from cartesian_pair_kernel import reset_aux
from native.controller import D
from state_estimation import leg_kinematics
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1'


class Buffer:
    def __init__(self,value):self.value=np.array(value,copy=True)
    def numpy(self):return self.value.copy()
    def assign(self,value):self.value=np.array(value,copy=True)


def geometry(obs):
    return np.concatenate([leg_kinematics(obs[12+2*s:14+2*s].astype(float),np.zeros(2))[0] for s in range(2)])


def exercise(cls,low,high,low_case,high_case):
    n=3;done=np.array([True,True,False]);public=np.array([low,high,low],np.float32)
    public[2,32:38]=[.25,-.25]*3
    initial_reference=np.array([geometry(x) for x in public])
    latent=wp.full((n,3),.25,dtype=D);shadow=wp.full((n,22),.25,dtype=D)
    reference=wp.array(initial_reference,dtype=D);mask=wp.zeros(n,dtype=int)
    raw=SimpleNamespace(num_envs=n,obs=wp.array(public),mask=wp.zeros(n,dtype=int),reset_args=[],
        model=SimpleNamespace(body_mass=Buffer(np.ones((n,2))*7),stat=SimpleNamespace(meaninertia=Buffer(np.ones(n)))),
        q0=Buffer(np.zeros((n,17))),param=Buffer(np.zeros((n,17))),obs0=Buffer(public),k={'reference':Buffer(np.zeros((n,6)))},
        data=SimpleNamespace(geom_xpos=Buffer(np.zeros((n,1,3))),geom_xmat=Buffer(np.zeros((n,1,9)))),
        scenarios=[base.HeightTerrainScenario(**x['scenario']) for x in (low_case,high_case,low_case)],
        stand_heights=np.array([.115,.38,.115]),task_goals=[2.55]*n,
        _cartesian_buffers=dict(latent=latent,shadow=shadow,reference=reference,mask=mask))
    terminal=[np.arange(38,dtype=np.float32)+100*i for i in range(n)]
    def raw_wait():
        # The previously accepted raw adapter refreshes the old reset packet before the parent changes bank.
        mask.assign(done.astype(np.int32));wp.launch(reset_aux,n,[mask,raw.obs,latent,shadow,reference])
        return raw.obs.numpy(),np.zeros(n),done.copy(),[{'terminal_observation':terminal[i].copy()} if done[i] else {} for i in range(n)]
    raw.step_wait=raw_wait
    obj=cls.__new__(cls);obj.venv=raw;obj.num_envs=n;obj.policy_steps=0;obj.milestones=[2,5];obj.stages=np.array([1,2,1])
    obj.banks={1:[dict(seed=101+i,scenario=low_case['scenario']) for i in range(n)],2:[dict(seed=201+i,scenario=high_case['scenario']) for i in range(n)]}
    next_public=np.array([high,high,high],np.float32)
    obj.cache={2:dict(model={'body_mass':np.ones((n,2))*8},meaninertia=np.ones(n)*2,q0=np.ones((n,17)),param=np.ones((n,17)),
        obs0=next_public,reference=np.ones((n,6)),geom_xpos=np.ones((n,1,3)),geom_xmat=np.ones((n,1,9)),info={'task_goals':[3.,3.,3.]})}
    obj.transition_rows=0;obj.transition_log=[];launch=wp.launch;events=[]
    def fake_reset(kernel,dim,inputs=None,**kwargs):
        if kernel is base.reset_rows:
            selected=raw.mask.numpy().astype(bool);value=raw.obs.numpy();value[selected]=raw.obs0.numpy()[selected];raw.obs.assign(value)
            events.append(('parent_bank_reset',np.flatnonzero(selected).tolist()));return
        return launch(kernel,dim,inputs,**kwargs)
    with patch.object(wp,'launch',fake_reset),patch.object(base.mjw,'forward',lambda *a:events.append(('fake_forward',[]))):
        result=obj.step_wait()
    np.testing.assert_array_equal(obj.stages,[2,2,1]);assert obj.transition_rows==1
    assert obj.transition_log[0]['worlds']==[0] and events==[('parent_bank_reset',[0]),('fake_forward',[])]
    assert result[3][0]['seed']==101 and result[3][0]['curriculum_stage']==1
    assert result[3][1]['seed']==202 and result[3][1]['curriculum_stage']==2
    for w in (0,1):np.testing.assert_array_equal(result[3][w]['terminal_observation'],terminal[w])
    np.testing.assert_array_equal(result[0][0],high);np.testing.assert_array_equal(result[0][2],public[2])
    np.testing.assert_array_equal(latent.numpy()[2],.25);np.testing.assert_array_equal(shadow.numpy()[2],.25)
    np.testing.assert_array_equal(reference.numpy()[2],initial_reference[2]);np.testing.assert_array_equal(raw.mask.numpy(),[1,0,0])
    np.testing.assert_array_equal(raw.model.body_mass.numpy()[1:],7)
    error=float(abs(reference.numpy()[0]-geometry(high)).max())
    if cls is CartesianCurriculum:
        assert error<1e-7
        np.testing.assert_array_equal(latent.numpy()[:2],0);np.testing.assert_array_equal(shadow.numpy()[:2,16:22],0)
        before=reference.numpy().copy();refresh_after_curriculum(raw,result[0],np.zeros(n,bool));np.testing.assert_array_equal(reference.numpy(),before)
        bad=result[0].copy();bad[0,12]+=.1
        try:refresh_after_curriculum(raw,bad,done)
        except AssertionError:pass
        else:raise AssertionError('Foreign reset packet accepted')
        try:refresh_after_curriculum(raw,result[0],done.astype(int))
        except ValueError:pass
        else:raise AssertionError('Non-Boolean reset mask accepted')
    else:assert error>.2
    return dict(stale_reference_error_m=error,stages=obj.stages.tolist(),terminal_old_episode_preserved=True,nonreset_world_unchanged=True,
        parent_mask_preserved=True,model_change_rows=[0],events=events)


def run():
    data=OUT/'cartesian_pair_delivery_repair_v2';receipt=json.loads((data/'review.json').read_text());packets=[];cases=[];hashes={}
    for label in ('6301003_B0_zero','6301033_B0_zero'):
        folder=data/'runtime'/label;file=folder/'initial.npz';assert sha(file)==receipt['raw_sha256'][str(file.relative_to(ROOT))]
        with np.load(file) as z:packets.append(z['frames'][0,-1,:38].copy())
        cases.append(json.loads((folder/'result.json').read_text())['runs'][0]);hashes[str(file.relative_to(ROOT))]=sha(file)
    wp.init();wp.set_device('cuda:0')
    before=exercise(base.CurriculumEnv,*packets,*cases);after=exercise(CartesianCurriculum,*packets,*cases)
    path=OUT/'round367_curriculum_lifecycle.json';assert not path.exists()
    atomic_json(path,dict(round=367,verified=True,before=before,after=after,input_sha256=hashes,
        source_sha256={f:sha(ROOT/f) for f in ['wheelleg_warp/cartesian_curriculum.py','wheelleg_warp/test_cartesian_curriculum.py','wheelleg_warp/train_height_comparison.py','wheelleg_warp/cartesian_pair_kernel.py']},
        scope='Actual CurriculumEnv.step_wait bank/metadata logic with fake plant reset_rows/forward, real GPU reset_aux,3 rows. Not a real multiworld robot or training run.',
        new_physics_steps=0,new_FD=0,new_learning_samples=0,new_model_forward_rows=0,training_admitted=False))
    print('PASS367 real parent order: old reference error',before['stale_reference_error_m'],'post-parent refresh error',after['stale_reference_error_m'],'unchanged rows/terminal ownership/mask/rejection checks passed;0physics',flush=True)


if __name__=='__main__':run()
