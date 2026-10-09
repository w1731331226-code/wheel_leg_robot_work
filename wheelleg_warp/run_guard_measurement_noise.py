"""Guard-only measurement perturbation; never writes simulator q or v."""
import json
import numpy as np
import torch
import mujoco
import warp as wp
import execution_history_evaluation as evaluation
import joint_state_guard as guard
from fixed_reference_force import PriorActor
from reference_learning_engineering import agent
from test_reference_learning_policy import NoStep
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/joint_guard_measurement_noise_v1'


@wp.kernel
def measure(q:wp.array2d[float],v:wp.array2d[float],ids:wp.array[int],memory:wp.array2d[guard.D],
            noise:wp.array3d[float],qobs:wp.array2d[float],vobs:wp.array2d[float],overflow:wp.array[int]):
    w=wp.tid();step=int(memory[w,15])
    for j in range(q.shape[1]):qobs[w,j]=q[w,j]
    for j in range(v.shape[1]):vobs[w,j]=v[w,j]
    if step>=noise.shape[0]:overflow[w]=1;return
    for j in range(4):
        qobs[w,ids[j]]+=noise[step,w,j];vobs[w,ids[4+j]]+=noise[step,w,4+j]


def instrument(factory,cases,directory,noise):
    n=len(cases);qobs=wp.zeros((n,17));vobs=wp.zeros((n,16));overflow=wp.zeros(n,dtype=int)
    launch=wp.launch;calls=0
    def intercept(kernel,dim,inputs=None,**kwargs):
        nonlocal calls
        if kernel is guard.guard:
            calls+=1
            launch(measure,dim,[inputs[1],inputs[2],inputs[4],inputs[8],noise,qobs,vobs,overflow])
            args=list(inputs);args[1]=qobs;args[2]=vobs
            return launch(kernel,dim,args,**kwargs)
        return launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
    wp.launch=intercept
    try:result=guard.instrument(factory,cases,directory)
    finally:wp.launch=launch
    assert calls==80;raw=result[0];raw._guard_measurement_buffers=(noise,qobs,vobs,overflow)
    wait=raw.step_wait
    def step_wait():
        value=wait();assert not overflow.numpy().any(),'Registered noise table consumed'
        return value
    raw.step_wait=step_wait
    return result


def self_check():
    q=np.arange(34,dtype=np.float32).reshape(2,17);v=np.arange(32,dtype=np.float32).reshape(2,16)
    ids=np.array([7,10,12,15,6,9,11,14],np.int32);noise=np.full((1,2,8),.001,np.float32)
    qa=wp.array(q);va=wp.array(v);qo=wp.zeros_like(qa);vo=wp.zeros_like(va);memory=wp.zeros((2,16),dtype=guard.D);overflow=wp.zeros(2,dtype=int)
    wp.launch(measure,2,[qa,va,wp.array(ids,dtype=int),memory,wp.array(noise),qo,vo,overflow])
    np.testing.assert_array_equal(qa.numpy(),q);np.testing.assert_array_equal(va.numpy(),v)
    expectedq=q.copy();expectedq[:,ids[:4]]+=.001;expectedv=v.copy();expectedv[:,ids[4:]]+=.001
    np.testing.assert_array_equal(qo.numpy(),expectedq);np.testing.assert_array_equal(vo.numpy(),expectedv);assert not overflow.numpy().any()
    memory.fill_(1);wp.launch(measure,2,[qa,va,wp.array(ids,dtype=int),memory,wp.array(noise),qo,vo,overflow]);assert (overflow.numpy()==1).all()
    print('PASS330 measurement copy/onlyactivechannels/physicalreadonly/tableoverflow;0physics')


def run():
    p=json.loads((OUT/'proposal.json').read_text());assert not any((OUT/n).exists() for n in ('started.json','completion.json','failure.json'))
    assert all(sha(ROOT/f)==h for f,h in p['source_sha256'].items());self_check();torch.set_num_threads(1)
    rng=np.random.default_rng(p['noise_seed']);table=rng.normal(size=(p['noise_table_steps'],12,8));table[:,:,:4]*=p['sigma_position_rad'];table[:,:,4:]*=p['sigma_velocity_rad_s'];table=table.astype(np.float32)
    np.savez_compressed(OUT/'noise_table.npz',noise=table);noise=wp.array(table)
    model,_=agent(p,NoStep(3));directory=OUT/'noisy_D3';directory.mkdir(exist_ok=False);torch.manual_seed(p['action_seed'])
    torch.save(dict(RNG=torch.get_rng_state(),cuda_RNG=torch.cuda.get_rng_state_all()),directory/'initial_RNG.pt')
    torch.save(dict(policy=model.policy.state_dict(),optimizer=model.policy.optimizer.state_dict()),OUT/'initial_D3.pt')
    original=evaluation.make_env;fd=mujoco.mjd_transitionFD;graph=wp.capture_launch;steps=0;fd_calls=[]
    def make_env(cases,arm,norm,out):return instrument(lambda:original(cases,arm,norm,out),cases,out,noise)
    def counted(*args,**kwargs):fd_calls.append(float(args[2]));assert len(fd_calls)<=p['FD_budget'];return fd(*args,**kwargs)
    def capture(*args,**kwargs):
        nonlocal steps
        steps+=40*12;assert steps<=p['graph_world_step_budget'];return graph(*args,**kwargs)
    evaluation.make_env=make_env;mujoco.mjd_transitionFD=counted;wp.capture_launch=capture
    atomic_json(OUT/'started.json',dict(proposal_sha256=sha(OUT/'proposal.json'),noise_table_sha256=sha(OUT/'noise_table.npz')))
    try:
        result,checked=evaluation.evaluate(p['cases'],'H1',PriorActor(model,'D3'),None,directory)
        assert model.num_timesteps==model._n_updates==0 and not model.policy.optimizer.state_dict()['state']
        atomic_json(OUT/'completion.json',dict(verified=True,episodes=12,physical=result['physical'],design=result['design'],success=result['summary']['success_count'],
            actual_graph_world_steps=steps,first_episode_world_steps=sum(r['physical_steps'] for r in result['runs']),FD_calls=fd_calls,checked=checked,
            learning_samples=0,new_learning_admitted=False,formal5_admitted=False,result_sha256=sha(directory/'result.json'),scope=p['measurement_scope']))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),actual_graph_world_steps=steps,FD_calls=fd_calls,implicit_retry=False));raise
    finally:evaluation.make_env=original;mujoco.mjd_transitionFD=fd;wp.capture_launch=graph


if __name__=='__main__':run()
