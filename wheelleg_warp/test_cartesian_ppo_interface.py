"""No-physics policy/RMS/recording admission checks, plus a two-row GPU copy check."""
import copy
import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from test_reference_learning_policy import NoStep
from test_cartesian_ppo_recorder import run as recorder_check
import run_cartesian_ppo_interface as worker


def run():
    recorder_check();torch.set_num_threads(1);p=worker.inputs()
    budget=worker.NetworkBudget(dict(policy_forward_row_budget=10,value_only_forward_row_budget=1,value_only_forward_call_budget=1))
    obs=torch.zeros((10,481));calls=[]
    def policy(x):calls.append('policy');return torch.zeros((10,3)),torch.zeros((10,1)),torch.zeros(10)
    def value(x):calls.append('value');return torch.zeros((len(x),1))
    budget.call('policy',policy,obs);budget.call('value',value,obs[:1])
    for kind,fn,x in [('policy',policy,obs),('value',value,obs[:1])]:
        try:budget.call(kind,fn,x)
        except RuntimeError:pass
        else:raise AssertionError('Network budget overflow accepted')
    assert calls==['policy','value']
    tree=ast.parse(Path(worker.__file__).read_text());cb=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReceiptCallback')
    step=next(n for n in cb.body if isinstance(n,ast.FunctionDef) and n.name=='_on_step')
    assignment=next(n for n in step.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='record' for t in n.targets))
    assert {'terminal_expected_history','terminal_expected_verified'} <= {k.arg for k in assignment.value.keywords if k.arg}
    with tempfile.TemporaryDirectory() as directory:
        records=[dict(terminal_expected_history=np.zeros((10,481),np.float32),terminal_expected_verified=np.array(True)),
                 dict(terminal_expected_history=np.full((10,481),np.nan,np.float32),terminal_expected_verified=np.array(False))];blocks=[]
        worker.parent.flush(Path(directory),records,blocks)
        with np.load(Path(directory)/'block_000.npz') as z:
            assert z['terminal_expected_verified'].tolist()==[True,False] and np.isnan(z['terminal_expected_history'][1]).all()
    try:worker.forbidden()
    except RuntimeError:pass
    else:raise AssertionError('Optimization accepted')
    class Toy(NoStep):
        def __init__(self):super().__init__(6)
        def reset(self):return np.zeros((10,481),np.float32)
        def step_async(self,a):self.actions=a.copy()
        def step_wait(self):
            done=np.zeros(10,bool);done[0]=True
            return np.full((10,481),.1,np.float32),np.full(10,.123456789,np.float64),done,[{'terminal_observation':np.full(481,12.34567,np.float32)}]+[{} for _ in range(9)]
    raw=SimpleNamespace(_cartesian_ppo_buffers={'nominal':SimpleNamespace(numpy=lambda:np.zeros((40,10,5)))})
    toy=Toy();tap=worker.RawReceiptTap(toy,raw);norm=VecNormalize(worker.ForceActions(tap,'D3'),**p['normalization']);tap.norm=norm
    norm.reset()
    with patch.object(worker.recorder,'post_private',lambda r:{'previous_sensor_steps':np.zeros(10)}),patch.object(worker.recorder,'bank_actual',lambda r:{'param':np.zeros((10,17))}):
        _,reward,done,infos=norm.step(np.full((10,3),.2,np.float32))
    np.testing.assert_array_equal(reward,tap.packet['reward'].astype(np.float32));assert not np.array_equal(reward,tap.packet['reward'])
    np.testing.assert_array_equal(tap.packet['infos'][0]['terminal_observation'],np.full(481,12.34567,np.float32))
    assert not np.array_equal(infos[0]['terminal_observation'],tap.packet['infos'][0]['terminal_observation'])
    infos[0]['terminal_observation'][0]=999;assert tap.packet['infos'][0]['terminal_observation'][0]!=999
    np.testing.assert_array_equal(toy.actions,worker.embed(np.full((10,3),.2,np.float32),'D3',10))
    assert norm.obs_rms.count==20.0001 and norm.ret_rms.count==10.0001
    norm.close()
    snapshots=[]
    for _ in range(2):
        env=VecNormalize(NoStep(3),**p['normalization']);kwargs=copy.deepcopy(p['ppo'])
        model=PPO('MlpPolicy',env,seed=33531,device='cpu',**kwargs)
        with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_();model.policy.log_std.fill_(p['reference_log_std'])
        snapshots.append(worker.policy_snapshot(model,env))
        assert model.rollout_buffer.buffer_size==50 and model._n_updates==model.num_timesteps==0 and not model.policy.optimizer.state_dict()['state']
        np.testing.assert_array_equal(torch.exp(model.policy.log_std).detach().numpy(),np.full(3,.25,np.float32));env.close()
    worker.equal(snapshots[0],snapshots[1])
    wp.init();wp.set_device('cuda:0')
    speed=np.arange(32,dtype=np.float32).reshape(2,16)/4;ids=np.arange(11,dtype=np.int32);diag=np.zeros((2,38));diag[0,13]=1
    out=wp.full((40,2,5),9,dtype=worker.recorder.D)
    wp.launch(worker.recorder.record,2,[0,wp.array(speed),wp.array(np.array([1,0],np.int32)),wp.array(ids),wp.array(diag,dtype=worker.recorder.D),out])
    np.testing.assert_array_equal(out.numpy()[0],[[1,speed[0,8],speed[0,9],1,0],[0,0,0,0,0]])
    np.testing.assert_array_equal(out.numpy()[1:],9)
    print('PASS376 network pre-call caps;actual terminal deepcopy;float64-to-float32 reward;RMS/reset;fresh policy/optimizer/fullRNG/std matching;2GPU readonly copy rows;0physics/FD/network forwards/optimizer',flush=True)


if __name__=='__main__':run()
