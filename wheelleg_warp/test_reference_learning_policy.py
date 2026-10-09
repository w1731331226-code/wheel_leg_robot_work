"""Fresh policy initialization/reload checks; fake environment forbids stepping."""
import tempfile
from pathlib import Path
import json
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from reference_learning_engineering import OUT,agent,equal


class NoStep(VecEnv):
    def __init__(self,dimension):
        super().__init__(10,Box(-np.inf,np.inf,(481,),dtype=np.float32),Box(-1,1,(dimension,),dtype=np.float32))
    def reset(self):
        raise AssertionError('No real reset/model query admitted')
    def step_async(self,actions):
        raise AssertionError('No simulation or rollout admitted')
    def step_wait(self):
        raise AssertionError('No simulation or rollout admitted')
    def close(self):
        pass
    def get_attr(self,name,indices=None):
        return [None]*10
    def set_attr(self,*args,**kwargs):
        raise AssertionError('No environment mutation')
    def env_method(self,*args,**kwargs):
        raise AssertionError('No environment mutation')
    def env_is_wrapped(self,*args,**kwargs):
        return [False]*10


def run():
    torch.set_num_threads(1)
    config=json.loads((OUT/'runtime_config.json').read_text())
    models=[];shared=None
    for dimension in (3,6):
        model,shared=agent(config,NoStep(dimension),shared);models.append(model)
        assert model.num_timesteps==model._n_updates==0 and not model.policy.optimizer.state_dict()['state']
        assert all(p.device.type=='cuda' and torch.isfinite(p).all() for p in model.policy.parameters())
    x=torch.as_tensor(np.random.default_rng(31631).normal(size=(128,481)).astype(np.float32),device='cuda')
    with torch.no_grad():
        for model in models:
            d=model.policy.get_distribution(x).distribution
            assert torch.count_nonzero(d.mean)==0
            torch.testing.assert_close(d.stddev,torch.full_like(d.stddev,.25),rtol=0,atol=3e-8)
        equal(models[0].policy.predict_values(x),models[1].policy.predict_values(x))
        equal(models[0].policy.mlp_extractor.state_dict(),models[1].policy.mlp_extractor.state_dict())
    for model in models:
        owned={id(p) for group in model.policy.optimizer.param_groups for p in group['params']}
        assert owned=={id(p) for p in model.policy.parameters()}
    with tempfile.TemporaryDirectory(prefix='reference-policy-unit-') as temporary:
        for i,model in enumerate(models):
            file=Path(temporary)/f'arm_{i}.zip';model.save(str(file))
            loaded=PPO.load(str(file),device='cuda')
            equal(model.policy.state_dict(),loaded.policy.state_dict());equal(model.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict())
            with torch.no_grad():equal(model.policy.get_distribution(x).distribution.mean,loaded.policy.get_distribution(x).distribution.mean)
    print('PASS316 fresh3/6CUDA policy zero-means/sharedfeatures/critic/std/optimizerownership/exactreload;0learn/rollout/physics',flush=True)


if __name__=='__main__':run()
