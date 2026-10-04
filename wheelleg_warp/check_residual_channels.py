"""Fixed-policy channel intervention, not a new policy/performance promotion."""
from pathlib import Path
import argparse,json
import numpy as np
from stable_baselines3 import PPO
from train_height_comparison import evaluate,summary
from training_contract import digest
from dashboard.live_env import atomic_json
from smoke_reward_training import weight_digest

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reward_pilot_v1'

class MaskedPolicy:
    def __init__(self,agent,mask):self.agent=agent;self.mask=np.array(mask,np.float32)
    def predict(self,obs,deterministic=True):
        assert deterministic
        action,state=self.agent.predict(obs,deterministic=True);masked=action*self.mask
        assert np.isfinite(masked).all() and np.max(abs(masked))<=1
        np.testing.assert_array_equal(masked[:,self.mask==0],np.zeros_like(masked[:,self.mask==0]))
        return masked,state

def run(out):
    cases=json.loads((S/'proposal.json').read_text())['development'];p=S/'runs/potential/1610/step_200000'
    record=json.loads(p.with_suffix('.json').read_text())
    assert digest(p.with_suffix('.zip'))==record['checkpoint']['checkpoint_sha256']
    assert digest(p.with_suffix('.pkl'))==record['checkpoint']['normalization_sha256']
    contract=json.loads((S/'trainer_contract.json').read_text())
    assert all(digest(ROOT/name)==value for name,value in contract['source_sha256'].items())
    conditions={'all':[1,1,1],'legs_only':[1,1,0],'wheel_only':[0,0,1],'zero':[0,0,0]}
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'registration.json',dict(conditions=conditions,cases=cases,checkpoint=record['checkpoint'],
        selection='Exploratory worst-decline seed1610 potential final200k after failed pilot',
        intervention='Fixed deterministic actor means multiplied by0/1 per channel; body/contacts/action history respond in closed loop',
        constraints='Same model/base/torque authority/evaluation gates; no gain/weight update, no reduced-task success criterion',
        inference_limit='Only causal effect of imposed channel mask in this fixed-policy/public-case system; not cause of learning failure or pure physical-prior attribution',
        training_updates=0,old_gate_or_final_used=False,source_sha256=digest(Path(__file__))))
    agent=PPO.load(str(p)+'.zip',device='cuda');before=(agent.num_timesteps,agent._n_updates,weight_digest(agent));results={}
    for name,mask in conditions.items():
        rows=evaluate(cases,'diff3',MaskedPolicy(agent,mask),str(p)+'.pkl')
        assert before==(agent.num_timesteps,agent._n_updates,weight_digest(agent))
        result=dict(summary=summary(rows),physical=sum(r['physical_safety_passed'] for r in rows),design=sum(r['design_joint_passed'] for r in rows),
            terrain_missing=sum(not r['terrain_passed'] for r in rows),yaw_failed=sum(r['peak_deg'][2]>5 for r in rows),runs=rows)
        atomic_json(out/(name+'.json'),result);results[name]={k:v for k,v in result.items() if k!='runs'}
        print(name,results[name],flush=True)
    atomic_json(out/'result.json',dict(verified=True,episodes=384,unique_cases=96,results=results,source_sha256=digest(Path(__file__)),
        weights_optimizer_counters_unchanged=True,normalization_updates=0,training_updates=0,
        scope='FreshGPU fixed-policy closed-loop mask intervention, not independent test or model promotion; original pilot gate/disposition unchanged'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
