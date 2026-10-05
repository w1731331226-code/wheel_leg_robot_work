"""Registered nominal-only command controls, evaluated after all main learners."""
from pathlib import Path
import json
from stable_baselines3 import PPO
from reference_budget_train import OUT,proposal
from pilot_reference_budget import capture
from review_yaw_sector import sha
from dashboard.live_env import atomic_json


def run():
    p=proposal();progress=json.loads((OUT/'main_progress.json').read_text());assert progress['status']=='complete' and progress['trained_policy_steps']==2400000
    directory=OUT/'zero_controls';directory.mkdir(exist_ok=False);records=[]
    for seed in p['seeds']:
        path=OUT/'runs/L2-room'/str(seed);report=json.loads((path/'verification.json').read_text());assert report['verified'] and report['policy_steps']==200000
        prefix=path/'step_200000';model=PPO.load(str(prefix)+'.zip',device='cuda')
        for panel in ['regular','controlled']:
            result=capture(p[panel],model,str(prefix)+'.pkl',None,2);target=directory/f'{seed}_{panel}.json';atomic_json(target,result)
            records.append(dict(seed=seed,panel=panel,path='zero_controls/'+target.name,sha256=sha(target),episodes=len(p[panel])))
    assert sum(r['episodes'] for r in records)==408
    atomic_json(OUT/'zero_control_completion.json',dict(episodes=408,records=records,training_updates=0,scope='Nominal-only executedleg command on eachroom finalpolicy; controllerhistory/information same, no parameter tuning or selection.'))
    print('COMPLETED408 registeredzero controls',flush=True)


if __name__=='__main__':run()
