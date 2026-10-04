"""One registered fixed-model confirmation; no learning, tuning or silent resume."""
from pathlib import Path
import argparse,json
from stable_baselines3 import PPO
from train_height_comparison import protocol,evaluate,summary
from check_residual_channels import MaskedPolicy
from smoke_reward_training import weight_digest
from training_contract import digest
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
S=ROOT/'wheelleg_warp/results/paper_recovery_20261004/channel_validation_v1'
FILES=['evaluate_channel_validation.py','check_residual_channels.py','smoke_reward_training.py']

def freeze():
    base=protocol(BASE);assert not (S/'source_contract.json').exists()
    atomic_json(S/'source_contract.json',dict(protocol_sha256=digest(S/'protocol.json'),
        source_sha256={**base['source_sha256'],**{str(Path('wheelleg_warp')/n):digest(ROOT/'wheelleg_warp'/n) for n in FILES}},
        evaluator='Existing frozen original task evaluator, fixed mean requests with0/1mask; pressure factor rows reported separately',
        lifecycle='No retries or model/case selection; independent namespace untouched before source freeze',training_steps=0))
    print('FROZEN channel validation',digest(S/'source_contract.json'),flush=True)

def run():
    spec=json.loads((S/'protocol.json').read_text());contract=json.loads((S/'source_contract.json').read_text())
    assert contract['protocol_sha256']==digest(S/'protocol.json')
    assert all(digest(ROOT/n)==value for n,value in contract['source_sha256'].items())
    out=S/'results';out.mkdir(exist_ok=False)
    completed=0;attempt=None;records=[]
    pressure=[v for rows in spec['pressure'].values() for v in rows]
    sets={'regular':spec['regular'],'pressure':pressure,'legacy_regression':spec['legacy_regression']}
    controllers=[dict(label=k,candidate=v,model=None,mask=None) for k,v in spec['fixed_classical'].items()]
    for model in spec['models']:
        for name,mask in spec['masks'].items():controllers.append(dict(label=f"{model['arm']}_{model['seed']}_{name}",candidate=None,model=model,mask=mask))
    try:
        for controller in controllers:
            model=controller['model'];agent=None;normalization=None;before=None
            if model:
                prefix=Path(model['prefix'])
                assert digest(prefix.with_suffix('.zip'))==model['checkpoint']['checkpoint_sha256']
                assert digest(prefix.with_suffix('.pkl'))==model['checkpoint']['normalization_sha256']
                loaded=PPO.load(str(prefix)+'.zip',device='cuda');before=(loaded.num_timesteps,loaded._n_updates,weight_digest(loaded))
                assert before[:2]==(200000,400)
                agent=MaskedPolicy(loaded,controller['mask']);normalization=str(prefix)+'.pkl'
            for category,cases in sets.items():
                label=controller['label'];attempt=dict(controller=label,category=category,attempted_cases=len(cases))
                atomic_json(S/'progress.json',dict(status='running',completed_episodes=completed,budget=spec['evaluation_episodes_budget'],pending_job=attempt))
                assert all(digest(ROOT/n)==value for n,value in contract['source_sha256'].items())
                rows=evaluate(cases,'diff3',agent,normalization,controller['candidate'])
                assert len(rows)==len(cases) and [r['seed'] for r in rows]==[r['seed'] for r in cases]
                if model:assert before==(loaded.num_timesteps,loaded._n_updates,weight_digest(loaded))
                stats=summary(rows);factors={}
                if category=='pressure':
                    for name,selected in spec['pressure'].items():
                        ids={s['seed'] for s in selected};part=[r for r in rows if r['seed'] in ids]
                        factors[name]=dict(summary=summary(part),physical=sum(r['physical_safety_passed'] for r in part),design=sum(r['design_joint_passed'] for r in part))
                result=dict(controller=controller,category=category,summary=stats,physical=sum(r['physical_safety_passed'] for r in rows),
                    design=sum(r['design_joint_passed'] for r in rows),terrain_missing=sum(not r['terrain_passed'] for r in rows),factors=factors,runs=rows,
                    protocol_sha256=contract['protocol_sha256'],source_contract_sha256=digest(S/'source_contract.json'))
                path=out/f'{label}_{category}.json';atomic_json(path,result)
                completed+=len(rows);assert completed<=spec['evaluation_episodes_budget']
                records.append(dict(path=str(path.relative_to(S)),sha256=digest(path),episodes=len(rows)))
                atomic_json(S/'completed_jobs.json',dict(completed_episodes=completed,records=records))
                print('COMPLETED',completed,label,category,stats,'physical/design',result['physical'],result['design'],flush=True)
                attempt=None
        assert completed==spec['evaluation_episodes_budget']==4472 and len(records)==78
        atomic_json(S/'completion.json',dict(verified_producer=True,completed_episodes=completed,jobs=len(records),training_steps=0,
            all_fixed_controllers_and_categories_complete=True,protocol_sha256=contract['protocol_sha256'],source_contract_sha256=digest(S/'source_contract.json'),
            scope='Finite fixed-model independent cases. Normalization unchanged and original gates; old regression nonindependent; results not policy promotion or universal learning claim.'))
        atomic_json(S/'progress.json',dict(status='complete',completed_episodes=completed,budget=4472))
    except BaseException as error:
        atomic_json(S/'interruption.json',dict(status='incomplete',confirmed_completed_episodes=completed,pending_job=attempt,
            pending_job_consumption_unknown=True,error_type=type(error).__name__,error=str(error),silently_resumable=False))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');a=parser.parse_args()
    freeze() if a.freeze else run()
