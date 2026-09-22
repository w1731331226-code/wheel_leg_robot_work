"""Reproduce the frozen endpoint comparison, without physics, training or holdout access."""
from pathlib import Path
import hashlib,json,math,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'wheelleg_warp'))
import torch
from stable_baselines3 import PPO
from dashboard.live_env import atomic_json as write

torch.set_num_threads(1)
read=lambda p:json.loads(p.read_text())
baseline=read(HERE/'baseline.json');protocol=read(HERE/'control/protocol.json')
arms={name:read(HERE/name/'targeted_1024000.json') for name in ('control','early')}
a=read(HERE/'control/round_001/run_config.json');b=read(HERE/'early/round_001/run_config.json')
assert not a.pop('terminate_on_attitude_failure') and b.pop('terminate_on_attitude_failure')
assert a==b and a['budget']==1024000
assert all(read(HERE/name/'status.json')['status']=='completed' for name in arms)
assert read(HERE/'engineering.json')['passed']
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protocol['source_sha256'].items())
success=lambda r:r['success'] and r['terrain_evidence_passed']
paired={}
for name,ref in [('source',baseline),('control',arms['control'])]:
    gained=[];lost=[]
    for old,new in zip(ref['runs'],arms['early']['runs']):
        assert old['scenario']==new['scenario'] and old['seed']==new['seed']
        if success(new) and not success(old):gained.append(new['seed'])
        if success(old) and not success(new):lost.append(new['seed'])
    paired[name]=dict(gained=gained,lost=lost)
updates={}
for name,path in [('source',protocol['checkpoint'])]+[(k,str(HERE/k/'round_001/step_1024000')) for k in arms]:
    model=PPO.load(path+'.zip',device='cpu')
    steps=[float(s['step']) for s in model.policy.optimizer.state_dict()['state'].values() if 'step' in s]
    assert len(set(steps))==1
    updates[name]=dict(epochs=model._n_updates,adam_steps=steps[0],std=model.policy.log_std.exp().detach().numpy().tolist())
    if name!='source':
        updates[name]['additional_epochs']=model._n_updates-updates['source']['epochs']
        updates[name]['additional_adam_steps']=steps[0]-updates['source']['adam_steps']
        logs=[json.loads(x) for x in (HERE/name/'round_001/progress.json').read_text().splitlines()]
        updates[name]['logged_cycles']=len(logs)
        updates[name]['last_logged_train_metrics']={k:v for k,v in logs[-1].items() if k.startswith('train/')}
threshold=baseline['summary']['step']['success']+math.ceil(.20*48)
candidate=arms['early']['summary']
passed=candidate['step']['success']>=max(threshold,math.ceil(.85*48)) and candidate['step']['complete']>=math.ceil(.95*48)
passed=passed and all(candidate[k]['success']>=baseline['summary'][k]['success'] for k in ('legacy','surface','advanced'))
result=dict(engineering_passed=True,paired_run_config_verified=True,source_hashes_verified=True,
    summaries={'source':baseline['summary'],**{k:v['summary'] for k,v in arms.items()}},
    required_step_success=threshold,required_step_complete=math.ceil(.95*48),development_gate_passed=passed,
    paired_early_vs=paired,optimizer=updates,promoted=False,holdout_evaluated=False,formal_training_ready=False,
    conclusion='Early failure feedback alone did not deliver a large success gain and reduced completion; reject this candidate. Task-deadline terminal semantics fixed independently in both arms.')
write(HERE/'verification.json',result)
print(json.dumps({k:result[k] for k in ('required_step_success','development_gate_passed','paired_early_vs','optimizer')},ensure_ascii=False),flush=True)
