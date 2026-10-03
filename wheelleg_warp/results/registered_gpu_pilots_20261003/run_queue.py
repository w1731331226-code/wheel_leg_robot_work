"""Sequential dispatch of the existing registered GPU runs; never replays a budget."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def protocol():
    p=read(P/'protocol.json');r=read(P/'readiness.json');registration=read(HERE/'registration.json')
    assert sha(P/'protocol.json')==registration['protocol_sha256']==r['protocol_sha256'] and r['passed']
    for name,value in p['source_sha256'].items():assert sha(ROOT/name)==value,name
    return p,registration

def unit(method,seed):return f"wheelleg-height115-gpu-{'b2v' if method=='B2-V' else method.lower()}-{seed}.service"
def snapshot(name):
    result=subprocess.run(['systemctl','--user','show',name,'-p','LoadState','-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],text=True,capture_output=True)
    return dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
def checkpoint(prefix,record):
    assert sha(str(prefix)+'.zip')==record['checkpoint_sha256']
    assert sha(str(prefix)+'.pkl')==record['normalization_sha256']
def complete(directory,p):
    selection=read(directory/'selection.json');last=read(directory/'last_checkpoint.json');progress=read(directory/'progress.json');cap=p['policy_steps_per_seed']
    assert selection['protocol_sha256']==last['protocol_sha256']==sha(P/'protocol.json')
    assert selection['consumed_policy_steps']==last['consumed_policy_steps']==progress['executed_policy_steps']==cap,'Budget incomplete or unsaved'
    assert selection['evaluation_count']==cap//p['evaluation_interval'] and selection['best'] is not None
    def owned(prefix):assert Path(prefix).resolve().parent==directory.resolve(),'Checkpoint path outside owned run'
    owned(last['path']);checkpoint(last['path'],last);owned(selection['best']['path']);checkpoint(selection['best']['path'],selection['best'])
    records=[]
    for steps in range(p['evaluation_interval'],cap+1,p['evaluation_interval']):
        r=read(directory/f'step_{steps}.json');assert r['policy_steps']==steps
        assert [x['seed'] for x in r['runs']]==[x['seed'] for x in p['selection']]
        assert [x['scenario'] for x in r['runs']]==[x['scenario'] for x in p['selection']]
        owned(r['path']);checkpoint(r['path'],r);records.append(r)
    eligible=[r for r in records if r['summary']['complete']]
    assert eligible and min(eligible,key=lambda r:(r['summary']['total']-r['summary']['success_count'],r['summary']['mean_yaw_score_deg'],r['policy_steps']))==selection['best']
    # Inspect only after the exact budget and all metrics are complete.
    import pickle,numpy as np,torch
    from stable_baselines3 import PPO
    agent=PPO.load(last['path']+'.zip',device='cpu');assert agent.num_timesteps==cap
    assert all(torch.isfinite(v).all() for v in agent.policy.state_dict().values())
    assert agent.policy.optimizer.state and all(torch.isfinite(v).all() for state in agent.policy.optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor))
    with open(last['path']+'.pkl','rb') as f:normalizer=pickle.load(f)
    assert np.isfinite(normalizer.obs_rms.mean).all() and np.isfinite(normalizer.obs_rms.var).all() and np.all(normalizer.obs_rms.var>=0)
    return selection

def save_status(rows,current=None,error=None):
    path=HERE/'queue_status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(protocol_sha256=sha(P/'protocol.json'),completed=rows,current=current,error=error,gate_or_final_simulated=False),indent=2)+'\n');tmp.replace(path)

def run():
    p,registration=protocol();finished=[];current=None
    try:
        for job in registration['queue']:
            method,seed=job['method'],job['seed'];assert method in p['methods'] and seed in p['training_seeds']
            name=unit(method,seed);directory=P/'runs'/method/str(seed);state=snapshot(name);current=dict(**job,unit=name)
            if state.get('ActiveState') in ('active','activating','deactivating'):
                print('ADOPT existing run',current,flush=True)
            elif directory.exists():
                complete(directory,p);finished.append(job);save_status(finished);continue
            else:
                assert state.get('LoadState') in ('not-found',None),'Existing stopped/failed unit needs explicit audit'
                command=['systemd-run','--user','--unit='+name.removesuffix('.service'),'--description=轮腿论文注册GPU预实验'+method+'种子'+str(seed),
                    '--property=MemoryMax=24G','--property=MemorySwapMax=0','--property=TimeoutStopSec=30','--property=KillSignal=SIGINT',
                    '--property=WorkingDirectory='+str(ROOT),'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
                    str(ROOT/'.venv/bin/python'),'-u','-B',str(ROOT/'wheelleg_warp/train_height_comparison.py'),'train','--output',str(P),'--method',method,'--seed',str(seed)]
                subprocess.run(command,check=True);print('START',current,flush=True)
            save_status(finished,current)
            while True:
                state=snapshot(name)
                if state.get('ActiveState') not in ('active','activating','deactivating'):break
                assert int(state.get('MainPID','0'))>0 or state.get('ActiveState') in ('activating','deactivating')
                time.sleep(20)
            assert state.get('ActiveState')!='failed' and state.get('ExecMainStatus','0')=='0',state
            assert int(state.get('MainPID','0'))==0,'Previous process still live'
            p,_=protocol();selected=complete(directory,p);finished.append(job);save_status(finished)
            print('COMPLETE',job,'best',selected['best']['policy_steps'],selected['best']['summary'],flush=True)
        print('REGISTERED QUEUE COMPLETE; research gate remains closed',flush=True)
    except BaseException as exc:
        save_status(finished,current,str(exc));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dispatch',action='store_true');a=parser.parse_args()
    if a.dispatch:run()
    else:print('Read-only entry; pass--dispatch to execute the already registered queue')
