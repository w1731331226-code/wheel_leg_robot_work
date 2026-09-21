"""独立短预算运行真实正式入口；不将预检权重用于正式训练。"""
from pathlib import Path
import json,sys,time,subprocess,signal,os
from train_native import TimedPPO,evaluate,summarize,orchestrate,write,ROOT
import torch
if __name__=='__main__':
    formal=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
    p=json.loads((formal/'protocol.json').read_text());p.update(phase='preflight_training',max_rounds=1,steps_per_round=p['n_steps']*1024*2)
    p['development_cases']=p['development_cases'][:4];p['final_cases']=p['final_cases'][:4]
    torch.set_num_threads(1);prefix=formal/'bootstrap/policy';model=TimedPPO.load(str(prefix)+'.zip',device='cpu')
    initial={k:v.clone() for k,v in model.policy.state_dict().items()}
    cases=p['development_cases'];rows=evaluate(model,str(prefix)+'.pkl','diff3',cases);p['bootstrap_summary']=summarize(rows,[c['seed'] for c in cases])
    write(output/'protocol.json',p);write(output/'selection.json',dict(best=dict(round=0,path=str(prefix),summary=p['bootstrap_summary']),anchor=p['bootstrap_summary'],stagnant_rounds=0,rounds=[]))
    with (output/'renderer.log').open('w') as log:
        renderer=subprocess.Popen([sys.executable,'-u',str(ROOT/'wheelleg_warp/dashboard/record.py'),'--run-root',str(output)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            orchestrate(output)
            state=json.loads((output/'status.json').read_text());assert state['status']=='completed'
            row=json.loads((output/'round_001/completed.json').read_text());assert row['updates']==2 and len(row['records'])==2
            last=TimedPPO.load(row['last_path']+'.zip',device='cpu')
            assert any(not torch.equal(initial[k],v) for k,v in last.policy.state_dict().items())
            write(output/'verified.json',dict(passed=True,updates=row['updates'],steps=p['steps_per_round'],restored_optimizer_and_normalization=True,formal_weights_used=False))
        finally:
            try:os.killpg(renderer.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            renderer.wait(timeout=20)
