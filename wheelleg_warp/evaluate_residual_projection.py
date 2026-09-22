"""Fixed-policy AB/BA regression; no training, new action space or holdout access."""
from pathlib import Path
import argparse,hashlib,json,math,os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import torch
from stable_baselines3 import PPO
from native.terrain import TerrainScenario
from terrain_eval import evaluate_terrain
from optimize_residual import summary
from dashboard.live_env import atomic_json as write


def run(output):
    assert json.loads((output/'engineering_final.json').read_text())['passed']
    source=ROOT/'wheelleg_warp/results/failure_feedback_20260922/control/protocol.json'
    previous=json.loads(source.read_text());cases=[TerrainScenario(**s) for s in previous['development_cases']];labels=previous['labels']
    files=[Path(__file__),ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/terrain_eval.py']
    checkpoint=previous['checkpoint'];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    protocol=dict(checkpoint=checkpoint,checkpoint_sha256=hashlib.sha256(Path(checkpoint+'.zip').read_bytes()).hexdigest(),
        development_cases=previous['development_cases'],labels=labels,orders=[['legacy','projected'],['projected','legacy']],
        required_success_floor=44,minimum_gain_pp=20,minimum_complete=46,original_regression='no lost success in other three groups',
        source_sha256=hashes,trained=False,holdout_evaluated=False,default_changed=False,
        intervention='Allow original-direction common-lambda residual on clipped base; preserve all torque/speed, mapping and nonfinite guards.')
    if (output/'protocol.json').exists():raise FileExistsError('Refusing to overwrite frozen protocol')
    write(output/'protocol.json',protocol);torch.set_num_threads(1);model=PPO.load(checkpoint+'.zip',device='cpu')
    initial={k:v.clone() for k,v in model.policy.state_dict().items()};pairs=[]
    for repeat,order in enumerate(protocol['orders'],1):
        runs={}
        for mode in order:
            write(output/'status.json',dict(status='evaluating',repeat=repeat,mode=mode,trained=False))
            rows=evaluate_terrain(model,checkpoint+'.pkl',cases,{'project_clipped_base':mode=='projected'})
            result=dict(summary=summary(rows,labels),runs=rows,project_clipped_base=mode=='projected')
            write(output/f'{mode}_{repeat}.json',result);runs[mode]=result
            print(repeat,mode,result['summary'],flush=True)
        gained=[];lost=[]
        for a,b in zip(runs['legacy']['runs'],runs['projected']['runs']):
            assert a['scenario']==b['scenario'] and a['seed']==b['seed']
            old=a['success'] and a['terrain_evidence_passed'];new=b['success'] and b['terrain_evidence_passed']
            if new and not old:gained.append(b['seed'])
            if old and not new:lost.append(b['seed'])
        baseline=runs['legacy']['summary'];candidate=runs['projected']['summary']
        required=max(44,baseline['step']['success']+math.ceil(.2*48))
        passed=candidate['step']['success']>=required and candidate['step']['complete']>=46
        for k in ('legacy','surface','advanced'):passed=passed and candidate[k]['success']>=baseline[k]['success']
        nonstep_seeds={s.terrain_seed for s,k in zip(cases,labels) if k!='step'}
        passed=passed and not (set(lost)&nonstep_seeds)
        pairs.append(dict(repeat=repeat,baseline=baseline,candidate=candidate,gained=gained,lost=lost,required_success=required,passed=passed))
    assert all(torch.equal(initial[k],v) for k,v in model.policy.state_dict().items())
    result=dict(pairs=pairs,development_gate_passed=all(p['passed'] for p in pairs),weights_unchanged=True,
        promoted=False,trained=False,holdout_evaluated=False,default_changed=False,
        conclusion='Only both repeated strong gates permit independent validation; no promotion from mean-angle or lambda improvement.')
    write(output/'verification.json',result);write(output/'status.json',dict(status='completed',promoted=False,trained=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
