"""等上一轮完成后，串行补测中等采样周期；不并发创建另一组GPU环境。"""
import argparse,copy,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=a.base.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(status='waiting_for_previous_experiment'))
    while not (base/'comparison.json').exists():
        state=json.loads((base/'status.json').read_text())
        if state['status']=='failed':raise RuntimeError('前序实验失败，不自动忽略')
        time.sleep(20)
    write(out/'status.json',dict(status='live_preflight'))
    with (out/'live_preflight.log').open('x') as log:
        probe=subprocess.run([sys.executable,'-u',str(ROOT/'wheelleg_warp/test_native_live.py'),str(ROOT/'wheelleg_warp/results/native_live_preflight_20260921')],stdout=log,stderr=subprocess.STDOUT)
    write(out/'live_preflight_exit.json',dict(returncode=probe.returncode))
    original=json.loads((base/'comparison.json').read_text());assert original['complete']
    protocol=copy.deepcopy(original['protocol']);protocol['rollout_lengths']=[50]
    protocol['purpose']='16步有种子成功率下降，补测50步；同3种子、同预算、同开发集，禁止并行超额实例'
    protocol['source_sha256'][str(Path(__file__).resolve().relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(out/'protocol.json',protocol)
    results=[{**r,'run_directory':str(base/f"rollout_{r['n_steps']}_seed_{r['seed']}")} for r in original['results']]
    for seed in protocol['seeds']:
        name=f'rollout_50_seed_{seed}';write(out/'status.json',dict(status='running',current=name))
        with (out/f'{name}.log').open('x') as log:
            child=subprocess.run([sys.executable,'-u',str(ROOT/'wheelleg_warp/test_convergence_1024.py'),'run','--output',str(out),'--length','50','--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
        if child.returncode:
            write(out/'status.json',dict(status='failed',current=name,returncode=child.returncode));raise RuntimeError(name)
        row=json.loads((out/name/'completed.json').read_text());row['run_directory']=str(out/name);results.append(row)
    write(out/'comparison.json',dict(protocol=protocol,results=results,complete=True))
    write(out/'status.json',dict(status='completed'))
