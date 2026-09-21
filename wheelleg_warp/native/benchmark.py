"""完整GPU控制/物理/奖励/观测吞吐，含真实终止与重置，不是力矩回放。"""
from pathlib import Path
import sys,argparse,json,time,threading,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from native.environment import NativeEnv

p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
env=NativeEnv(a.worlds,stage=3);env.reset()
actions=np.tile([.1,-.1,.1],(a.worlds,1)).astype(np.float32)
for _ in range(10):env.step(actions)
samples=[];stop=threading.Event()
def gpu_samples():
    while not stop.is_set():
        try:
            line=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip()
            samples.append([float(x.strip()) for x in line.split(',')])
        except Exception:pass
        stop.wait(1)
thread=threading.Thread(target=gpu_samples,daemon=True);thread.start()
rounds=[];ends=0
for repeat in range(3):
    started=time.perf_counter()
    for i in range(200):
        o,r,d,infos=env.step(actions);ends+=int(d.sum())
    rounds.append(time.perf_counter()-started)
stop.set();thread.join(timeout=2)
result=dict(worlds=a.worlds,policy_steps_per_repeat=a.worlds*200,seconds=rounds,
            median_policy_sps=a.worlds*200/float(np.median(rounds)),episodes_finished=ends,
            gpu_samples=samples,gpu_mean=np.mean(samples,axis=0).tolist() if samples else None,
            scope='真实控制/物理/奖励/观测/重置；不含PPO梯度；GPU利用率含同机其他任务')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False),flush=True)
