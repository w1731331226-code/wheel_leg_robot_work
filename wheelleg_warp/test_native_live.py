"""核对只读采集与全环境快照；随后运行真实1024 PPO与动画集成预检。"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,struct,time,gc,subprocess,signal
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from native.environment import NativeEnv
from native.live import LiveNativeEnv,SELECTED
from benchmark_parallel import TimedPPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
from dashboard.live_env import atomic_json as write

if __name__=='__main__':
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    SELECTED.parent.mkdir(parents=True,exist_ok=True);write(SELECTED,{'environment':0})
    reference=NativeEnv(8,seed=730000);recorded=LiveNativeEnv(output/'paired/live',n=8,seed=730000,phase='engineering_probe')
    reference.reset();recorded.reset();peaks=dict(observation=0.,reward=0.);episodes=0
    for i in range(420):
        action=np.tile([.1,-.1,.1],(8,1)).astype(np.float32)
        a=reference.step(action);b=recorded.step(action)
        np.testing.assert_array_equal(a[2],b[2]);episodes+=int(b[2].sum())
        peaks['observation']=max(peaks['observation'],float(np.max(abs(a[0]-b[0]))));peaks['reward']=max(peaks['reward'],float(np.max(abs(a[1]-b[1]))))
    assert peaks['observation']<.02 and peaks['reward']<.001 and episodes>=8,peaks
    # 查看者切换到世界3时，必须返回世界3真实状态，不得只更改标签。
    write(SELECTED,{'environment':3});recorded.last_select=0;recorded.step(np.zeros((8,3),np.float32))
    meta=json.loads((output/'paired/live/latest.json').read_text());assert meta['environment_index']==3
    with np.load(output/'paired/live/latest.npz') as frames:
        np.testing.assert_allclose(frames['qpos'][-1],recorded.data.qpos.numpy()[3],atol=1e-6)
    recorded.save_overview();packet=(output/'paired/live/overview.bin').read_bytes();length=struct.unpack('<I',packet[:4])[0];header=json.loads(packet[4:4+length]);values=np.frombuffer(packet[4+length:],dtype='<f4').reshape(header['environments'],header['width'])
    assert header['environments']==8 and len(header['wheels'])==2 and np.isfinite(values).all()
    np.testing.assert_array_equal(values[:,12:12+3*header['bodies']],recorded.data.xpos.numpy().reshape(8,-1))
    write(output/'paired.json',dict(passed=True,peaks=peaks,episodes=episodes,selected_world_verified=3,overview_worlds=8))
    reference.close();recorded.close();del reference,recorded;gc.collect();write(SELECTED,{'environment':0})
    cfg=json.loads((ROOT/'wheelleg_ppo/tools/results/yaw_precision_v2_2026-09-21/training_config.json').read_text())
    hashes={k:v for k,v in json.loads((ROOT/'wheelleg_warp/CPU_REFERENCE.json').read_text())['source_sha256'].items() if not k.endswith('/resume_yaw.py')}
    write(output/'protocol.json',dict(source_sha256=hashes,phase='preflight_training',environments=1024))
    torch.set_num_threads(1);directory=output/'round_000';directory.mkdir()
    raw=LiveNativeEnv(directory/'live',n=1024,seed=730000,phase='preflight_training')
    env=VecNormalize(VecCheckNan(raw,raise_exception=True),**cfg['normalization']);kwargs=dict(cfg['ppo']);kwargs['n_steps']=16
    model=TimedPPO('MlpPolicy',env,seed=1609,device='cpu',**kwargs);model.timings=[]
    initial={k:v.clone() for k,v in model.policy.state_dict().items()}
    with (output/'renderer.log').open('w') as log:
        renderer=subprocess.Popen([sys.executable,'-u',str(ROOT/'wheelleg_warp/dashboard/record.py'),'--run-root',str(output)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            started=time.perf_counter();model.learn(total_timesteps=524288);elapsed=time.perf_counter()-started
            assert all(torch.isfinite(v).all() for v in model.policy.state_dict().values())
            assert any(not torch.equal(initial[k],v) for k,v in model.policy.state_dict().items())
            live=json.loads((directory/'live/latest.json').read_text());assert live['environments']==1024
            packet=(directory/'live/overview.bin').read_bytes();length=struct.unpack('<I',packet[:4])[0];meta=json.loads(packet[4:4+length]);assert meta['environments']==1024
            write(output/'training.json',dict(passed=True,policy_steps=model.num_timesteps,seconds=elapsed,policy_sps=model.num_timesteps/elapsed,real_recording=True,overview_worlds=1024))
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                captures=list((ROOT/'wheelleg_warp/dashboard/local_data/captures/native').glob(output.name+'__*/metadata.json'))
                if captures and all((p.parent/'animation_50.webp').exists() or (p.parent/'export_failed.json').exists() for p in captures):break
                time.sleep(2)
            assert captures and all((p.parent/'animation_50.webp').exists() for p in captures),'真实录像未成功生成'
            write(output/'completed.json',dict(passed=True,paired=peaks,policy_steps=model.num_timesteps,captures=len(captures),source='actual_training_frames'))
        finally:
            try:os.killpg(renderer.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            renderer.wait(timeout=20);env.close()
