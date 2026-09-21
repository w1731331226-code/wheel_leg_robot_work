"""同GPU物理后端比较批量控制/奖励链，包含自动重置；不替代CPU物理配对门。"""
from pathlib import Path
import sys,json
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]
import numpy as np
from ppo_env import Scenario
from fast_physics import FastWarpEnv
from native.environment import NativeEnv

if __name__=='__main__':
    scenarios=[Scenario(height_l=.012,height_r=.006,mu_l=.7,mu_r=.9),
               Scenario(speed=-.7,mass=7.3,height_l=.008,height_r=.015,mu_l=.8,mu_r=.7,delay_ms=10.)]
    rows=[]
    for scenario in scenarios:
        ref=FastWarpEnv(scenario=scenario);obs,_=ref.reset(seed=1609)
        env=NativeEnv(1,scenario=scenario);actual=env.reset()
        peaks=dict(obs=0.,reward=0.,qpos=0.,qvel=0.);episodes=0;mismatches=[]
        for step in range(850):
            action=np.array([.1,-.1,.1],dtype=np.float32)
            expected,reward,terminated,truncated,info=ref.step(action)
            actual,rewards,done,infos=env.step(action[None,:])
            if bool(done[0])!=bool(terminated or truncated):
                mismatches.append(dict(step=step,reference=info['reason'],native=infos[0].get('reason')))
                break
            peaks['reward']=max(peaks['reward'],abs(reward-float(rewards[0])))
            native_obs=infos[0]['terminal_observation'] if done[0] else actual[0]
            peaks['obs']=max(peaks['obs'],float(np.max(abs(expected-native_obs))))
            if not done[0]:
                peaks['qpos']=max(peaks['qpos'],float(np.max(abs(ref.env.data.qpos-env.data.qpos.numpy()[0]))))
                peaks['qvel']=max(peaks['qvel'],float(np.max(abs(ref.env.data.qvel-env.data.qvel.numpy()[0]))))
            else:
                episodes+=1
                if info['reason']!=infos[0]['reason'] or info['success']!=infos[0]['success']:
                    mismatches.append(dict(step=step,reference=info['reason'],native=infos[0]['reason']))
                expected,_=ref.reset()
                peaks['obs']=max(peaks['obs'],float(np.max(abs(expected-actual[0]))))
        ref.close();env.close()
        rows.append(dict(scenario=scenario.__dict__,peaks=peaks,episodes=episodes,mismatches=mismatches))
    # 沿用既有物理工程门；观测/奖励另设显式门，不把数值漂移写成逐位一致。
    passed=all(not r['mismatches'] and r['episodes']>=2 and r['peaks']['qpos']<=.02 and r['peaks']['qvel']<=.5 and r['peaks']['obs']<=.5 and r['peaks']['reward']<=.05 for r in rows)
    result=dict(passed=passed,rows=rows,thresholds=dict(qpos=.02,qvel=.5,obs=.5,reward=.05))
    p=Path(sys.argv[1]);assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    assert passed,'GPU原生闭环配对失败，不得标记等价通过'
