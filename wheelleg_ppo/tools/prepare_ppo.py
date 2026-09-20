"""Reproducible pre-training evidence: baseline, pulse responses, rollout/update throughput."""
import argparse
import contextlib
import copy
from dataclasses import asdict
from functools import partial
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

from ppo_env import (MODES, Residual, Scenario, WheelLegEnv, build_model, sample_scenario, TORQUE_SCALE)
import wheelleg_sim as sim

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tools/results/preppo_2026-09-16'


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def evaluate(item):
    name, s = item
    with contextlib.redirect_stdout(io.StringIO()):
        env = WheelLegEnv(scenario=s)
        env.reset(seed=0)
        while not env.done:
            env.step(np.zeros(3, dtype=np.float32))
    result = dict(name=name, scenario=asdict(s), **env.metrics())
    print(name, result['success'], result['reason'], result['peak_deg'], result['velocity_rmse'], flush=True)
    return result


def baseline():
    cases = [(f'nominal_{v}', Scenario(speed=v)) for v in (-1., -.5, .5, 1.)]
    cases += [(f'symmetric_{v}', Scenario(speed=v,height_l=.015,height_r=.015)) for v in (-1.,1.)]
    cases += [(f'legacy_bump_{v}_{side}_{h}', Scenario(speed=v, **{f'height_{side}':h}))
              for v in (-1.,1.) for side in ('l','r') for h in (.01,.015)]
    cases += [(f'validation_{seed}',sample_scenario('validation',seed)) for seed in range(12)]
    cases += [('development_friction', Scenario(mu_l=.6,mu_r=1.)),
              ('development_solver',Scenario(height_l=.015,solver_iterations=50))]
    # No final test set is evaluated here. Difficult validation is selected before RL training.
    write('development_scenarios.json', [{'name':n,'scenario':asdict(s)} for n,s in cases])
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(evaluate,cases))
    write('baseline.json',dict(method='B0',protocol='preppo-v2-ramp',test_set_evaluated=False,
                              success_count=sum(r['success'] for r in rows),total=len(rows),runs=rows))


def snapshot(s, condition):
    model = build_model(s); data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.keyframe('stand').id); mujoco.mj_forward(model,data)
    state = sim.make_state(model,True,True)
    wheels = {model.geom('wheel_collide_'+side).id for side in ('L','R')}
    encountered = False
    for step in range(12000):
        state.cmd_vel = s.speed if step >= 2000 else 0.
        sim.control(model,data,state); mujoco.mj_step(model,data)
        contacted=set()
        for c in data.contact:
            pair={c.geom1,c.geom2}; contacted |= pair & wheels
            encountered |= any((model.geom(g).name or '').startswith('bump_') for g in pair)
        lengths=[sim.fk_joints(data.qpos[a],data.qpos[b])['leg_len'] for a,b in ((7,10),(12,15))]
        if ((condition=='flat' and step==3999)
            or (condition=='unequal_height' and encountered and abs(lengths[0]-lengths[1])>.005)
            or (condition=='single_contact' and encountered and len(contacted)==1)):
            return model,data,state,dict(time_s=float(data.time),lengths_m=lengths,
                                        contact_wheels=[model.geom(g).name for g in sorted(contacted)])
    raise AssertionError(f'Cannot reach pulse condition: {condition}')


def pulses():
    rows=[]
    for condition,s in [('flat',Scenario()),('unequal_height',Scenario(height_l=.02,center=1.5)),
                        ('single_contact',Scenario(height_l=.02,center=1.5))]:
        model,start,state,context=snapshot(s,condition)
        runs={}
        for channel,sign in [(-1,0)]+[(c,sig) for c in range(3) for sig in (-1,1)]:
            data=mujoco.MjData(model); mujoco.mj_copyData(data,model,start)
            trial=copy.deepcopy(state); residual=Residual('diff3')
            trace=[]
            for step in range(600):
                target=np.zeros(3)
                if channel>=0 and step<40: target[channel]=.1*sign
                residual.set_action(target)
                sim.control(model,data,trial,residual); mujoco.mj_step(model,data)
                trace.append([step*model.opt.timestep,*sim.euler(data),sim.forward_component(data.qvel,sim.euler(data)[2]),
                              residual.lam,*data.ctrl])
            key=f'{channel}_{sign}'; runs[key]=np.array(trace)
        np.savez_compressed(OUT/f'pulse_{condition}.npz',**runs)
        reference=runs['-1_0']
        for key,trace in runs.items():
            if key=='-1_0': continue
            delta=trace[:,1:5]-reference[:,1:5]
            rows.append(dict(condition=condition,context=context,channel_sign=key,
                             peak_abs_delta_roll_pitch_yaw_deg=np.degrees(np.max(abs(delta[:,:3]),axis=0)).tolist(),
                             delta_at_20ms=delta[39].tolist(),delta_at_300ms=delta[-1].tolist(),
                             max_abs_velocity_delta=float(max(abs(delta[:,3]))),
                             min_lambda=float(min(trace[:,5]))))
        # Same state, same N(0,1) clipped initial action sampling for every parameterization.
        distributions=[]
        for mode,n in MODES.items():
            residual=Residual(mode); mapped=[]; lambdas=[]
            low,high=model.actuator_ctrlrange.T.copy()
            for aid in range(model.nu):
                speed=start.qvel[model.jnt_dofadr[model.actuator_trnid[aid,0]]]
                allowed=sim.hw.torque_limit(float('inf'),speed,aid<4,0,model.opt.timestep)[0]
                low[aid]=max(low[aid],-allowed); high[aid]=min(high[aid],allowed)
            base=np.clip(start.ctrl,low,high)
            scratch=mujoco.MjData(model); mujoco.mj_copyData(scratch,model,start)
            for action in np.clip(np.random.default_rng(610).normal(size=(256,n)),-1,1):
                residual.target[:]=residual.action[:]=action; scratch.ctrl[:]=base
                residual.apply(model,scratch,state,low,high)
                mapped.append(residual.executed.copy()); lambdas.append(residual.lam)
            mapped=np.array(mapped)
            distributions.append(dict(mode=mode,per_motor_rms_Nm=np.sqrt(np.mean(mapped**2,axis=0)).tolist(),
                                       normalized_rms=float(np.sqrt(np.mean((mapped/TORQUE_SCALE)**2))),
                                       saturation_fraction=float(np.mean(np.array(lambdas)<1-1e-10))))
        write(f'initial_distribution_{condition}.json',dict(context=context,distributions=distributions,
              note='Steady held targets; same clipped N(0,1) sampling. Reachable sets are intentionally different.'))
    write('pulses.json',dict(amplitude_normalized=.1,pulse_s=.02,observation_s=.3,runs=rows))


def benchmark():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    torch.set_num_threads(1)
    versions={}
    from importlib.metadata import version
    for name in ('numpy','scipy','mujoco','gymnasium','stable-baselines3','torch'):
        versions[name]=version(name)
    env=WheelLegEnv(scenario=Scenario())
    started=time.perf_counter(); env.reset(seed=1); first_reset=time.perf_counter()-started
    times=[]
    for i in range(3):
        started=time.perf_counter(); env.reset(seed=i); times.append(time.perf_counter()-started)
    started=time.perf_counter()
    for _ in range(100): env.step(np.zeros(3,dtype=np.float32))
    single_sps=100/(time.perf_counter()-started)
    rows=[]
    for count in (1,4,8):
        factories=[partial(WheelLegEnv,stage=1) for _ in range(count)]
        raw=DummyVecEnv(factories) if count==1 else SubprocVecEnv(factories,start_method='spawn')
        vec=VecNormalize(raw,norm_obs=True,norm_reward=False)
        vec.seed(1609); vec.reset()
        # A single small PPO update is an interface/throughput test, not a learning result.
        model=PPO('MlpPolicy',vec,n_steps=256,batch_size=64,n_epochs=2,device='cpu',seed=1609,
                  policy_kwargs=dict(net_arch=[64,64]),verbose=0)
        optimizer_times=[]
        train=model.train
        def timed_train():
            start=time.perf_counter(); train(); optimizer_times.append(time.perf_counter()-start)
        model.train=timed_train
        started=time.perf_counter(); model.learn(total_timesteps=256*count)
        total=time.perf_counter()-started
        opt=sum(optimizer_times)
        rows.append(dict(environments=count,policy_steps=256*count,nominal_physical_steps=256*count*40,
                         total_s=total,update_s=opt,rollout_and_overhead_s=total-opt,
                         aggregate_steps_per_s=256*count/total))
        if count==1:
            model.train=train
            model.save(OUT/'smoke_policy')
            vec.save(OUT/'smoke_normalization.pkl')
            vec.training=False
            observation=vec.reset(); before=model.predict(observation,deterministic=True)[0]
            reloaded=PPO.load(OUT/'smoke_policy',device='cpu')
            np.testing.assert_array_equal(before,reloaded.predict(observation,deterministic=True)[0])
            stats=vec.obs_rms.mean.copy(); vec.step(before)
            np.testing.assert_array_equal(stats,vec.obs_rms.mean)
            # Reload normalization separately, freeze it before any evaluation steps.
            loaded=VecNormalize.load(OUT/'smoke_normalization.pkl',DummyVecEnv([partial(WheelLegEnv,stage=1)]))
            loaded.training=False; loaded.norm_reward=False
            vec.seed(225); loaded.seed(225)
            np.testing.assert_array_equal(vec.reset(),loaded.reset())
            stats=loaded.obs_rms.mean.copy(); loaded.reset(); loaded.step(np.zeros((1,3),dtype=np.float32))
            np.testing.assert_array_equal(stats,loaded.obs_rms.mean); loaded.close()
        vec.close()
        print('benchmark',rows[-1],flush=True)
    smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True)
    write('benchmark.json',dict(versions=versions,python=platform.python_version(),cpu_count=os.cpu_count(),
        first_reset_s=first_reset,cached_reset_median_s=float(np.median(times)),
        single_nominal_rollout_steps_per_s=single_sps,torch_cuda_available=torch.cuda.is_available(),
        nvidia_smi_returncode=smi.returncode,nvidia_smi=smi.stdout+smi.stderr,
        runs=rows,save_reload_and_frozen_normalization_pass=True,
        note='CPU smoke updates only; no convergence claim. GPU not benchmarked without working driver.'))


def training_gate(device='cpu'):
    """Two rollout/update cycles for M and B2-V, with the full randomized distribution."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecCheckNan
    torch.set_num_threads(1)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，不能自动回退CPU并宣称GPU验收通过')
    if device == 'cuda':
        tensor=torch.ones(256,device='cuda',requires_grad=True)
        tensor.square().sum().backward()
        assert torch.all(tensor.grad == 2)
        torch.cuda.synchronize()
    rows=[]
    for mode in ('diff3','virtual6'):
        raw=SubprocVecEnv([partial(WheelLegEnv,mode=mode,stage=3) for _ in range(8)],start_method='spawn')
        vec=VecNormalize(VecCheckNan(raw,raise_exception=True),norm_reward=False)
        vec.seed(1609)
        model=PPO('MlpPolicy',vec,n_steps=256,batch_size=256,n_epochs=10,device=device,seed=1609,
                  policy_kwargs=dict(net_arch=[64,64]),verbose=0)
        assert model.device.type == device
        initial={key:value.clone() for key,value in model.policy.state_dict().items()}
        if device == 'cuda': torch.cuda.synchronize()
        started=time.perf_counter(); model.learn(total_timesteps=4096)
        if device == 'cuda': torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        assert all(torch.isfinite(value).all() for value in model.policy.state_dict().values())
        assert any(not torch.equal(initial[key],value) for key,value in model.policy.state_dict().items())
        losses={key:float(value) for key,value in model.logger.name_to_value.items()
                if key.startswith('train/') and np.isscalar(value)}
        assert all(np.isfinite(value) for value in losses.values())
        model.save(OUT/f'gate_{mode}_policy'); vec.save(OUT/f'gate_{mode}_normalization.pkl')
        vec.training=False
        observation=vec.reset()
        loaded=PPO.load(OUT/f'gate_{mode}_policy',device=device)
        np.testing.assert_array_equal(model.predict(observation,deterministic=True)[0],
                                      loaded.predict(observation,deterministic=True)[0])
        portable=PPO.load(OUT/f'gate_{mode}_policy',device='cpu')
        np.testing.assert_allclose(model.predict(observation,deterministic=True)[0],
                                   portable.predict(observation,deterministic=True)[0],rtol=1e-5,atol=1e-6)
        stats=vec.obs_rms.mean.copy(); vec.step(loaded.predict(observation,deterministic=True)[0])
        np.testing.assert_array_equal(stats,vec.obs_rms.mean)
        rows.append(dict(mode=mode,device=str(model.device),passed=True,policy_steps=4096,updates=2,epochs_per_update=10,
                         seconds=elapsed,policy_steps_per_s=4096/elapsed,losses=losses,
                         peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                         note='Smoke policy only; not a usable controller or convergence result.'))
        vec.close()
        print('training gate',mode,'PASS',elapsed,flush=True)
    commands={}
    for name,args in [('memory',['free','-b']),('disk',['df','-B1',str(ROOT)]),
                      ('cpu',['lscpu']),('kernel',['uname','-a']),('gpu',['nvidia-smi'])]:
        result=subprocess.run(args,capture_output=True,text=True)
        commands[name]=dict(returncode=result.returncode,output=result.stdout+result.stderr)
    write('training_gate.json',dict(passed=True,device=device,environments=8,runs=rows,host=commands,
          formal_training_started=False,cuda_build=torch.version.cuda,torch_version=torch.__version__,
          gpu_name=torch.cuda.get_device_name(0) if device=='cuda' else None,
          gpu_capability=torch.cuda.get_device_capability(0) if device=='cuda' else None,
          cuda_basic_autograd_pass=device=='cuda'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('baseline','pulses','benchmark','training_gate'))
    parser.add_argument('--require-success', action='store_true', help='baseline全部开发场景成功才返回0')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    if args.device != 'cpu' and args.task != 'training_gate':
        parser.error('--device cuda只适用于training_gate')
    if args.require_success and args.task != 'baseline':
        parser.error('--require-success只适用于baseline')
    OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.task == 'training_gate':
        training_gate(args.device)
    else:
        globals()[args.task]()
    write('source_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [*sorted((ROOT/'tools').glob('*.py')),ROOT/'xml/wheelleg.xml',ROOT/'requirements.txt']})

    if args.require_success:
        result=json.loads((OUT/'baseline.json').read_text())
        assert result['success_count'] == result['total'], '开发场景未全部通过，见baseline.json'
