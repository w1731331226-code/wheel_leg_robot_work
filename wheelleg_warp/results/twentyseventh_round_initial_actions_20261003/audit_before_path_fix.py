"""Same-state initial PPO exploration audit; calibrate log_std, never train a policy."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import warp as wp
import gymnasium as gym
from scipy.optimize import least_squares
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario
from native.controller import control_physical_nominal,control_physical_nominal_torque6,D
from native.shared_reference import load,update
from state_estimation import leg_kinematics
import wheelleg_sim as sim
from training_contract import ROOT,source_hashes,digest

MODES=('diff3','virtual6','torque6')
SCALE=np.array([40.,40.,40.,40.,4.5,4.5])
UPPER=np.array([1.,1.,1.,1.,1.05,1.05])


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def collect(out,reg):
    parent=out.parent/'twentyfourth_round_operating_regions_20261002'
    paths=[parent/'broad_v2/registered_cases.json',parent/'random_v2/registered_cases.json']
    lists=[json.loads(p.read_text())['scenarios'] for p in paths];scenes=sum(lists,[])
    assert [len(s) for s in lists]==[84,64]
    env=NativeEnv.height115_candidate(n=len(scenes),scenario=[HeightTerrainScenario(**s) for s in scenes],shared_reference=True)
    saved=[];completed=[None]*len(scenes);seen=[set() for _ in scenes]
    def capture(selected):
        arrays=dict(q=env.data.qpos.numpy(),v=env.data.qvel.numpy(),sensor=env.data.sensordata.numpy(),
            memory=env.k['state'].numpy(),task=env.state.numpy(),param=env.param.numpy(),obs=env.obs.numpy(),
            command=env.command.numpy(),reference=env.k['reference'].numpy(),nominal=env.nominal_correction.numpy())
        for w,phase,fallback in selected:
            saved.append(dict(world=w,phase=phase,terminal_fallback=fallback,**{k:v[w].copy() for k,v in arrays.items()}));seen[w].add(phase)
    try:
        env.reset();capture([(w,0,False) for w in range(len(scenes))])
        deadline=int(np.ceil((env.param.numpy()[:,3].max()+2)/.02))+2
        for step in range(deadline):
            env.step_async(np.zeros((len(scenes),3),np.float32))
            state=env.state.numpy();done=env.done.numpy();selected=[]
            for w in range(len(scenes)):
                if completed[w] is not None:continue
                t=state[w,0]*.0005;arrival=state[w,1]
                thresholds={1:1.5,2:2.5}
                if arrival>=0:thresholds.update({3:arrival,4:arrival+.5,5:arrival+1.5})
                for phase in range(1,6):
                    if phase not in seen[w] and ((phase in thresholds and t+1e-9>=thresholds[phase]) or done[w]):
                        selected.append((w,phase,bool(done[w] and (phase not in thresholds or t<thresholds[phase]))))
            if selected:capture(selected)
            _,_,terminated,infos=env.step_wait()
            for w in np.flatnonzero(terminated):
                if completed[w] is None:completed[w]={k:v for k,v in infos[w].items() if k!='terminal_observation'}
            if all(r is not None for r in completed):break
        assert all(r is not None for r in completed) and len(saved)==888
        saved.sort(key=lambda r:(r['world'],r['phase']))
        bank={key:np.stack([s[key] for s in saved]) for key in saved[0]}
        for key in ('ids','heights','gains','feed','angles','yaw'):bank[key]=env.k[key].numpy() if key!='ids' else env.ids.numpy()
        actual=env.model.actuator_gainprm.numpy()[:,:,0]
        bank['actual_gain']=actual[bank['world']]
        bank['fit']=bank['world']<84
        assert np.all(bank['memory'][:,16:19]==0) and np.all(bank['obs'][:,32:]==0)
        np.savez_compressed(out/'state_bank.npz',**bank)
        write(out/'state_collection.json',dict(scenarios=scenes,episodes=completed,snapshots=len(saved),
            terminal_fallback_count=int(bank['terminal_fallback'].sum()),fit_states=int(bank['fit'].sum()),
            validation_states=int((~bank['fit']).sum()),input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},source_sha256=source_hashes(__file__)))
        print('COLLECT',len(saved),'same-state snapshots; failures',sum(not r['success'] for r in completed),flush=True)
        return bank
    finally:env.close()


def arguments(bank,mode,index):
    n=len(index);dim=3 if mode=='diff3' else 6
    get=lambda key,dtype:wp.array(bank[key][index],dtype=dtype)
    memory=wp.array(np.c_[bank['memory'][index,:16],np.zeros((n,dim)),bank['memory'][index,19:]],dtype=D)
    q,v,sensor=[get(k,wp.float32) for k in ('q','v','sensor')]
    ids=wp.array(bank['ids'],dtype=int);heights,gains,feed,angles,yaw=[wp.array(bank[k],dtype=D) for k in ('heights','gains','feed','angles','yaw')]
    active=wp.ones(n,dtype=int);targets=wp.zeros((n,dim));nom=get('nominal',D)
    ctrl=wp.zeros((n,6));diag=wp.zeros((n,38),dtype=D)
    reference,effort,ends,metric,thresholds=load()
    shared=[q,v,sensor,ids,heights,angles,memory,16+dim+4,get('task',D),get('param',D),active,
        wp.array(reference,dtype=D),wp.array(effort,dtype=D),wp.array(ends,dtype=int),wp.array(metric,dtype=D),wp.array(thresholds,dtype=D),nom]
    args=[q,v,sensor,targets,get('command',D),active,memory,ids,heights,gains,feed,angles,get('reference',D),yaw,ctrl,diag,0,0,
        wp.array(np.tile(UPPER,(n,1)),dtype=D),nom]
    return args,shared,targets,ctrl,diag


def nominal(bank):
    args,shared,_,_,diag=arguments(bank,'diff3',np.arange(len(bank['q'])))
    base=[];flags=[]
    for t in range(40):
        if t%10==0:wp.launch(update,len(bank['q']),shared)
        wp.launch(control_physical_nominal,len(bank['q']),args,block_dim=32)
        d=diag.numpy();assert np.all(d[:,6:12]==0) and np.all(d[:,14]==0)
        base.append(d[:,:6]);flags.append(d[:,13])
    bank['base']=np.stack(base,axis=1);bank['nominal_flags']=np.stack(flags,axis=1)
    n=len(bank['q']);ids=bank['ids'];bounds=np.empty((n,6));maps={m:np.zeros((n,6,3 if m=='diff3' else 6)) for m in MODES}
    for w in range(n):
        for j in range(6):bounds[w,j]=sim.hw.torque_limit(float('inf'),float(bank['v'][w,ids[4+j]]),j<4,0.,.0005)[0]/UPPER[j]
        for side in range(2):
            jac=leg_kinematics(bank['q'][w,ids[2*side:2*side+2]],np.zeros(2))[3]
            assert np.linalg.cond(jac)<=1e6
            maps['diff3'][w,2*side:2*side+2,:2]=jac*np.array([.1*7*9.81/2,1])*(1 if side==0 else -1)
            maps['virtual6'][w,2*side:2*side+2,side]=jac[:,0]*(.1*7*9.81/2)
            maps['virtual6'][w,2*side:2*side+2,2+side]=jac[:,1]
        maps['diff3'][w,4:,2]=[.3,-.3];maps['virtual6'][w,4,4]=.3;maps['virtual6'][w,5,5]=.3
        maps['torque6'][w]=np.diag([1.,1.,1.,1.,.3,.3])
    bank['bounds']=bounds
    for m in MODES:bank['map_'+m]=maps[m]


def policy_mean(bank,mode,seed):
    dim=3 if mode=='diff3' else 6
    def factory():
        e=gym.Env();e.observation_space=gym.spaces.Box(-np.inf,np.inf,(38,),dtype=np.float32);e.action_space=gym.spaces.Box(-1.,1.,(dim,),dtype=np.float32);return e
    vec=DummyVecEnv([factory])
    try:
        policy=PPO('MlpPolicy',vec,seed=seed,device='cpu',policy_kwargs=dict(net_arch=[64,64],log_std_init=0.)).policy
        normalized=np.clip(bank['obs']/np.sqrt(1+1e-8),-10,10).astype(np.float32)
        with torch.no_grad():mean=policy.get_distribution(torch.as_tensor(normalized)).distribution.mean.cpu().numpy()
        assert np.isfinite(mean).all() and np.all(policy.log_std.detach().numpy()==0)
        return mean.astype(float)
    finally:vec.close()


def context(bank,mode,mean,index,z):
    # Pair left virtual F/H/wheel noise with M; right noise remains independent.
    noise=z[:,[0,2,4]] if mode=='diff3' else z
    return dict(mean=mean[index],noise=noise,matrix=bank['map_'+mode][index],base=bank['base'][index],
        bounds=bank['bounds'][index],gain=bank['actual_gain'][index],height=bank['param'][index,13])


def packet(c,std,t):
    unbounded=c['mean'][:,None,:]+c['noise'][None,:,:]*std
    actions=np.clip(unbounded,-1,1).astype(np.float32).astype(float);filtered=np.clip(actions,-.01*t,.01*t)
    request=np.einsum('sij,saj->sai',c['matrix'],filtered)
    base=c['base'][:,t-1,None,:];bounds=c['bounds'][:,None,:]
    ratios=np.ones_like(request)
    np.divide(np.where(request>0,bounds-base,-bounds-base),request,out=ratios,where=request!=0)
    lam=np.clip(ratios.min(axis=2),0,1)
    ctrl=(base+lam[:,:,None]*request).astype(np.float32).astype(float)
    residual=(ctrl-base.astype(np.float32).astype(float))*c['gain'][:,None,:]
    return residual,request,ctrl,lam,unbounded


def rms(c,std):return np.sqrt(np.mean((packet(c,std,40)[0]/SCALE)**2,axis=(0,1)))


def statistics(c,std):
    result={}
    for t in (1,10,40):
        residual,request,ctrl,lam,unbounded=packet(c,std,t)
        x=(residual/SCALE).reshape(-1,6);total=(ctrl*c['gain'][:,None,:]/SCALE).reshape(-1,6)
        cov=np.cov(x,rowvar=False,bias=True);eigen=np.linalg.eigvalsh(cov)
        cuts=[.115,.12,.16,.25,.3,.3800001];groups=[]
        for low,high in zip(cuts,cuts[1:]):
            selected=(c['height']>=low)&(c['height']<high)
            if selected.any():groups.append(dict(height_interval_m=[low,high],states=int(selected.sum()),motor_rms_normalized=np.sqrt(np.mean((residual[selected]/SCALE)**2,axis=(0,1))).tolist()))
        result[str(t)]=dict(actual_motor_rms_normalized=np.sqrt(np.mean(x*x,axis=0)).tolist(),
            actual_motor_mean_normalized=x.mean(axis=0).tolist(),covariance_normalized=cov.tolist(),covariance_eigenvalues=eigen.tolist(),
            mapped_request_rms_Nm=np.sqrt(np.mean(request*request,axis=(0,1))).tolist(),
            total_actual_motor_rms_normalized=np.sqrt(np.mean(total*total,axis=0)).tolist(),
            lambda_mean=float(lam.mean()),lambda_zero_fraction=float(np.mean(lam==0)),lambda_below_one_fraction=float(np.mean(lam<1)),
            lambda_histogram=np.histogram(lam,bins=np.linspace(0,1,21))[0].tolist(),
            command_envelope_saturation_fraction=np.mean(abs(ctrl)>=c['bounds'][:,None,:]-1e-7,axis=(0,1)).tolist(),
            policy_coordinate_clipping_fraction=float(np.mean(abs(unbounded)>1)),
            policy_vector_clipping_fraction=float(np.mean(np.any(abs(unbounded)>1,axis=2))),height_groups=groups)
    return result


def native_check(bank,means,stds,z):
    states=np.linspace(0,len(bank['q'])-1,12,dtype=int);index=np.repeat(states,16);maximum=0.;force_error=0.
    import mujoco
    from native.terrain import model
    first=model(HeightTerrainScenario(stand_height_m=.115))
    data=mujoco.MjData(first)
    for mode in MODES:
        args,shared,targets,ctrl,diag=arguments(bank,mode,index)
        c=context(bank,mode,means[mode],states,z[:16]);noise=c['noise']
        a=np.clip(means[mode][states,None,:]+noise[None,:,:]*stds[mode],-1,1).astype(np.float32)
        targets.assign(a.reshape(len(index),-1));kernel=control_physical_nominal_torque6 if mode=='torque6' else control_physical_nominal
        for t in range(1,41):
            if (t-1)%10==0:wp.launch(update,len(index),shared)
            wp.launch(kernel,len(index),args,block_dim=32)
            if t in (1,10,40):
                # Mirror the actual float32 target buffer, not an ideal double target.
                filtered=np.clip(a.astype(float),-.01*t,.01*t);req=np.einsum('sij,saj->sai',c['matrix'],filtered)
                base=c['base'][:,t-1,None,:];ratios=np.ones_like(req)
                np.divide(np.where(req>0,c['bounds'][:,None,:]-base,-c['bounds'][:,None,:]-base),req,out=ratios,where=req!=0)
                lam=np.clip(ratios.min(axis=2),0,1);expected=(base+lam[:,:,None]*req).astype(np.float32).reshape(-1,6)
                error=float(abs(ctrl.numpy()-expected).max());maximum=max(maximum,error);assert error<1e-5
                assert np.all(diag.numpy()[:,14]==0)
        for i in (0,79,191):
            first.actuator_gainprm[:,0]=bank['actual_gain'][index[i]]
            data.qpos[:]=bank['q'][index[i]];data.qvel[:]=bank['v'][index[i]];data.ctrl[:]=ctrl.numpy()[i]
            mujoco.mj_forward(first,data)
            error=float(abs(data.actuator_force-data.ctrl*bank['actual_gain'][index[i]]).max());force_error=max(force_error,error);assert error<1e-12
    return dict(states=12,draws_per_state=16,substeps=40,modes=3,checked_ticks=[1,10,40],
        native_host_command_max_difference_Nm=maximum,cpu_direct_motor_force_max_error_Nm=force_error)


def run(out):
    reg=json.loads((out/'preregistration.json').read_text());assert not (out/'state_bank.npz').exists()
    torch.set_num_threads(1);bank=collect(out,reg);nominal(bank)
    rng=np.random.default_rng(reg['draw_seed']);half=rng.standard_normal((reg['gaussian_draws']//2,6));z=np.r_[half,-half]
    means={m:policy_mean(bank,m,reg['policy_seed']) for m in MODES}
    fit=np.flatnonzero(bank['fit']);validation=np.flatnonzero(~bank['fit']);contexts={m:context(bank,m,means[m],fit,z) for m in MODES}
    pilot={m:rms(contexts[m],np.full(means[m].shape[1],reg['pilot_std'])) for m in MODES}
    target=np.exp(np.mean(np.log(np.stack(list(pilot.values()))),axis=0));assert np.all(target>0)
    stds={};calibration={}
    for mode in MODES:
        dim=means[mode].shape[1];start=np.full(dim,np.log(reg['pilot_std']));calls=0
        def objective(log_std):
            nonlocal calls
            calls+=1
            return np.r_[np.log(rms(contexts[mode],np.exp(log_std))/target),np.sqrt(.001)*(log_std-start)]
        solved=least_squares(objective,start,bounds=np.log(reg['std_bounds']),max_nfev=reg['max_nfev'],ftol=reg['ftol'],xtol=reg['xtol'],gtol=reg['gtol'])
        stds[mode]=np.exp(solved.x);after=rms(contexts[mode],stds[mode])
        calibration[mode]=dict(std=stds[mode].tolist(),log_std=solved.x.tolist(),nfev=int(solved.nfev),all_objective_calls=calls,
            optimizer_success=bool(solved.success),optimizer_message=solved.message,pilot_rms=pilot[mode].tolist(),
            calibrated_rms=after.tolist(),relative_error_to_target=(after/target-1).tolist())
        print('CALIBRATE',mode,calibration[mode],flush=True)
    native=native_check(bank,means,stds,z)
    reports={}
    for seed in reg['verification_policy_seeds']:
        by_mode={}
        for mode in MODES:
            mean=means[mode] if seed==reg['policy_seed'] else policy_mean(bank,mode,seed)
            by_mode[mode]={split:{name:statistics(context(bank,mode,mean,idx,z),sigma) for name,sigma in
                [('default',np.ones(mean.shape[1])),('pilot',np.full(mean.shape[1],reg['pilot_std'])),('calibrated',stds[mode])]} for split,idx in [('fit',fit),('validation',validation)]}
        reports[str(seed)]=by_mode
    np.savez_compressed(out/'state_bank.npz',**bank,gaussian_z=z,**{'mean_'+m:means[m] for m in MODES})
    config=dict(baseline=reg['baseline'],methods={m:dict(action_dim=len(stds[m]),initial_log_std=np.log(stds[m]).tolist(),residual_scale=1.) for m in MODES},
        net_arch=reg['net_arch'],state_bank_sha256=digest(out/'state_bank.npz'),preregistration_sha256=digest(out/'preregistration.json'),
        target_rms_normalized=target.tolist(),source_sha256=source_hashes(__file__),
        weights_promoted=False,learning=False,authority_limits_unchanged=True,full_distribution_equivalence_claimed=False)
    write(out/'initial_action_config.json',config)
    write(out/'verification.json',dict(passed=True,kind=reg['role'],calibration=calibration,target_rms_normalized=target.tolist(),native_check=native,
        statistics=reports,source_sha256=source_hashes(__file__),config_sha256=digest(out/'initial_action_config.json'),
        final_holdout_opened=False,formal_training_admitted=False,learning=False,full_distribution_equivalence_claimed=False))
    print('PASS CONDITIONAL INITIAL ACTION AUDIT',native,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
