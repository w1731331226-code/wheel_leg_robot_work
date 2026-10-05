"""Frozen-policy2x2 request allocation; steering is not an inside-Nom replacement."""
from pathlib import Path
import json,gc
import numpy as np
import warp as wp
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper,VecNormalize,VecCheckNan
from stable_baselines3 import PPO
import route_pilot_env as experiment
from route_state import RouteState
from probe_route_feedback import actions
from native.controller import D,control_physical_nominal
from gpu_reference_budget import apply_budget,WIDTH
from smoke_reward_training import weight_digest
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/fixed_steering_leg_diagnostic_v1'
CONDITIONS={'zero_leg_no_assist':(False,False),'learned_room_leg_no_assist':(True,False),
            'zero_leg_fixed_assist':(False,True),'learned_room_leg_fixed_assist':(True,True)}


class LegSteering(VecEnvWrapper):
    def __init__(self,venv,candidate,leg,assist):
        assert venv.observation_space.shape==(39,) and venv.action_space.shape==(3,)
        super().__init__(venv,action_space=Box(-1,1,(2,),dtype=np.float32));self.candidate=candidate;self.leg=leg;self.assist=assist;self.packet=None
    def reset(self):self.packet=self.venv.reset().copy();return self.packet.copy()
    def step_async(self,a):
        a=np.asarray(a,np.float32)
        if self.packet is None or a.shape!=(self.num_envs,2) or not np.isfinite(a).all() or np.any(abs(a)>1):raise ValueError('Invalid leg request/cache')
        out=np.zeros((self.num_envs,3),np.float32)
        if self.leg:out[:,:2]=a
        if self.assist:out[:,2]=actions(self.packet,self.candidate,1)[0][:,2]
        self.venv.step_async(out)
    def step_wait(self):
        result=self.venv.step_wait();self.packet=result[0].copy();return result


@wp.kernel
def steering_stats(active:wp.array[int],tracking:wp.array[int],targets:wp.array2d[float],memory:wp.array2d[D],diagnostic:wp.array2d[D],ctrl:wp.array2d[float],stats:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0 or tracking[w]==0 or diagnostic[w,14]!=D(0):return
    raw=D(.3)*D(targets[w,2]);filtered=D(.3)*memory[w,18];accepted=diagnostic[w,10]
    left=D(ctrl[w,4])-D(float(diagnostic[w,4]));right=D(ctrl[w,5])-D(float(diagnostic[w,5]))
    stats[w,0]+=D(1);stats[w,1]+=raw*raw;stats[w,2]+=filtered*filtered;stats[w,3]+=accepted*accepted
    stats[w,4]+=left*left;stats[w,5]+=right*right;stats[w,6]+=diagnostic[w,12]
    stats[w,7]=wp.max(stats[w,7],wp.abs(accepted-diagnostic[w,12]*filtered))


def collect(cases,model,norm,condition,owner_only=False):
    leg,assist=CONDITIONS[condition];p=json.loads((OUT/'proposal.json').read_text());factory=experiment.raw_env;launch=wp.launch;n=len(cases)
    initial=np.zeros((n,WIDTH));initial[:,2]=1.;stats=wp.array(initial,dtype=D);wheel=wp.zeros((n,8),dtype=D);tracking=wp.array(np.ones(n,np.int32));captured={};pending=set(range(n))
    def patched(kernel,dim,inputs=None,**kwargs):
        result=launch(kernel,dim,**kwargs) if inputs is None else launch(kernel,dim,inputs,**kwargs)
        if kernel is control_physical_nominal:
            launch(apply_budget,n,[1,inputs[0],inputs[7],inputs[12],inputs[5],inputs[15],inputs[14],tracking,stats])
            launch(steering_stats,n,[inputs[5],tracking,inputs[3],inputs[6],inputs[15],inputs[14],wheel])
        return result
    wp.launch=patched
    try:raw=factory(cases,'diff3')
    finally:wp.launch=launch
    raw._fixed_steering_buffers=(stats,wheel,tracking);gc.collect();assert raw._fixed_steering_buffers[1] is wheel
    wait=raw.step_wait
    def step_wait():
        result=wait();done=[w for w in np.flatnonzero(result[2]) if w in pending]
        if done:
            s=stats.numpy();t=wheel.numpy();mask=tracking.numpy()
            for w in done:
                result[3][w]['leg_room_stats']=s[w].tolist();result[3][w]['steering_stats']=t[w].tolist();mask[w]=0;pending.remove(w)
            tracking.assign(mask)
        return result
    raw.step_wait=step_wait
    env=LegSteering(RouteState(raw,'route'),p['fixed_wheel_controller']['candidate'],leg,assist)
    if owner_only:
        env.reset();assert env.packet.shape==(n,39) and raw._fixed_steering_buffers[0] is stats;env.close();return dict(verified=True,worlds=n,owner_reset_checked=True,physical_steps=0)
    env=VecNormalize.load(str(norm),VecCheckNan(env,raise_exception=True));env.training=False;env.norm_reward=False
    rms=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count);before=(model.num_timesteps,model._n_updates,weight_digest(model));rows=[None]*n
    try:
        obs=env.reset();ids=raw.ids.numpy();deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
        for _ in range(deadline):
            obs,_,done,infos=env.step(model.predict(obs,deterministic=True)[0]);stopped=raw.stopped_q.numpy() if done.any() else None
            for w in np.flatnonzero(done):
                if rows[w] is None:
                    length=float(np.mean([sim.fk_joints(float(stopped[w,ids[2*s]]),float(stopped[w,ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    rows[w]={**cases[w],**{k:v for k,v in infos[w].items() if k!='terminal_observation'},'final_mean_fk_leg_m':length}
            if all(x is not None for x in rows):break
        assert all(x is not None for x in rows) and not pending
        np.testing.assert_array_equal(rms[0],env.obs_rms.mean);np.testing.assert_array_equal(rms[1],env.obs_rms.var);assert rms[2]==env.obs_rms.count and before==(model.num_timesteps,model._n_updates,weight_digest(model))
        return dict(summary=experiment.summary(rows),physical=sum(x['physical_safety_passed'] for x in rows),design=sum(x['design_joint_passed'] for x in rows),runs=rows)
    finally:env.close()


def unit():
    from test_route_state import ScriptedEnv
    p=json.loads((OUT/'proposal.json').read_text());candidate=p['fixed_wheel_controller']['candidate'];checks=[]
    for condition,(leg,assist) in CONDITIONS.items():
        raw=ScriptedEnv();raw.packet[:,2]=[.1,-.1];raw.packet[:,5]=[.03,-.03];raw.packet[:,9]=[.7,-.7]
        wrapped=LegSteering(RouteState(raw,'route'),candidate,leg,assist);obs=wrapped.reset();a=np.array([[.2,-.3],[-.2,.3]],np.float32)
        for tick in range(3):
            expected=np.zeros((2,3),np.float32)
            if leg:expected[:,:2]=a
            if assist:expected[:,2]=actions(obs,candidate,1)[0][:,2]
            obs,_,done,infos=wrapped.step(a);np.testing.assert_array_equal(raw.actions,expected)
            if tick==1:assert done[0] and obs[0,38]==0 and wrapped.packet[0,38]==0 and infos[0]['terminal_observation'].shape==(39,)
        wrapped.reset();assert not wrapped.packet[:,38].any();wrapped.close();checks.append(condition)
    # Normalization changes the policy input, not the cached raw feedback packet.
    raw=ScriptedEnv();raw.packet[:,2]=[.1,-.1];raw.packet[:,5]=[.03,-.03];raw.packet[:,9]=[.7,-.7]
    allocated=LegSteering(RouteState(raw,'route'),candidate,False,True);normal=VecNormalize(allocated,training=False,norm_reward=False);normal.obs_rms.mean=np.full(39,2.);normal.obs_rms.var=np.full(39,4.)
    try:
        obs=normal.reset();assert not np.array_equal(obs,allocated.packet);expected=actions(allocated.packet,candidate,1)[0];normal.step(a);np.testing.assert_array_equal(raw.actions,expected)
    finally:normal.close()
    atomic_json(OUT/'interface_unit.json',dict(verified=True,conditions=checks,raw_packet_feedback_not_normalized=True,zeroleg_B0_B1route_request_equivalence=True,leg_passthrough=True,asynchronous_terminal_reset=True,physical_rollouts=0,training_updates=0))
    print('PASS4condition raw39/zero request/equivalence/normalization/terminal checks',flush=True)


def freeze():
    assert not (OUT/'source_contract.json').exists();unit();p=json.loads((OUT/'proposal.json').read_text())
    cases=p['cases']['regular']+p['cases']['controlled'];owner=collect(cases,None,None,'zero_leg_fixed_assist',True);atomic_json(OUT/'owner_check.json',owner)
    prior=json.loads((ROOT/'wheelleg_warp/results/paper_recovery_20261004/reference_budget_learning_v1/trainer_contract.json').read_text())
    sources={**prior['source_sha256'],'wheelleg_warp/fixed_steering_leg.py':sha(__file__),'wheelleg_warp/test_route_state.py':sha(ROOT/'wheelleg_warp/test_route_state.py')};assert all(sha(ROOT/n)==v for n,v in sources.items())
    atomic_json(OUT/'source_contract.json',dict(proposal_sha256=sha(OUT/'proposal.json'),source_sha256=sources,unit_sha256=sha(OUT/'interface_unit.json'),owner_sha256=sha(OUT/'owner_check.json'),budget_episodes=1632,training_updates=0,
        estimator='FixedPD/path uses raw RouteState before VecNormalize; actualActorinput normalized as savedmodel. Onlyleg output on/off fromepisode start, originalthirdrequest nolearnedwheel.',
        steering_fields=['valid_samples','raw_request_Nm_energy','filtered_request_Nm_energy','accepted_diag_left_Nm_energy','executed_left_increment_energy','executed_right_increment_energy','sum_original_lambda','max_diag_acceptance_identity_error'],
        limitation='Commonrequestedsteering is not commonacceptedsteering dueoriginalglobal lambda. Room onlyleg command. Fixedpolicycontext intervention, no changedNom/source or retrainedcontroller.'))
    print('FROZEN1632 fixedpolicy diagnostic;no rollouts yet',flush=True)


def verify():
    c=json.loads((OUT/'source_contract.json').read_text());assert c['proposal_sha256']==sha(OUT/'proposal.json') and all(sha(ROOT/n)==v for n,v in c['source_sha256'].items());return c


def run():
    verify();p=json.loads((OUT/'proposal.json').read_text());destination=OUT/'runs';destination.mkdir(exist_ok=False);ledger=[];completed=0
    try:
        for m in p['models']:
            prefix=Path(m['prefix']);assert sha(prefix.with_suffix('.zip'))==m['checkpoint']['checkpoint_sha256'] and sha(prefix.with_suffix('.pkl'))==m['checkpoint']['normalization_sha256'];model=PPO.load(str(prefix)+'.zip',device='cuda')
            for condition in CONDITIONS:
                for panel in ['regular','controlled']:
                    label=f'{m["seed"]}_{condition}_{panel}';atomic_json(OUT/'progress.json',dict(status='running',completed_episodes=completed,pending_job=label));result=collect(p['cases'][panel],model,str(prefix)+'.pkl',condition);path=destination/(label+'.json');atomic_json(path,result)
                    ledger.append(dict(seed=m['seed'],condition=condition,panel=panel,path='runs/'+path.name,sha256=sha(path),episodes=len(p['cases'][panel])));completed+=len(p['cases'][panel]);atomic_json(OUT/'completed_jobs.json',dict(completed_episodes=completed,records=ledger));verify();print('COMPLETED',completed,label,result['summary']['success_count'],flush=True)
        assert completed==1632;atomic_json(OUT/'completion.json',dict(completed_episodes=completed,records=ledger,training_updates=0,source_contract_sha256=sha(OUT/'source_contract.json')));atomic_json(OUT/'progress.json',dict(status='complete',completed_episodes=1632))
    except BaseException as e:
        atomic_json(OUT/'interruption.json',dict(completed_episodes=completed,error=str(e),pending_job_consumption_unknown=True,silently_resumable=False));raise


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['unit','freeze','run']);globals()[p.parse_args().command]()
