"""Two CUDA PPO updates per route/zero arm; engineering weights never promoted."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize,VecCheckNan
from route_state import RouteState
from train_height_comparison import protocol,CurriculumEnv,new_agent,equal
from smoke_reward_training import weight_digest,check_agent
from training_contract import digest,checkpoint_hashes
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'


def run(out):
    torch.set_num_threads(1);p=protocol(BASE)
    interface=ROOT/'wheelleg_warp/results/paper_recovery_20261004/route_interface_v1/verification.json'
    checked_interface=json.loads(interface.read_text());assert checked_interface['verified']
    assert all(digest(ROOT/n)==v for n,v in checked_interface['source_sha256'].items())
    out.mkdir(parents=True,exist_ok=False)
    sources={**p['source_sha256'],**{f'wheelleg_warp/{n}':digest(ROOT/'wheelleg_warp'/n) for n in ['route_state.py','probe_packet_odometry.py','smoke_route_training.py','smoke_reward_training.py']}}
    atomic_json(out/'registration.json',dict(role='CUDA39D engineering only, no method-performance test',modes=['zero','route'],
        environments=10,seed=1609,policy_steps_per_arm=1000,ppo=p['ppo'],normalization=p['normalization'],
        updates_per_arm=2,initialization='Same seed,39-input architecture and original M3 calibrated log_std',
        interface_verification_sha256=digest(interface),source_sha256=sources,old_gate_or_final_used=False,
        restore_scope='Weights, Adam, normalization and consumed counters; no physical or odometry trajectory resume'))
    reports={};initial=None;completed=0
    try:
        for mode in ['zero','route']:
            atomic_json(out/'progress.json',dict(status='running',completed_training_steps=completed,pending_arm=mode))
            raw=CurriculumEnv(p,'diff3',1609,10);route=RouteState(raw,mode);checked=VecCheckNan(route,raise_exception=True)
            env=VecNormalize(checked,**p['normalization']);agent=new_agent(p,'M3',1609,env)
            initial_digest=weight_digest(agent)
            extra_before=[network[0].weight[:,38].detach().clone() for network in
                [agent.policy.mlp_extractor.policy_net,agent.policy.mlp_extractor.value_net]]
            if initial is None:initial=initial_digest
            else:assert initial_digest==initial
            try:
                agent.learn(total_timesteps=1000)
                assert agent.num_timesteps==1000 and agent._n_updates==20 and weight_digest(agent)!=initial_digest
                adam=check_agent(agent,40)
                assert env.obs_rms.mean.shape==(39,) and np.isfinite(env.obs_rms.var).all()
                assert abs(env.obs_rms.mean[38])>1e-9 if mode=='route' else env.obs_rms.mean[38]==0
                extra_delta=[float((network[0].weight[:,38]-before_column).detach().abs().max().item()) for network,before_column in
                    zip([agent.policy.mlp_extractor.policy_net,agent.policy.mlp_extractor.value_net],extra_before)]
                assert all(v>0 for v in extra_delta) if mode=='route' else extra_delta==[0.,0.]
                prefix=out/mode;before=weight_digest(agent)
                agent.save(prefix);env.save(str(prefix)+'.pkl')
                restored=PPO.load(str(prefix)+'.zip',device='cuda')
                assert restored.observation_space.shape==(39,) and restored.num_timesteps==1000 and restored._n_updates==20
                assert before==weight_digest(agent)==weight_digest(restored)
                equal(agent.policy.optimizer.state_dict(),restored.policy.optimizer.state_dict())
                check_agent(restored,40)
                loaded=VecNormalize.load(str(prefix)+'.pkl',checked)
                equal(env.obs_rms.__dict__,loaded.obs_rms.__dict__);equal(env.ret_rms.__dict__,loaded.ret_rms.__dict__)
                losses={k:float(v) for k,v in agent.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}
                assert losses and all(np.isfinite(v) for v in losses.values())
                reports[mode]=dict(policy_steps=1000,ppo_updates=2,ppo_epochs=20,adam_updates=40,adam=adam,
                    initial_weight_sha256=initial_digest,final_weight_sha256=before,checkpoint=checkpoint_hashes(prefix),
                    route_mean=float(env.obs_rms.mean[38]),route_variance=float(env.obs_rms.var[38]),
                    route_input_weight_max_deltas=extra_delta,
                    weights_adam_normalization_and_counters_restored=True,losses=losses,checkpoint_promoted=False)
                completed+=1000;atomic_json(out/(mode+'.json'),reports[mode]);print('PASS CUDA',mode,reports[mode],flush=True)
            finally:env.close()
        assert completed==2000 and all(digest(ROOT/n)==v for n,v in sources.items())
        atomic_json(out/'verification.json',dict(verified=True,total_training_steps=completed,reports=reports,
            source_sha256=sources,scope='Real two CUDA updates per arm and checkpoint/RMS lifecycle. No performance, convergence, independent test or trajectory-resume conclusion.'))
        atomic_json(out/'progress.json',dict(status='complete',completed_training_steps=completed))
    except BaseException as error:
        atomic_json(out/'interruption.json',dict(completed_arms_steps=completed,error=str(error),pending_arm_consumption_unknown=True,silently_resumable=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
