"""Route memory lifecycle and real GPU curriculum/physical identity check."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnv
from route_state import RouteState

ROOT=Path(__file__).resolve().parents[1]


class ScriptedEnv(VecEnv):
    def __init__(self):
        self.packet=np.zeros((2,38),np.float32);self.packet[:,7]=[1,2]
        self.actions=None;self.tick=0
        super().__init__(2,Box(-np.inf,np.inf,(38,),dtype=np.float32),Box(-1,1,(3,),dtype=np.float32))
    def reset(self):self.tick=0;return self.packet.copy()
    def step_async(self,actions):self.actions=actions
    def step_wait(self):
        self.tick+=1;packet=self.packet.copy();infos=[{},{}];done=np.array([self.tick==2,False])
        if done[0]:
            infos[0]=dict(duration_s=.03,terminal_observation=packet[0].copy(),reason='completed')
            packet[0,7]=3 # New episode/curriculum packet, not terminal velocity.
        return packet,np.array([1.,2.]),done,infos
    def close(self):pass
    def get_attr(self,name,indices=None):return [getattr(self,name,None)]*2
    def set_attr(self,name,value,indices=None):setattr(self,name,value)
    def env_method(self,*args,**kwargs):raise NotImplementedError
    def env_is_wrapped(self,*args,**kwargs):return [False,False]


def unit():
    for mode in ['route','zero']:
        base=ScriptedEnv();wrapped=RouteState(base,mode)
        initial=wrapped.reset();np.testing.assert_array_equal(initial[:,:38],base.packet)
        np.testing.assert_array_equal(initial[:,38],0)
        a=np.array([[.2,-.3,.4],[-.2,.3,-.4]],np.float32)
        obs,reward,done,infos=wrapped.step(a);assert base.actions is a
        np.testing.assert_array_equal(reward,[1,2]);assert not done.any()
        np.testing.assert_allclose(obs[:,38],[.02,.04] if mode=='route' else [0,0],atol=2e-9)
        obs,_,done,infos=wrapped.step(a);assert done.tolist()==[True,False]
        terminal=infos[0]['terminal_observation'];assert terminal.shape==(39,)
        np.testing.assert_allclose(terminal[38],.03 if mode=='route' else 0,atol=2e-9)
        assert terminal[7]==1 and obs[0,7]==3 and obs[0,38]==0
        assert wrapped.clock[0]==0 and wrapped.odometry.last_velocity[0]==3
        assert wrapped.clock[1]==.04
        assert wrapped.reset().shape==(2,39) and not wrapped.odometry.y.any()
    print('PASS reset, asynchronous terminal, partial dt, switched packet, zero ablation, unchanged actions/rewards')


def gpu(out):
    from train_height_comparison import protocol,CurriculumEnv,b1_action
    from training_contract import digest
    from dashboard.live_env import atomic_json
    base_path=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
    p=protocol(base_path);prior=json.loads((ROOT/'wheelleg_warp/results/paper_recovery_20261004/yaw_sector_v1/registration.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    sources={**p['source_sha256'],**{f'wheelleg_warp/{n}':digest(ROOT/'wheelleg_warp'/n) for n in ['route_state.py','test_route_state.py','probe_packet_odometry.py']}}
    atomic_json(out/'registration.json',dict(role='GPU route wrapper engineering, not method performance',environments=10,training_steps=0,
        modes=['route','zero'],maximum_actor_steps_per_mode=2500,curriculum_milestones=[20,10000],
        source_sha256=sources,protocol_sha256=digest(base_path/'protocol.json'),old_gate_or_final_used=False))
    reports={}
    for mode in ['route','zero']:
        curriculum=CurriculumEnv(p,'diff3',1609,10,milestones=[20,10000]);raw=curriculum.venv;env=RouteState(curriculum,mode)
        assert raw.param.numpy()[:,3].max()+2<20,'Second milestone must follow every stage1 deadline'
        original=curriculum.step_wait;capture={}
        def observed_step():
            result=original()
            capture['packet']=result[0].copy();capture['reward']=result[1].copy();capture['done']=result[2].copy()
            capture['terminal']={w:info['terminal_observation'].copy() for w,info in enumerate(result[3]) if result[2][w]}
            capture['buffers']=[x.numpy() for x in [raw.data.qpos,raw.data.qvel,raw.data.ctrl,raw.state,raw.k['state'],raw.param]]
            return result
        curriculum.step_wait=observed_step
        ended=np.zeros(10,int);terminal_nonzero=0;max_y=0.;checked=0
        try:
            obs=env.reset();np.testing.assert_array_equal(obs[:,:38],raw.obs.numpy())
            for _ in range(2500):
                action=np.stack([b1_action(o,prior['classical']) for o in obs])
                obs,reward,done,infos=env.step(action);checked+=1
                np.testing.assert_array_equal(obs[:,:38],capture['packet']);np.testing.assert_array_equal(reward,capture['reward']);np.testing.assert_array_equal(done,capture['done'])
                for buffer,before in zip([raw.data.qpos,raw.data.qvel,raw.data.ctrl,raw.state,raw.k['state'],raw.param],capture['buffers']):np.testing.assert_array_equal(buffer.numpy(),before)
                for w in np.flatnonzero(done):
                    np.testing.assert_array_equal(infos[w]['terminal_observation'][:38],capture['terminal'][w]);assert obs[w,38]==0 and env.clock[w]==0
                    terminal_nonzero+=int(abs(infos[w]['terminal_observation'][38])>1e-6)
                ended+=done;max_y=max(max_y,float(abs(obs[:,38]).max()))
                if mode=='zero':assert not obs[:,38].any() and all(info['terminal_observation'][38]==0 for w,info in enumerate(infos) if done[w])
                if np.all(ended>=2) and np.all(curriculum.stages==3):break
            assert np.all(ended>=2) and np.all(curriculum.stages==3) and curriculum.transition_rows==20
            assert {r['stage'] for r in curriculum.transition_log}=={2,3}
            assert max_y>1e-4 and terminal_nonzero>0 if mode=='route' else max_y==0 and terminal_nonzero==0
            assert env.reset().shape==(10,39) and not env.odometry.y.any()
            reports[mode]=dict(actor_steps=checked,completed_episodes=int(ended.sum()),per_world_episodes=ended.tolist(),stage3_worlds=10,
                actual_curriculum_transitions=curriculum.transition_rows,max_route_feature_m=max_y,nonzero_terminal_features=terminal_nonzero,
                physical_control_buffers_unchanged_by_wrapper=True,base_packet_reward_done_and_terminal_exact=True)
            print('PASS GPU',mode,reports[mode],flush=True)
        finally:env.close()
    assert all(digest(ROOT/n)==v for n,v in sources.items())
    atomic_json(out/'verification.json',dict(verified=True,reports=reports,training_steps=0,wrapper_contract='packet_route_y_v1',
        scope='Same-execution output-stage physical/control buffer identity, actual two-episode resets and curriculum transitions. Separate mode runs not claimed bitwise paired trajectories or control improvement.',source_sha256=sources))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();unit()
    if args.output:gpu(args.output)
