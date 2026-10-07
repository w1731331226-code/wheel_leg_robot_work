"""Frozen481D final-model evaluation with existing dense physical evidence."""
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
import execution_input_collector as collector
import parking_withdrawal_probe as parking
import run_phase_support_qualification as recorder
from execution_history_env import ExecutionHistory
from route_state import RouteState
from smoke_reward_training import weight_digest
from train_height_comparison import summary, equal
from probe_route_feedback import actions
import wheelleg_sim as sim
from review_yaw_sector import sha


def make_env(cases, arm, normalization, directory):
    # The old strong classicalPD keeps its original parking requests.
    condition = 'original' if arm == 'B1-route' else 'parking_request_withdrawal'
    raw = collector.instrument(lambda rows,mode: parking.instrument(rows,mode,condition,directory),
                               cases,'virtual6',expected_captures=2)
    route = RouteState(raw)
    history = ExecutionHistory(route,raw,arm if arm in ('H0','H1') else 'H1')
    if normalization is None: return raw,route,history,None,history
    norm = VecNormalize.load(str(normalization),VecCheckNan(history,raise_exception=True))
    norm.training=False; norm.norm_reward=False
    return raw,route,history,norm,norm


def evaluate(cases, arm, model, normalization, directory, classical=None):
    raw,route,history,norm,env = make_env(cases,arm,normalization,directory)
    model_before = (model.num_timesteps,model._n_updates,weight_digest(model)) if model else None
    stats = (norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count,
             norm.ret_rms.mean.copy(),norm.ret_rms.var.copy(),norm.ret_rms.count) if norm else None
    rows = [None]*len(cases)
    actor = [[] for _ in cases]
    try:
        obs=env.reset(); assert obs.shape==(len(cases),481)
        ids=raw.ids.numpy(); deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
        for _ in range(deadline):
            raw_input = history.encode()
            if model: action=model.predict(obs,deterministic=True)[0]
            elif arm=='B0': action=np.zeros((len(cases),6),np.float32)
            else:
                old,_=actions(history.frames[:,-1],classical['candidate'],classical['arm'])
                assert not old[:,:2].any()
                action=np.zeros((len(cases),6),np.float32);action[:,4]=old[:,2];action[:,5]=-old[:,2]
            for w in range(len(cases)):
                if rows[w] is None: actor[w].append(np.r_[raw_input[w],obs[w],action[w]].astype(np.float32))
            obs,_,done,infos=env.step(action)
            stopped=raw.stopped_q.numpy() if done.any() else None
            for w in np.flatnonzero(done):
                if rows[w] is None:
                    length=float(np.mean([sim.fk_joints(float(stopped[w,ids[2*s]]),float(stopped[w,ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    rows[w]=dict(**cases[w],**{k:v for k,v in infos[w].items() if k!='terminal_observation'},final_mean_fk_leg_m=length)
            if all(r is not None for r in rows):break
        assert all(r is not None for r in rows)
        for w,row in enumerate(rows):
            data=np.array(actor[w]); assert len(data)==int(np.ceil(row['physical_steps']/40)) and data.shape[1]==968
            assert np.isfinite(data).all()
            path=directory/f'actor_world_{w}.npz';np.savez_compressed(path,trace=data,
                layout=np.array(['raw481','normalized481_or_rawclassical','submitted6']))
            row['actor_trace']=dict(path=path.name,sha256=sha(path),rows=len(data))
        if norm:
            equal(stats,(norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count,norm.ret_rms.mean,norm.ret_rms.var,norm.ret_rms.count))
            assert model_before==(model.num_timesteps,model._n_updates,weight_digest(model))
        result=dict(runs=rows,summary=summary(rows),physical=sum(r['physical_safety_passed'] for r in rows),
                    design=sum(r['design_joint_passed'] for r in rows))
        recorder.write(directory/'result.json',result)
        checked=recorder.prior.prior.check_files(directory,cases,{r['seed']:r for r in rows})
        checked.pop('original_label_differences')
        assert recorder.logs(directory,cases)==checked['physics_steps']
        collector.preserve(raw,directory/'last_command_buffers.npz')
        np.savez_compressed(directory/'last_history.npz',frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)
        return result,checked
    except BaseException:
        recorder.preserve(raw,directory,cases);parking.preserve(raw,directory);collector.preserve(raw,directory/'interrupted_command.npz')
        np.savez_compressed(directory/'interrupted_history.npz',frames=history.frames,inputs=history.inputs,elapsed=history.elapsed,valid=history.valid)
        for w,part in enumerate(actor):
            if part:np.savez_compressed(directory/f'interrupted_actor_{w}.npz',trace=np.array(part))
        raise
    finally:env.close()
