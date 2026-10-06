"""Final paired78 policy evaluation on unchanged recorded native tasks."""
import json
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
import phase_support_probe as phase
import equal_exposure_probe as force
import run_phase_support_qualification as phase_runner
import run_equal_exposure_recovery as force_runner
from nominal_reference_env import RawReferenceCache, NormalizedReferencePair
from route_state import RouteState
from nominal_mean_training import equal, weight_digest
from train_height_comparison import summary
import wheelleg_sim as sim


def make_env(cases, panel, normalization, directory=None):
    raw = force.instrument(cases, 'virtual6', directory) if panel == 'auxiliary' else phase.instrument(cases, 'virtual6', directory=directory)
    cache = RawReferenceCache(RouteState(raw, 'route'))
    norm = VecNormalize.load(normalization, VecCheckNan(cache, raise_exception=True))
    norm.training = False; norm.norm_reward = False
    return raw, cache, norm, NormalizedReferencePair(norm, cache)


def evaluate(cases, panel, model, normalization, directory):
    raw, cache, norm, env = make_env(cases, panel, normalization, directory)
    before = (model.num_timesteps, model._n_updates, weight_digest(model))
    stats = (norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count,
             norm.ret_rms.mean.copy(),norm.ret_rms.var.copy(),norm.ret_rms.count)
    rows = [None]*len(cases)
    try:
        obs = env.reset(); assert obs.shape == (len(cases),78)
        ids = raw.ids.numpy(); deadline = int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
        for _ in range(deadline):
            action = model.predict(obs,deterministic=True)[0]
            obs,_,done,infos = env.step(action); stopped = raw.stopped_q.numpy() if done.any() else None
            for w in np.flatnonzero(done):
                if rows[w] is None:
                    length = float(np.mean([sim.fk_joints(float(stopped[w,ids[2*s]]),float(stopped[w,ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    rows[w] = dict(**cases[w],**{k:v for k,v in infos[w].items() if k!='terminal_observation'},final_mean_fk_leg_m=length)
            if all(r is not None for r in rows): break
        assert all(r is not None for r in rows)
        equal(stats,(norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count,norm.ret_rms.mean,norm.ret_rms.var,norm.ret_rms.count))
        assert before == (model.num_timesteps,model._n_updates,weight_digest(model))
        result = dict(runs=rows,summary=summary(rows),physical=sum(r['physical_safety_passed'] for r in rows),design=sum(r['design_joint_passed'] for r in rows))
        phase.rec.atomic_json(directory/'result.json',json.loads(json.dumps(result,allow_nan=False)))
        checked = phase_runner.prior.prior.check_files(directory,cases,{r['seed']:r for r in rows})
        checked.pop('original_label_differences')
        steps = force_runner.logs(directory,cases) if panel == 'auxiliary' else phase_runner.logs(directory,cases)
        assert steps == checked['physics_steps']
        return result,checked
    except BaseException:
        if panel == 'auxiliary': force_runner.preserve(raw,directory,cases)
        else: phase_runner.preserve(raw,directory,cases)
        raise
    finally: env.close()
