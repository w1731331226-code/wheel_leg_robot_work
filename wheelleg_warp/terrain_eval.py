"""Deterministic evaluation for fixed terrain-v1 cases."""
from dataclasses import asdict
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from native.terrain_env import TerrainEnv


def evaluate_terrain(model,normalization,cases,env_kwargs=None):
    raw=TerrainEnv(len(cases),scenario=cases,**(env_kwargs or {}))
    env=VecNormalize.load(str(normalization),raw);env.training=False;env.norm_reward=False
    stats=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count);rows=[]
    try:
        obs=env.reset();pending=set(range(len(cases)));found={}
        while pending:
            obs,_,done,infos=env.step(model.predict(obs,deterministic=True)[0])
            for i in list(pending):
                if done[i]:
                    row={k:v for k,v in infos[i].items() if k!='terminal_observation'}
                    found[i]=dict(seed=cases[i].terrain_seed,scenario=asdict(cases[i]),**row);pending.remove(i)
        rows=[found[i] for i in range(len(cases))]
        np.testing.assert_array_equal(stats[0],env.obs_rms.mean);np.testing.assert_array_equal(stats[1],env.obs_rms.var)
        assert stats[2]==env.obs_rms.count
    finally:env.close()
    return rows
