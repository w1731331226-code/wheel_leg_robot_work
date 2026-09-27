"""Check a serialized public Warp prefix in a fresh Python process."""
from pathlib import Path
import argparse, hashlib, json, sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from native.terrain import TerrainScenario
from native.terrain_env import TerrainEnv
from probe_replay_prefix_v2 import PANEL,SEED,model_hashes,restore,rollout
from dashboard.live_env import atomic_json as write


def check(directory):
    manifest=json.loads((directory/'prefix_manifest.json').read_text())
    assert hashlib.sha256((directory/'prefix_state.npz').read_bytes()).hexdigest()==manifest['snapshot_sha256']
    frozen=json.loads(PANEL.read_text());cases=[TerrainScenario(**row['scenario']) for row in frozen['panels']]
    index=next(i for i,case in enumerate(cases) if case.terrain_seed==SEED)
    checkpoint=frozen['checkpoints']['terrain_v3']['path']
    env=TerrainEnv(len(cases),scenario=cases)
    norm=VecNormalize.load(checkpoint+'.pkl',env);norm.training=False;norm.norm_reward=False
    torch.set_num_threads(1);model=PPO.load(checkpoint+'.zip',device='cpu')
    try:
        assert model_hashes(env)==manifest['static_model_array_sha256']
        assert [env.data.nworld,env.data.naconmax,env.data.njmax]==manifest['data_capacity']
        with np.load(directory/'prefix_state.npz',allow_pickle=False) as saved:restore(env,saved)
        result=rollout(env,norm,model,index,'baseline')
        result.pop('records')
        write(directory/'cross_process_baseline.json',dict(snapshot_restored_bitwise=True,
            static_model_matched=True,baseline=result,
            verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            note='Fresh-process replay may differ in trajectory because GPU stepping is not bitwise deterministic.'))
        print('cross-process restore passed',result,flush=True)
    finally:norm.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    if (a.input/'cross_process_baseline.json').exists():p.error('Refusing to overwrite check')
    check(a.input)
