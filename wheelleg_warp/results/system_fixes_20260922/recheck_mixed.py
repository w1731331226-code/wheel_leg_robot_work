"""Re-evaluate only already-public affected mixed cases; not a fresh holdout or policy gain."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import torch
from stable_baselines3 import PPO
from native.terrain import TerrainScenario
from terrain_eval import evaluate_terrain,summarize_terrain
from dashboard.live_env import atomic_json as write

if (HERE/'mixed_recheck.json').exists():raise FileExistsError('Refusing to overwrite evidence')
old=json.loads((ROOT/'wheelleg_warp/results/terrain_v4_diagnostics_20260922/terrain_evidence_audit.json').read_text())
cases=[TerrainScenario(**r['scenario']) for r in old['runs'] if r['scenario']['terrain']=='mixed']
checkpoint=json.loads((ROOT/'wheelleg_warp/results/failure_feedback_20260922/control/protocol.json').read_text())['checkpoint']
torch.set_num_threads(1);model=PPO.load(checkpoint+'.zip',device='cpu')
rows=evaluate_terrain(model,checkpoint+'.pkl',cases)
result=dict(task_contract_version=2,checkpoint=checkpoint,checkpoint_sha256=hashlib.sha256(Path(checkpoint+'.zip').read_bytes()).hexdigest(),
    normalization_sha256=hashlib.sha256(Path(checkpoint+'.pkl').read_bytes()).hexdigest(),
    summary=summarize_terrain(rows,[s.terrain_seed for s in cases]),complete_count=sum(r['reason']=='completed' for r in rows),
    contact_count=sum((r['touched_terrain_contact_mask']&r['required_terrain_contact_mask'])==r['required_terrain_contact_mask'] for r in rows),
    exit_count=sum(r['terrain_exit_passed'] for r in rows),evidence_count=sum(r['terrain_evidence_passed'] for r in rows),
    new_holdout=False,trained=False,policy_gain_claimed=False,runs=rows)
write(HERE/'mixed_recheck.json',result);print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False),flush=True)
