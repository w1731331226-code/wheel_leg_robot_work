"""Current-source radial fix: original28 definitions, no modified admission gates."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from native.environment import NativeEnv
import train_height_comparison as runner
from train_height_comparison import evaluate,summary,write
from training_contract import digest,source_hashes
from score_yaw_gate import at_least
HERE=Path(__file__).resolve().parent
old=ROOT/'wheelleg_ppo/tools/results/fixes_2026-09-17/final/baseline.json';baseline=json.loads(old.read_text())['runs']
cases=json.loads((HERE.parent/'twentyeighth_round_comparison_entry_20261003/protocol_v3/protocol.json').read_text())['regression']
runner.raw_env=lambda rows,mode:NativeEnv.height115_candidate(n=len(rows),scenario=[runner.HeightTerrainScenario(**r['scenario']) for r in rows],residual_mode=mode,shared_reference=True)
rows=evaluate(cases,'diff3',candidate=dict(kp=0.,kd=0.,roll_gain=0.))
report=dict(role='controlled current-source zeroActor regression check; not admission',summary=summary(rows),runs=rows,
    original_source_sha256=digest(old),source_sha256=source_hashes(__file__),
    old_velocity_passes=sum(at_least(b['velocity_rmse']*1.05+.005,r['velocity_rmse']) for r,b in zip(rows,baseline)),
    old_roll_pitch_passes=sum(all(at_least(b['peak_deg'][j]+.1,r['peak_deg'][j]) for j in (0,1)) for r,b in zip(rows,baseline)))
write(HERE/'fixed_capped_legacy.json',report)
print('FIXED CURRENT',report['summary'],'old speed/attitude',report['old_velocity_passes'],report['old_roll_pitch_passes'],flush=True)
print('FAILED',[(r['seed'],r['peak_deg'],r['max_requested_radial_guard_N']) for r in rows if not r['success']],flush=True)
