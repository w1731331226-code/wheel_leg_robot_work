"""Same physical definitions of original28, including separate solver50 world."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from train_height_comparison import protocol,evaluate,summary,write
from score_yaw_gate import at_least
HERE=Path(__file__).resolve().parent
p=protocol(HERE/'protocol_v3');rows=evaluate(p['regression'],'diff3',candidate=p['b1_candidates'][0])
comparisons=[]
for r,old in zip(rows,p['original_regression']):
    comparisons.append(dict(name=r['seed'],success=r['success'],physical=r['physical_safety_passed'],design=r['design_joint_passed'],
        old_velocity_gate=at_least(old['velocity_rmse']*1.05+.005,r['velocity_rmse']),
        old_roll_pitch_gate=all(at_least(old['peak_deg'][j]+.1,r['peak_deg'][j]) for j in (0,1)),
        solver_iterations=r['scenario']['solver_iterations']))
write(HERE/'legacy_baseline.json',dict(kind='currentB0 versus originalCPU28 engineering check, not learned M3 gate',
    summary=summary(rows),runs=rows,comparisons=comparisons,original_source_unchanged=True,gate_opened=False))
print('LEGACY currentB0',sum(r['success'] for r in rows),'/28; original speed/rollpitch',sum(c['old_velocity_gate'] for c in comparisons),sum(c['old_roll_pitch_gate'] for c in comparisons),flush=True)
