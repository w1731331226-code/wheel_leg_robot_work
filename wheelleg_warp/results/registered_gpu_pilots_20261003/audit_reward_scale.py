"""Read-only reward-scale counterfactual using already evaluated classic cases."""
from pathlib import Path
import json,sys,math
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from training_contract import digest
HERE=Path(__file__).resolve().parent;P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'
classic=json.loads((P/'classical_selection.json').read_text())['selected'];rows=classic['runs'];mean=classic['summary']['mean_yaw_score_deg'];factor=(mean-.05)/mean
assert 0<factor<1 and len(rows)==32
costs=[];deltas=[];references=[];success_term=[]
for r in rows:
    assert r['reason']=='completed'
    # Native reward uses accumulated absolute-world yaw^2; terrain yaw is identical.
    yaw_rad=math.radians(r['rms_deg'][2]);cost=yaw_rad*yaw_rad*r['duration_s']/(.08726646**2)
    costs.append(cost);deltas.append(cost*(1-factor*factor));references.append(3.5+1.5*r['task_goal_progress_m']/abs(r['scenario']['speed']))
    success_term.append(10 if r['success'] else -10)
assert abs(mean*factor-(mean-.05))<1e-12
result=dict(passed=True,scope='Reward-unit algebra, constant duration/roll/pitch/speed/control/terminal outcomes; not a policy experiment or causal learning diagnosis',
 classic_mean_Jpsi_deg=mean,counterfactual_mean_Jpsi_deg=mean*factor,uniform_yaw_amplitude_factor=factor,
 mean_episode_yaw_penalty=float(np.mean(costs)),mean_return_improvement_for_absolute_point05deg=float(np.mean(deltas)),
 min_max_return_improvement=[float(min(deltas)),float(max(deltas))],
 success_vs_failure_terminal_return_difference=20,mean_yaw_improvement_fraction_of_terminal_gap=float(np.mean(deltas)/20),
 fixed_nominal_reward_terms='velocity Gaussian width0.25m/s; equal roll/pitch/yaw penalty scaled5deg; physical torque/smoothness penalties; terminal±10',
 risk='Fine yaw improvement is small relative to terminal success difference, so learning may first chase success/contact. This algebra does not establish reward as failure cause.',
 configuration_changed=False,extra_evaluations=0,gate_or_final_simulated=False,
 source_sha256={'wheelleg_warp/native/environment.py':digest(ROOT/'wheelleg_warp/native/environment.py'),'wheelleg_warp/terrain_eval.py':digest(ROOT/'wheelleg_warp/terrain_eval.py')},verifier_sha256=digest(__file__))
(HERE/'reward_scale_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('REWARD SCALE: Jpsi',mean,'->',mean*factor,'mean return delta',result['mean_return_improvement_for_absolute_point05deg'],'vs terminal gap20; no causal assertion/no change')
