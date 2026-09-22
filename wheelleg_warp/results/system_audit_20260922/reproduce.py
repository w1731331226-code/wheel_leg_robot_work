"""CPU-only reproductions of audit findings. No rollout, training or source repair."""
from pathlib import Path
import ast,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
import warp as wp
from native.terrain import TerrainScenario,model as terrain_model
from native.models import model,batch
from ppo_env import Scenario
from dashboard.live_env import atomic_json as write

wp.init();wp.set_device('cpu')
environment=ROOT/'wheelleg_warp/native/environment.py';tree=ast.parse(environment.read_text())
declared=next(ast.literal_eval(n) for n in ast.walk(tree) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='mixed' for k in n.keys))
goal_expr=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='goal' for k in n.targets))
extent=[]
for direction in (1,-1):
    s=TerrainScenario(speed=direction*.7,terrain='mixed',roughness_m=.006,terrain_seed=250005)
    m=terrain_model(s);end=-np.inf
    for i in range(m.ngeom):
        if not m.geom(i).name.startswith('terrain_') or m.geom_pos[i,2]<-1:continue
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,m.geom_quat[i])
        end=max(end,direction*m.geom_pos[i,0]-s.center+np.dot(abs(rotation.reshape(3,3)[0]),m.geom_size[i]))
    goal=eval(compile(ast.Expression(goal_expr),str(environment),'eval'),{'s':s,'abs':abs})-s.center
    extent.append(dict(direction=direction,actual_end=float(end),declared_end=declared['mixed'],goal=goal,
        exit_covers_geometry=bool(declared['mixed']>=end-1e-8),goal_beyond_geometry=bool(goal>end)))
old=json.loads((ROOT/'wheelleg_warp/results/terrain_v4_diagnostics_20260922/terrain_evidence_audit.json').read_text())
mixed=[r for r in old['runs'] if r['scenario']['terrain']=='mixed'];actual_end=extent[0]['actual_end']
exited=lambda r:min(r['wheel_progress_m'])>=actual_end-1e-8
mixed_result=dict(total=len(mixed),reported_evidence=sum(r['terrain_evidence_passed'] for r in mixed),
    reported_success=sum(r['success'] for r in mixed),all64_reported_success=sum(r['success'] for r in old['runs']),
    actual_exit=sum(exited(r) for r in mixed),success_and_actual_exit=sum(r['success'] and exited(r) for r in mixed),
    affected_seeds=[r['seed'] for r in mixed if r['terrain_evidence_passed'] and not exited(r)],
    all64_reported_evidence=sum(r['terrain_evidence_passed'] for r in old['runs']),
    all64_corrected_evidence=sum(r['terrain_evidence_passed'] and (r['scenario']['terrain']!='mixed' or exited(r)) for r in old['runs']),
    all64_corrected_success=sum(r['success'] and r['terrain_evidence_passed'] and (r['scenario']['terrain']!='mixed' or exited(r)) for r in old['runs']),
    frozen_final_result_rewritten=False)
delays=[]
for delay in (10.,15.,20.):
    history=np.zeros(21,dtype=int);steps=round(delay*2);first=None
    for step in range(1,141):
        history[step%21]=step;index=int(wp.mod(step-steps+21,21));observed=int(history[index])
        if step==40:first=dict(expected=max(0,step-steps),observed=observed)
    delays.append(dict(requested_ms=delay,implemented_ms=(step-observed)*.5,at_20ms=first,negative_indices_wrap_safely=True))
scenarios=[Scenario(mass=7.,solver_iterations=50),Scenario(mass=8.,solver_iterations=100)]
cpu=[model(s) for s in scenarios];_,batched,_,_=batch(cpu,scenarios)
batching=dict(expected_meaninertia=[float(m.stat.meaninertia) for m in cpu],actual_meaninertia=batched.stat.meaninertia.numpy().tolist(),
    expected_iterations=[int(m.opt.iterations) for m in cpu],actual_shared_iterations=int(batched.opt.iterations))
motor_model=model(Scenario(drive_difference=.03));motor_data=mujoco.MjData(motor_model)
mujoco.mj_resetDataKeyframe(motor_model,motor_data,motor_model.keyframe('stand').id)
wheel=motor_model.actuator('motor_wheelL').id;motor_data.ctrl[wheel]=4.5;mujoco.mj_forward(motor_model,motor_data)
torque=dict(command=float(motor_data.ctrl[wheel]),actuator_force=float(motor_data.actuator_force[wheel]),
    gain=float(motor_model.actuator_gainprm[wheel,0]),gear=float(motor_model.actuator_gear[wheel,0]),force_limited=bool(motor_model.actuator_forcelimited[wheel]))
scores={}
for filename in ('train_terrain.py','train_terrain_v3.py'):
    path=ROOT/'wheelleg_warp'/filename;module=ast.parse(path.read_text());fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='score')
    scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),scope);score=scope['score']
    terrain=dict(total=48,success_count=40,mean_yaw_score_deg=1.);legacy=dict(total=16,success_count=16,mean_yaw_score_deg=1.)
    call=lambda t,l:score(dict(terrain=t,legacy=l,terrain_runs=[])) if 'v3' in filename else score(t,l)
    exception=None
    try:call({**terrain,'mean_yaw_score_deg':None},legacy)
    except TypeError as exc:exception=str(exc)
    scores[filename]=dict(incomplete_candidate_exception=exception,
        legacy_regression_would_rank_better=call({**terrain,'success_count':42},{**legacy,'success_count':15})<call(terrain,legacy))
paths=[environment,ROOT/'wheelleg_warp/native/models.py',ROOT/'wheelleg_warp/native/terrain.py',ROOT/'wheelleg_warp/train_terrain.py',ROOT/'wheelleg_warp/train_terrain_v3.py']
result=dict(device='cpu',new_physics_rollouts=False,trained=False,production_code_changed=False,
    mixed_geometry=extent,mixed_existing_audit=mixed_result,delay_aliasing=delays,batch_parameters=batching,formal_selection=scores,nominal_command_vs_actual_torque=torque,
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
write(HERE/'reproduction.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
