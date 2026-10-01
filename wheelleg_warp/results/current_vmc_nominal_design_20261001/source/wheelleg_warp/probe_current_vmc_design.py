"""Fixed current-J nominal LQR design trial; original costs, limits and default tables retained."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import wheelleg_sim as sim
import model_lqr as ml
from audit_radial_braking_response import nominal_input_jacobian
from native.environment import NativeEnv
from native.terrain import bank_height_115
from probe_height_115_margin import cases
from probe_height_115_action_predict_loow import sha


def run(output):
    assert not output.exists();output.mkdir(parents=True)
    model,_=sim.load_model(ml.XML,True);table=[];reports=[]
    for height in (.115,sim.L_SQUAT_MIN,sim.L_PREP,sim.L_STAND,sim.L_MAX):
        ref,a,b,_=ml.design(model,height,min_height=.115);c,g,_=ml.vmc_coordinates(ref)
        outer=np.zeros((3,15));outer[2]=sim.hw.DESIGN_MASS/sim.hw.BASELINE_MASS*(sim.KP_LEG*c[6]+sim.KD_LEG*c[7])
        actual_input=nominal_input_jacobian(model,ref)
        assert np.allclose(actual_input,nominal_input_jacobian(model,ref,5e-7),rtol=.01,atol=1e-4)
        # Existing reducer subtracts the radial PD. Add geometric feed derivative
        # first, so its full closed-loop check represents the actual current-J map.
        effective_a=a+b@(actual_input+g@outer)
        gain,report=ml.reduced_design(model,ref,effective_a,b,6)
        assert report['linear_pass'];reports.append(dict(height_m=height,**report))
        k=np.linalg.solve(g,gain)[:2]@np.linalg.pinv(c[:6]);mapping=ml.sagittal_basis(model)[1]
        feed=np.linalg.solve(g,np.linalg.pinv(mapping)@ref.ctrl)
        angle=sim.fk_joints(ref.qpos[7],ref.qpos[10])['phi5']+np.pi/2
        table.append((k,feed,angle))
    scenes=cases()[:6]
    env=NativeEnv(n=6,scenario=scenes,bank_factory=bank_height_115,height_conditioned=True,height_design='range115',
                  residual_scale=0,feasible_reference=True,coordinated_reference=True,radial_guard=True)
    try:
        # Arrays captured by the existing graph retain their addresses.
        env.k['gains'].assign(np.stack([t[0] for t in table]));env.k['feed'].assign(np.stack([t[1] for t in table]))
        env.k['angles'].assign(np.array([t[2] for t in table]));env.reset();rows=[None]*6
        for _ in range(700):
            _,_,done,infos=env.step(np.zeros((6,3),np.float32))
            for w in np.flatnonzero(done):
                if rows[w] is None:rows[w]={k:v for k,v in infos[w].items() if k!='terminal_observation'}
            if all(row is not None for row in rows):break
        assert all(row is not None for row in rows)
        assert all(row['physical_steps']==row['physical_evidence_steps'] for row in rows)
    finally:env.close()
    np.savez_compressed(output/'design.npz',gains=np.stack([t[0] for t in table]),feed=np.stack([t[1] for t in table]),angles=[t[2] for t in table])
    result=dict(role='current_J_nominal_LQR_independent_trial',scenarios=[asdict(s) for s in scenes],candidate=rows,
        physical_pass_count=sum(row['physical_safety_passed'] for row in rows),task_pass_count=sum(row['success'] for row in rows),
        full_normal_gate_pass=all(row['success'] for row in rows),linear_design_reports=reports,
        default_table_unchanged=True,original_LQR_costs_and_limits_unchanged=True,
        limitations='Corrected equilibrium feed VMC derivative; still first-order, static reference projection and experimental radial guard. No arbitrary gain search, default promotion, online safety or all-height claim.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/audit_radial_braking_response.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_ppo/tools/model_lqr.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED physical',result['physical_pass_count'],'task',result['task_pass_count'],flush=True)
    print([(r['stop_distance_m'],r['tail_speed_m_s'],r['min_actual_A_leg_m']) for r in rows],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
