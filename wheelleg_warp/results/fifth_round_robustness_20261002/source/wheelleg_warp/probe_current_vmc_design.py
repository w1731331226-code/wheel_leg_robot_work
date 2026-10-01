"""Fixed current-J nominal LQR design trial; original costs, limits and default tables retained."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.design import current_vmc_table
from native.environment import NativeEnv
from probe_height_115_margin import cases
from probe_height_115_action_predict_loow import sha


def run(output,robustness_panel=False):
    assert not output.exists();output.mkdir(parents=True)
    scenes=cases()[:6]
    labels=['nominal']*6
    if robustness_panel:
        variations=[('nominal',{}),('training_corner',dict(mass=7.5,mu_l=.6,mu_r=1.,drive_difference=.05,delay_ms=10.)),
            ('pressure_corner',dict(mass=8.,mu_l=.4,mu_r=1.2,drive_difference=-.05,delay_ms=20.))]
        base=scenes;scenes=[replace(scene,**change) for _,change in variations for scene in base];labels=[name for name,_ in variations for _ in base]
        output.joinpath('registered_cases.json').write_text(json.dumps({'selection':'three fixed parameter settings crossed with existing six normal cases; public development, no final holdout','labels':labels,'scenarios':[asdict(s) for s in scenes]},indent=2)+'\n')
    n=len(scenes);env=NativeEnv.height115_candidate(n=n,scenario=scenes,residual_scale=0)
    table=[(k,f,a) for k,f,a in zip(env.k['gains'].numpy(),env.k['feed'].numpy(),env.k['angles'].numpy())];reports=env.design_reports
    try:
        env.reset();rows=[None]*n
        for _ in range(700):
            _,_,done,infos=env.step(np.zeros((n,3),np.float32))
            for w in np.flatnonzero(done):
                if rows[w] is None:rows[w]={k:v for k,v in infos[w].items() if k!='terminal_observation'}
            if all(row is not None for row in rows):break
        assert all(row is not None for row in rows)
        assert all(row['physical_steps']==row['physical_evidence_steps'] for row in rows)
    finally:env.close()
    np.savez_compressed(output/'design.npz',gains=np.stack([t[0] for t in table]),feed=np.stack([t[1] for t in table]),angles=[t[2] for t in table])
    result=dict(role='paired_public_robustness_development_check' if robustness_panel else 'current_J_nominal_LQR_independent_trial',scenarios=[asdict(s) for s in scenes],candidate=rows,parameter_groups=labels,
        physical_pass_count=sum(row['physical_safety_passed'] for row in rows),task_pass_count=sum(row['success'] for row in rows),
        full_normal_gate_pass=all(row['success'] for row in rows),linear_design_reports=reports,
        default_table_unchanged=True,original_LQR_costs_and_limits_unchanged=True,
        limitations='Corrected equilibrium feed VMC derivative; still first-order, static reference projection and experimental radial guard. No arbitrary gain search, default promotion, online safety or all-height claim.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/design.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_ppo/tools/model_lqr.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED physical',result['physical_pass_count'],'task',result['task_pass_count'],flush=True)
    print([(r['stop_distance_m'],r['tail_speed_m_s'],r['min_actual_A_leg_m']) for r in rows],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--robustness-panel',action='store_true');args=parser.parse_args()
    run(args.output,args.robustness_panel)
