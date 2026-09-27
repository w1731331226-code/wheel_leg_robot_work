"""复用既有2 kHz接触力遥测，核查多高度单侧轮失接触与滑移。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]

import numpy as np
from native.terrain import HeightTerrainScenario
from trace_failure_chain import COL,RecordedEnv


def run(output,timing):
    heights=(.16,.20,.25,.30,.35,.38)
    cases=[HeightTerrainScenario(speed=v,height_l=.027,stand_height_m=h) for v in (.5,-.5) for h in heights]
    reference=json.loads(timing.read_text())['rows']
    assert [row['scenario'] for row in reference]==[asdict(case) for case in cases]
    env=RecordedEnv(len(cases),scenario=cases,height_conditioned=True,residual_scale=0)
    env.reset();chunks=[];zero=np.zeros((len(cases),3),np.float32)
    for step in range(285):
        env.step_async(zero)
        if 4.8<=(step+1)*.02<=5.7:chunks.append(env.trace.numpy().copy())
        env.step_wait()
    trace=np.concatenate(chunks,axis=0);col={name:COL.index(name) for name in COL};rows=[]
    for w,case in enumerate(cases):
        a=trace[:,w];t=a[:,col['end_s']];start=reference[w]['contact_s'];end=reference[w]['yaw5_s']
        window=(t>=start)&(t<=end);b=a[window];assert len(b)>100
        load=b[:,col['left_normal_N']];weight=b[:,col['left_load_weight']]
        slip=np.divide(b[:,col['left_slip_weighted']],weight,out=np.zeros(len(b)),where=weight>1)
        no_left=b[:,col['left_contacts']]<.5
        actual=b[:,[col['actual_torque_4'],col['actual_torque_5']]]
        commanded=b[:,[col['base_4'],col['base_5']]]
        torque_error=float(np.max(abs(actual-commanded)))
        assert torque_error<1e-5,('仿真电机力矩与限幅指令不一致',case,torque_error)
        rows.append(dict(scenario=asdict(case),contact_s=start,yaw5_s=end,
            first_no_left_contact_s=round(float(b[np.flatnonzero(no_left)[0],col['end_s']]),4),
            left_no_contact_fraction=round(float(no_left.mean()),4),
            left_peak_normal_N=round(float(load.max()),2),
            right_min_normal_N=round(float(b[:,col['right_normal_N']].min()),2),
            left_peak_force_weighted_slip_m_s=round(float(slip.max()),3),
            left_mean_force_weighted_slip_m_s=round(float(b[:,col['left_slip_weighted']].sum()/max(weight.sum(),1e-9)),3),
            left_peak_rpm=round(float(np.max(abs(b[:,col['motor_speed_4']]))*60/(2*np.pi)),2),
            left_min_nominal_bound_Nm=round(float(b[:,col['bound_4']].min()),3),
            actual_torque_vs_command_max_abs_Nm=round(torque_error,8)))
    source=Path(__file__).resolve().parent
    names=('probe_height_contact_load.py','trace_failure_chain.py','native/environment.py','native/controller.py','native/terrain.py','native/models.py')
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'trace.npz',trace=trace)
    (output/'summary.json').write_text(json.dumps(dict(source_sha256=hashes,timing_reference_sha256=hashlib.sha256(timing.read_bytes()).hexdigest(),
        sample_period_s=.0005,columns=COL,rows=rows),ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(rows),output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--timing',type=Path,required=True);args=parser.parse_args()
    run(args.output,args.timing)
