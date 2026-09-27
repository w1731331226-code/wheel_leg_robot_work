"""只读记录多高度单侧障碍的50 Hz可见信号与轮指令余量。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'wheelleg_ppo/tools')]

import numpy as np
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario,bank_height_v3


def run(control=False,feedback=False):
    heights=(.16,.20,.25,.30,.35,.38)
    if control:
        scenarios=[HeightTerrainScenario(speed=.5,stand_height_m=h) for h in heights]
        scenarios += [HeightTerrainScenario(speed=.5,terrain='step',step_height_m=.02,relative_attitude=True,stand_height_m=h) for h in heights]
    else:scenarios=[HeightTerrainScenario(speed=v,height_l=.027,stand_height_m=h) for v in (.5,-.5) for h in heights]
    env=NativeEnv(n=len(scenarios),scenario=scenarios,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=1 if feedback else 0)
    env.reset();rows=[dict(scenario=asdict(s),events={},wheel_command_samples=[]) for s in scenarios]
    actions=np.zeros((len(scenarios),3),dtype=np.float32)
    for _ in range(700):
        env.step_async(actions)
        state=env.state.numpy();obs=env.obs.numpy();qpos=env.data.qpos.numpy()
        ctrl=env.data.ctrl.numpy();diag=env.diag.numpy();filtered=env.k['state'].numpy();done=env.done.numpy()
        for i,row in enumerate(rows):
            if 'result' in row:continue
            t=round(float(state[i,0]*.0005),3)
            mask=12 if scenarios[i].terrain=='step' else 1 if scenarios[i].height_l else 0
            signals={'contact':bool(mask and int(state[i,14])&mask),
                     'gyro_z_0.05rad_s':abs(obs[i,5])>=.05,
                     'wheel_speed_diff_1rad_s':abs(obs[i,20]-obs[i,21])>=1.,
                     'yaw_1deg':abs(obs[i,2])>=np.deg2rad(1),
                     'yaw_5deg':state[i,10]>=np.deg2rad(5)}
            for name,seen in signals.items():
                if seen and name not in row['events']:
                    row['events'][name]=dict(time_s=t,x_m=round(float(qpos[i,0]),3),yaw_deg=round(float(np.rad2deg(obs[i,2])),3))
            if actions[i,2] and 'first_feedback_action' not in row['events']:
                row['events']['first_feedback_action']=dict(time_s=round(t-.02,3),requested=float(actions[i,2]))
            contact=row['events'].get('contact')
            if contact and t<=contact['time_s']+.100001:
                row['wheel_command_samples'].append(dict(time_s=t,command_over_nominal_4_5Nm=np.round(abs(ctrl[i,4:6])/4.5,3).tolist(),
                                                       wheel_base_clipped=(abs(diag[i,19:21])>abs(diag[i,4:6])+1e-6).tolist(),
                                                       wheel_base_before_nm=np.round(diag[i,19:21],3).tolist(),
                                                       wheel_base_after_nm=np.round(diag[i,4:6],3).tolist(),
                                                       executed_wheel_residual_nm=np.round(diag[i,10:12],3).tolist(),
                                                       residual_lambda=round(float(diag[i,12]),3),
                                                       filtered_action3=round(float(filtered[i,18]),3),
                                                       cumulative_residual_limited_steps=int(state[i,16]),
                                                       cumulative_base_infeasible_steps=int(state[i,18])))
            if done[i]:
                row['result']=dict(reason=int(done[i]),success=bool(state[i,19]),
                                   peak_deg=np.round(np.rad2deg(state[i,8:11]),3).tolist(),
                                   height_rmse_m=round(float(np.sqrt(state[i,28]/state[i,29])),6) if state[i,29] else None)
        next_obs,_,_,_=env.step_wait()
        if feedback:actions[:,2]=np.where(abs(next_obs[:,5])>=.05,-np.sign(next_obs[:,5]),0)
        if all('result' in row for row in rows):break
    assert all('result' in row and (not (s.height_l or s.terrain=='step') or 'contact' in row['events']) for row,s in zip(rows,scenarios))
    if feedback and not control:assert all(any(abs(sample['filtered_action3'])>.01 for sample in row['wheel_command_samples']) for row in rows)
    source=Path(__file__).resolve().parent
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
            ('probe_height_contact_timing.py','native/environment.py','native/controller.py','native/terrain.py')}
    return dict(description='single-batch 50 Hz end-of-step engineering diagnosis',control_cases=control,gyro_feedback=feedback,
                sample_period_s=.02,source_sha256=hashes,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--control',action='store_true');parser.add_argument('--feedback',action='store_true');args=parser.parse_args()
    if args.output.exists():parser.error('不覆盖已有证据')
    result=run(args.control,args.feedback);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PASS',len(result['rows']),args.output)
