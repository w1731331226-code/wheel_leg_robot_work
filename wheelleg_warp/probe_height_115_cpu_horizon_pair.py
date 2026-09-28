"""平地首越线前20 ms：归档控制序列下CPU/Warp连续40物理步配对。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
from native.terrain import HeightTerrainScenario,model
from probe_height_115_cpu_step_pair import forces_cpu,forces_warp


def run(output):
    assert not output.exists()
    folder=ROOT/'wheelleg_warp/results/height_115_local_states_20260928'
    source=folder/'verification.json';record=json.loads(source.read_text())
    arrays=np.load(folder/'windows.npz');cols=record['columns'];rows=[];models={}
    for event_id,event in enumerate(record['events']):
        w=event['world']
        if w not in (0,4,5):continue
        if w not in models:models[w]=model(HeightTerrainScenario(**event['scenario']))
        m=models[w];a=arrays[f'event_{event_id}_pre'];b=arrays[f'event_{event_id}_post']
        contact=arrays[f'event_{event_id}_contacts']
        start=int(np.argmin(abs(a[:,0]-(event['reference_s']-.02))))
        assert start+40<=len(a)
        d=mujoco.MjData(m);d.time=float(a[start,0]);d.qpos[:]=a[start,cols['qpos_start']:cols['qpos_start']+m.nq]
        d.qvel[:]=a[start,cols['qvel_start']:cols['qvel_start']+m.nv]
        d.qacc_warmstart[:]=a[start,cols['warmstart_start']:cols['warmstart_start']+m.nv]
        qerr=[];verr=[];pairs=[];force_err=[]
        for step in range(40):
            k=start+step
            d.ctrl[:]=a[k,cols['ctrl_start']:cols['ctrl_start']+m.nu]
            mujoco.mj_step(m,d)
            qerr.append(float(np.max(abs(d.qpos-b[k,cols['qpos_start']:cols['qpos_start']+m.nq]))))
            verr.append(float(np.max(abs(d.qvel-b[k,cols['qvel_start']:cols['qvel_start']+m.nv]))))
            cpu=forces_cpu(m,d);warp=forces_warp(contact[k],w)
            pairs.append(set(cpu)==set(warp))
            force_err.append(max((abs(cpu[key]-warp[key]) for key in set(cpu)&set(warp)),default=0.))
        rows.append(dict(world=w,event=event['kind'],reference_s=event['reference_s'],
                         start_s=float(a[start,0]),end_s=float(b[start+39,0]),
                         max_qpos_abs_diff=float(max(qerr)),end_qpos_abs_diff=qerr[-1],
                         max_qvel_abs_diff=float(max(verr)),end_qvel_abs_diff=verr[-1],
                         contact_pair_steps_equal=int(sum(pairs)),max_shared_contact_normal_abs_diff_N=float(max(force_err))))
    assert len(rows)==6
    summary=dict(states=len(rows),contact_pair_steps_equal=sum(r['contact_pair_steps_equal'] for r in rows),
                 total_physical_steps=40*len(rows),
                 max_qpos_abs_diff=max(r['max_qpos_abs_diff'] for r in rows),
                 max_qvel_abs_diff=max(r['max_qvel_abs_diff'] for r in rows),
                 max_end_qpos_abs_diff=max(r['end_qpos_abs_diff'] for r in rows),
                 max_end_qvel_abs_diff=max(r['end_qvel_abs_diff'] for r in rows))
    summary['horizon_pass']=bool(summary['contact_pair_steps_equal']==summary['total_physical_steps'] and
                                 summary['max_qpos_abs_diff']<=5e-5 and summary['max_qvel_abs_diff']<=.02)
    sources=('wheelleg_warp/probe_height_115_cpu_horizon_pair.py','wheelleg_warp/probe_height_115_cpu_step_pair.py',
             'wheelleg_warp/probe_height_115_local_states.py','wheelleg_warp/native/terrain.py',
             'wheelleg_warp/native/models.py','wheelleg_ppo/xml/wheelleg.xml',
             'wheelleg_ppo/tools/hardware_profile.py')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(dict(role='public_flat_cpu_warp_20ms_open_loop_pair',training=False,final_holdout_opened=False,
            source_windows_sha256=record['windows_sha256'],summary=summary,rows=rows,
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}),
            stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(summary)
    assert summary['horizon_pass'],'CPU/Warp连续20 ms配对未过，停止CPU局部线性可达性'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
