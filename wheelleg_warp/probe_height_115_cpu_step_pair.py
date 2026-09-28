"""已归档pre-step状态的CPU MuJoCo／Warp一步物理配对。"""
from collections import defaultdict
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


def forces_cpu(m,d):
    groups=defaultdict(float);force=np.zeros(6)
    for i in range(d.ncon):
        c=d.contact[i];mujoco.mj_contactForce(m,d,i,force)
        groups[tuple(sorted((int(c.geom1),int(c.geom2))))]+=max(0.,float(force[0]))
    return dict(groups)


def forces_warp(rows,world):
    groups=defaultdict(float)
    for row in rows:
        if int(row[0])==world:
            groups[tuple(sorted((int(row[1]),int(row[2]))))]+=max(0.,float(row[16]))
    return dict(groups)


def run(output):
    assert not output.exists()
    folder=ROOT/'wheelleg_warp/results/height_115_local_states_20260928'
    source=folder/'verification.json';record=json.loads(source.read_text())
    arrays=np.load(folder/'windows.npz');columns=record['columns']
    worlds=(0,4,5)  # 平地+0.5、+1、−1 m/s；分别含腿长/关节首越界事件。
    models={};rows=[]
    for event_id,event in enumerate(record['events']):
        w=event['world']
        if w not in worlds:continue
        if w not in models:models[w]=model(HeightTerrainScenario(**event['scenario']))
        m=models[w]
        pre=arrays[f'event_{event_id}_pre'];post=arrays[f'event_{event_id}_post']
        contacts=arrays[f'event_{event_id}_contacts']
        for lead in (.05,.02,.01):
            target=event['reference_s']-lead
            sample=int(np.argmin(np.abs(pre[:,0]-target)))
            before=pre[sample];after=post[sample]
            d=mujoco.MjData(m);d.time=float(before[0])
            d.qpos[:]=before[columns['qpos_start']:columns['qpos_start']+m.nq]
            d.qvel[:]=before[columns['qvel_start']:columns['qvel_start']+m.nv]
            d.qacc_warmstart[:]=before[columns['warmstart_start']:columns['warmstart_start']+m.nv]
            d.ctrl[:]=before[columns['ctrl_start']:columns['ctrl_start']+m.nu]
            mujoco.mj_step(m,d)  # 不能预先调用mj_forward改变求解历史。
            cpu=forces_cpu(m,d);warp=forces_warp(contacts[sample],w)
            shared=set(cpu)&set(warp)
            rows.append(dict(world=w,event=event['kind'],reference_s=event['reference_s'],lead_ms=int(round(lead*1000)),
                pre_time_s=float(before[0]),qpos_max_abs_diff=float(np.max(np.abs(d.qpos-after[columns['qpos_start']:columns['qpos_start']+m.nq]))),
                qvel_max_abs_diff=float(np.max(np.abs(d.qvel-after[columns['qvel_start']:columns['qvel_start']+m.nv]))),
                warmstart_max_abs_diff=float(np.max(np.abs(d.qacc_warmstart-after[columns['warmstart_start']:columns['warmstart_start']+m.nv]))),
                contact_pairs_equal=set(cpu)==set(warp),cpu_contact_pairs=len(cpu),warp_contact_pairs=len(warp),
                contact_normal_max_abs_diff_N=max((abs(cpu[key]-warp[key]) for key in shared),default=None)))
    assert len(rows)==18
    summary=dict(states=len(rows),contact_pairs_equal=sum(r['contact_pairs_equal'] for r in rows),
        max_qpos_abs_diff=max(r['qpos_max_abs_diff'] for r in rows),
        max_qvel_abs_diff=max(r['qvel_max_abs_diff'] for r in rows),
        max_warmstart_abs_diff=max(r['warmstart_max_abs_diff'] for r in rows),
        max_shared_contact_normal_abs_diff_N=max((r['contact_normal_max_abs_diff_N'] for r in rows
                                                  if r['contact_normal_max_abs_diff_N'] is not None),default=None))
    summary['one_step_pass']=bool(summary['contact_pairs_equal']==len(rows) and
                                  summary['max_qpos_abs_diff']<=2.e-6 and
                                  summary['max_qvel_abs_diff']<=1.e-3)
    sources=('wheelleg_warp/probe_height_115_cpu_step_pair.py','wheelleg_warp/probe_height_115_local_states.py',
             'wheelleg_warp/native/terrain.py','wheelleg_warp/native/models.py',
             'wheelleg_ppo/tools/wheelleg_sim.py','wheelleg_ppo/tools/hardware_profile.py',
             'wheelleg_ppo/xml/wheelleg.xml')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(dict(role='public_flat_selected_cpu_warp_single_step_pair',training=False,final_holdout_opened=False,
            source_windows_sha256=record['windows_sha256'],summary=summary,rows=rows,
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}),
            stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(summary)
    assert summary['one_step_pass'],'CPU/Warp原扭矩一步配对未过，停止CPU局部QP'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
