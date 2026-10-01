"""Independent equivalent wheel-phase chart; unchanged full CPU/Warp one-step gates."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from native.models import batch
from native.terrain import model,HeightTerrainScenario
from probe_height_115_cpu_step_pair import forces_cpu
from probe_height_115_action_predict_loow import sha


def run(source,archive,output):
    source=source.resolve();archive=archive.resolve();output=output.resolve();assert not output.exists();output.mkdir(parents=True)
    meta=json.loads((archive/'verification.json').read_text());col=meta['columns'];info=json.loads((source/'verification.json').read_text())
    raw=np.load(source/'native_response.npz',allow_pickle=False);scenes=[HeightTerrainScenario(**s) for s in meta['scenarios']];models=[model(s) for s in scenes]
    starts=raw['start'];q0=starts[:,1:col['qvel_start']].copy();v0=starts[:,col['qvel_start']:col['warmstart_start']];warm=starts[:,col['warmstart_start']:col['ctrl_start']]
    phase_ids=[models[0].joint(n).qposadr[0] for n in ('wheel1','wheel2')]
    q0[:,phase_ids]=(q0[:,phase_ids]+np.pi)%(2*np.pi)-np.pi
    gauge=[]
    for i,sample in enumerate(info['samples']):
        m=models[sample['world']];a=mujoco.MjData(m);b=mujoco.MjData(m)
        a.qpos[:]=starts[i,1:col['qvel_start']];b.qpos[:]=q0[i]
        for d in (a,b):d.qvel[:]=v0[i];d.qacc_warmstart[:]=warm[i];d.ctrl[:]=raw['commands'][i,0];mujoco.mj_forward(m,d)
        error=dict(geom_position=float(abs(a.geom_xpos-b.geom_xpos).max()),geom_rotation=float(abs(a.geom_xmat-b.geom_xmat).max()),
                   mass_matrix=float(abs(a.qM-b.qM).max()),bias_force=float(abs(a.qfrc_bias-b.qfrc_bias).max()),acceleration=float(abs(a.qacc-b.qacc).max()))
        assert max(error['geom_position'],error['geom_rotation'],error['mass_matrix'],error['bias_force'])<1e-10 and error['acceleration']<1e-6
        gauge.append(error)
    n=len(starts);arms=raw['commands'].shape[1];assert n==18 and arms==26
    Q=np.repeat(q0,arms,axis=0).astype(np.float32);V=np.repeat(v0,arms,axis=0).astype(np.float32);W=np.repeat(warm,arms,axis=0).astype(np.float32)
    commands=raw['commands'].reshape(-1,6);cpus=[models[s['world']] for s in info['samples'] for _ in range(arms)]
    wp.init();wp.set_device('cuda:0');_,wm,data,_=batch(cpus,[scenes[s['world']] for s in info['samples'] for _ in range(arms)])
    data.qpos.assign(Q);data.qvel.assign(V);data.qacc_warmstart.assign(W);data.ctrl.assign(commands);mjw.step(wm,data)
    q=data.qpos.numpy().astype(float);v=data.qvel.numpy().astype(float)
    c=data.contact;count=int(data.nacon.numpy()[0]);world=c.worldid.numpy()[:count];geom=c.geom.numpy()[:count]
    native_contacts=[set() for _ in cpus]
    for w,pair in zip(world,geom):
        if w>=0:native_contacts[int(w)].add(tuple(sorted(map(int,pair))))
    cq=[];cv=[];rows=[]
    for index,m in enumerate(cpus):
        d=mujoco.MjData(m);d.qpos[:]=Q[index];d.qvel[:]=V[index];d.qacc_warmstart[:]=W[index];d.ctrl[:]=commands[index];mujoco.mj_step(m,d)
        qe=float(abs(d.qpos-q[index]).max());ve=float(abs(d.qvel-v[index]).max());contacts=set(forces_cpu(m,d))==native_contacts[index]
        cq.append(d.qpos.copy());cv.append(d.qvel.copy());rows.append(dict(sample=index//arms,arm=index%arms,qpos_error=qe,qvel_error=ve,contacts_match=contacts,
                                                                          passed=qe<=2e-6 and ve<=1e-3 and contacts))
    np.savez_compressed(output/'pairing.npz',initial_qpos=Q,initial_qvel=V,commands=commands,cpu_qpos=cq,cpu_qvel=cv,warp_qpos=q,warp_qvel=v)
    result=dict(role='equivalent_periodic_wheel_chart_one_step_pair',full_pair_pass_count=sum(r['passed'] for r in rows),total_pairs=len(rows),all_pairs_pass=all(r['passed'] for r in rows),
        original_gate=dict(qpos=2e-6,qvel=1e-3,contacts_equal=True),gauge_checks=gauge,rows=rows,
        phase_interval_rad=[-np.pi,np.pi],phase_ids=list(map(int,phase_ids)),
        limitations='Only an initial-state equivalent periodic chart in an independent468-arm test. Full coordinate and velocity gates remain unchanged. No production phase wrapping, online predictor, sustained-step equivalence or full task claim. Actual float32 chart starts shared by CPU and Warp; exact gauge invariance checked separately in CPU double precision.',
        input_sha256={str((source/f).relative_to(ROOT)):sha(source/f) for f in ('native_response.npz','verification.json')},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/models.py',ROOT/'wheelleg_warp/native/terrain.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED full pairs',result['full_pair_pass_count'],'/',len(rows),'q',max(r['qpos_error'] for r in rows),'v',max(r['qvel_error'] for r in rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.archive,args.output)
