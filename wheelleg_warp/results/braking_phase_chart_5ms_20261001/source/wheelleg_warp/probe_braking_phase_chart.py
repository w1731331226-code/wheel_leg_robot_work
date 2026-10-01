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
import wheelleg_sim as sim
from native.models import batch
from native.terrain import model,HeightTerrainScenario,HEIGHT_115_GEOMETRIC_MIN as LIMIT
from probe_braking_motor_response import features
from probe_height_115_contact_action_pair import margins
from probe_height_115_cpu_step_pair import forces_cpu
from probe_height_115_action_predict_loow import sha


def run(source,archive,output,steps=1):
    assert steps in (1,10)
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
        ma=np.empty((m.nv,m.nv));mb=np.empty_like(ma);mujoco.mj_fullM(m,a,ma);mujoco.mj_fullM(m,b,mb)
        error=dict(geom_position=float(abs(a.geom_xpos-b.geom_xpos).max()),geom_rotation=float(abs(a.geom_xmat-b.geom_xmat).max()),
                   mass_matrix=float(abs(ma-mb).max()),bias_force=float(abs(a.qfrc_bias-b.qfrc_bias).max()),acceleration=float(abs(a.qacc-b.qacc).max()))
        assert max(error['geom_position'],error['geom_rotation'],error['mass_matrix'],error['bias_force'])<1e-10 and error['acceleration']<1e-6
        gauge.append(error)
    n=len(starts);arms=raw['commands'].shape[1];assert n==18 and arms==26
    Q=np.repeat(q0,arms,axis=0).astype(np.float32);V=np.repeat(v0,arms,axis=0).astype(np.float32);W=np.repeat(warm,arms,axis=0).astype(np.float32)
    commands=raw['commands'].reshape(-1,6);cpus=[models[s['world']] for s in info['samples'] for _ in range(arms)]
    wp.init();wp.set_device('cuda:0');_,wm,data,_=batch(cpus,[scenes[s['world']] for s in info['samples'] for _ in range(arms)])
    data.qpos.assign(Q);data.qvel.assign(V);data.qacc_warmstart.assign(W);data.ctrl.assign(commands)
    cpu_data=[]
    for index,m in enumerate(cpus):
        d=mujoco.MjData(m);d.qpos[:]=Q[index];d.qvel[:]=V[index];d.qacc_warmstart[:]=W[index];d.ctrl[:]=commands[index];cpu_data.append(d)
    cq=[];cv=[];gq=[];gv=[];rows=[];physical=[];cpu_contacts=[];warp_contacts=[];before_v=V
    for step in range(steps):
        mjw.step(wm,data);q=data.qpos.numpy().astype(float);v=data.qvel.numpy().astype(float);force=data.actuator_force.numpy().astype(float)
        c=data.contact;count=int(data.nacon.numpy()[0]);world=c.worldid.numpy()[:count];geom=c.geom.numpy()[:count]
        native_contacts=[set() for _ in cpus]
        for w,pair in zip(world,geom):
            if w>=0:native_contacts[int(w)].add(tuple(sorted(map(int,pair))))
        now_q=[];now_v=[]
        for index,(m,d) in enumerate(zip(cpus,cpu_data)):
            mujoco.mj_step(m,d);qe=float(abs(d.qpos-q[index]).max());ve=float(abs(d.qvel-v[index]).max())
            contacts=set(forces_cpu(m,d));equal=contacts==native_contacts[index]
            cpu_contacts.extend([[step,index,*pair] for pair in sorted(contacts)]);warp_contacts.extend([[step,index,*pair] for pair in sorted(native_contacts[index])])
            now_q.append(d.qpos.copy());now_v.append(d.qvel.copy());rows.append(dict(step=step+1,sample=index//arms,arm=index%arms,qpos_error=qe,qvel_error=ve,contacts_match=equal,
                                                                               passed=qe<=2e-6 and ve<=1e-3 and equal))
        for i,sample in enumerate(info['samples']):
            m=models[sample['world']];sl=slice(i*arms,(i+1)*arms);f=features(m,q[sl],v[sl]);joint=margins(m,q[sl]);bound=np.zeros((arms,6))
            for j,name in enumerate(('alphaL','betaL','alphaR','betaR','wheel1','wheel2')):
                for arm in range(arms):bound[arm,j]=sim.hw.torque_limit(1e6,before_v[i*arms+arm,m.joint(name).dofadr[0]],j<4,0.,.0005)[0]
            quat=q[sl,3:7];pitch=np.arcsin(np.clip(2*(quat[:,0]*quat[:,2]-quat[:,3]*quat[:,1]),-1,1))
            roll=np.arctan2(2*(quat[:,0]*quat[:,1]+quat[:,2]*quat[:,3]),1-2*(quat[:,1]**2+quat[:,2]**2))
            yaw=np.arctan2(2*(quat[:,0]*quat[:,3]+quat[:,1]*quat[:,2]),1-2*(quat[:,2]**2+quat[:,3]**2))
            allowed=bound/np.array([1.,1.,1.,1.,1.05,1.05])
            safe=(f[:,:4].min(axis=1)>=LIMIT)&(joint.min(axis=1)>=0)&(np.max(abs(np.c_[roll,pitch,yaw]),axis=1)<=np.deg2rad(5))
            safe &= (np.max(abs(force[sl])-bound,axis=1)<=1e-6)&(np.max(abs(commands[sl])-allowed,axis=1)<=1e-6)&np.isfinite(np.c_[q[sl],v[sl]]).all(axis=1)
            physical.append(dict(step=step+1,sample=i,safe_arms=int(safe.sum()),min_A_B_m=float(f[:,:4].min()),min_joint_margin_rad=float(joint.min())))
        cq.append(now_q);cv.append(now_v);gq.append(q);gv.append(v);before_v=v
    np.savez_compressed(output/'pairing.npz',initial_qpos=Q,initial_qvel=V,commands=commands,cpu_qpos=cq[0] if steps==1 else cq,cpu_qvel=cv[0] if steps==1 else cv,
                        warp_qpos=gq[0] if steps==1 else gq,warp_qvel=gv[0] if steps==1 else gv,cpu_contacts=cpu_contacts,warp_contacts=warp_contacts)
    result=dict(role='equivalent_periodic_wheel_chart_one_step_pair',full_pair_pass_count=sum(r['passed'] for r in rows),total_pairs=len(rows),all_pairs_pass=all(r['passed'] for r in rows),
        original_gate=dict(qpos=2e-6,qvel=1e-3,contacts_equal=True),gauge_checks=gauge,rows=rows,
        propagation_steps=steps,horizon_ms=steps*.5,physical=physical,all_physical_steps_pass=all(r['safe_arms']==arms for r in physical),
        phase_interval_rad=[-np.pi,np.pi],phase_ids=list(map(int,phase_ids)),
        limitations='Initial-state equivalent periodic chart, both backends share float32 starts. Full q/v/contact gates unchanged at every step. Known issued total commands frozen throughout horizon, not future controller commands. No production phase wrapping, sensor-based predictor, recursive safety or full task claim. Exact initial gauge equivalence checked separately in CPU double precision.',
        input_sha256={str((source/f).relative_to(ROOT)):sha(source/f) for f in ('native_response.npz','verification.json')},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/models.py',ROOT/'wheelleg_warp/native/terrain.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED full pairs',result['full_pair_pass_count'],'/',len(rows),'q',max(r['qpos_error'] for r in rows),'v',max(r['qvel_error'] for r in rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,choices=(1,10),default=1);args=parser.parse_args();run(args.source,args.archive,args.output,args.steps)
