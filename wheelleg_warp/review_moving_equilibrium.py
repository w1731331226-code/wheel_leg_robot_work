"""Independent fresh/cold nominal force/geometry review;no solves or steps."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import mujoco
from review_yaw_sector import ROOT,sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import model_lqr as ml
import wheelleg_sim as sim
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/moving_equilibrium_contract_v1'


def run():
    assert not (OUT/'review.json').exists()
    p=json.loads((OUT/'proposal.json').read_text());c=json.loads((OUT/'completion.json').read_text())
    a=json.loads((OUT/'source_admission.json').read_text())
    assert c['verified'] and len(c['records'])==10 and c['residual_calls']<=12000
    assert c['admission_sha256']==sha(OUT/'source_admission.json') and a['proposal_sha256']==sha(OUT/'proposal.json')
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items())
    m,_=sim.load_model(ml.XML,True);assert (m.nq,m.nv,m.nu)==(17,16,6)
    common=np.zeros((16,8));common[[0,2,4],[0,1,2]]=1
    pairs=[('alphaL','alphaR'),('passA_L','passA_R'),('betaL','betaR'),('passC_L','passC_R'),('wheel1','wheel2')]
    for j,(left,right) in enumerate(pairs):common[[m.jnt_dofadr[m.joint(left).id],m.jnt_dofadr[m.joint(right).id]],j+3]=1
    hips=[m.joint(n).id for n in ('alphaL','passA_L','betaL','passC_L')]
    rights=[m.joint(n).id for n in ('alphaR','passA_R','betaR','passC_R')]
    wheel=np.array([m.jnt_dofadr[m.joint(n).id] for n in ('wheel1','wheel2')])
    amap=np.zeros((6,3))
    for j,names in enumerate([('motor_wheelL','motor_wheelR'),('motor_alphaL','motor_alphaR'),('motor_betaL','motor_betaR')]):
        for n in names:amap[m.actuator(n).id,j]=1
    joints=[m.joint(n).id for n in ('alphaL','betaL','alphaR','betaR','passA_L','passC_L','passA_R','passC_R')]
    qa=m.jnt_qposadr[joints];va=[m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
    records=[]
    for i,r in enumerate(c['records']):
        assert (r['height'],r['speed'])==(p['heights'][i//2],p['speeds_m_s'][i%2])
        path=OUT/r['path'];assert sha(path)==r['sha256']
        with np.load(path,allow_pickle=False) as z:
            x=z['solution'].copy();assert np.isfinite(x).all() and np.all((x>=z['lower'])&(x<=z['upper']))
            d=mujoco.MjData(m);d.qpos[2]=x[0];d.qpos[3:7]=[math.cos(x[1]/2),0,math.sin(x[1]/2),0]
            d.qpos[m.jnt_qposadr[hips]]=d.qpos[m.jnt_qposadr[rights]]=x[2:6]
            d.qvel[0]=r['speed'];d.qvel[wheel]=x[9];d.ctrl[:]=amap@x[6:9]
            assert not np.any(d.qacc_warmstart) and not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
            np.testing.assert_array_equal(d.qpos,z['q']);np.testing.assert_array_equal(d.qvel,z['v']);np.testing.assert_array_equal(d.ctrl,z['ctrl'])
            mujoco.mj_forward(m,d);coldacc=d.qacc.copy()
            limits=np.array([sim.hw.torque_limit(float('inf'),float(d.qvel[n]),k<4,0.,.0005)[0] for k,n in enumerate(va)])
            actual_excess=float(np.maximum(abs(d.actuator_force)-limits,0).max());command_excess=float(np.maximum(abs(d.ctrl)-limits,0).max())
            active_margin=float((1.4-abs(d.qpos[qa[:4]])).min())
            joint_margin=float(np.minimum(d.qpos[qa]-m.jnt_range[joints,0],m.jnt_range[joints,1]-d.qpos[qa]).min())
            lengths=[];loops=[]
            for side in ('L','R'):
                hip=(d.xanchor[m.joint('alpha'+side).id]+d.xanchor[m.joint('beta'+side).id])/2
                center=d.xpos[m.body('wheel'+side).id];end=d.site_xpos[m.site('couplerB_'+side+'_end').id]
                lengths.extend([float(np.linalg.norm(center-hip)),float(np.linalg.norm(end-hip))]);loops.append(float(np.linalg.norm(center-end)))
            floor=m.geom('floor').id;wheels={m.geom('wheel_collide_L').id,m.geom('wheel_collide_R').id}
            supported=d.ncon==2 and all(floor in set(contact.geom) and (set(contact.geom)-{floor}).issubset(wheels) for contact in d.contact)
            d.qacc[:]=0;mujoco.mj_inverse(m,d)
            residual=np.r_[common.T@(d.qfrc_inverse-d.qfrc_actuator),100*(sim.fk_joints(x[2],x[4])['leg_len']-r['height']),x[1]]
            cold_pass=bool(abs(coldacc).max()<=1e-4 and abs(residual).max()<=1e-6 and active_margin>=0 and joint_margin>=0
                           and min(lengths)>=.1147044660616607 and actual_excess<=1e-6 and command_excess<=1e-6 and supported)
            records.append(dict(height=r['height'],speed=r['speed'],path=r['path'],sha256=r['sha256'],solver_reported_pass=r['passed'],
                cold_forward_qacc_max=float(abs(coldacc).max()),cold_force_residual_max=float(abs(residual).max()),
                cold_minus_saved_qacc_max=float(abs(coldacc-z['qacc']).max()),cold_minus_saved_residual_max=float(abs(residual-z['residual']).max()),
                actual_leg_min=min(lengths),loop_error_max=max(loops),active_margin=active_margin,eight_margin=joint_margin,
                command_excess=command_excess,actual_excess=actual_excess,bilateral_support=bool(supported),cold_check_passed=cold_pass))
    atomic_json(OUT/'review.json',dict(verified=True,records=records,passed=all(r['cold_check_passed'] for r in records),
        completion_sha256=sha(OUT/'completion.json'),reviewer_sha256=sha(__file__),fresh_data_checks=10,
        additional_optimization_calls=0,integration_steps=0,transitionFD_calls=0,new_training_samples=0,controller_admitted=False,
        limits='Freshcold nominalforward/inverse andgeometry only;solverdeclaredzeroacc need notmatch bitwise. Existingthresholds unchanged. Not nonlinearflow/actualNom/GPUcontact/controller stability;264 finite admission decision.'))
    print('PASS263 independentcoldchecks',sum(r['cold_check_passed'] for r in records),'/10;no solves/steps/FD/PPO',flush=True)


if __name__=='__main__':run()
