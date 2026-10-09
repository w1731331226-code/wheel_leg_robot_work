"""Existing20-stream handover/pose decomposition; no new control or physics."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import wheelleg_sim as sim
from review_yaw_sector import sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/design_crossing_diagnostic_v2'


def alpha(radius,angle):return sim.ik(radius*math.cos(angle),radius*math.sin(angle))[0]


def split(start,end):
    r0,t0=start;r1,t1=end
    f00,f10,f01,f11=alpha(r0,t0),alpha(r1,t0),alpha(r0,t1),alpha(r1,t1)
    radial=.5*((f10-f00)+(f11-f01));angular=.5*((f01-f00)+(f11-f10))
    np.testing.assert_allclose(radial+angular,f11-f00,rtol=0,atol=2e-15)
    return dict(radial_contribution_rad=radial,angular_contribution_rad=angular,total_delta_alpha_rad=f11-f00)


def run():
    assert not (OUT/'handover_audit.json').exists()
    c=json.loads((OUT/'completion.json').read_text());a=json.loads((OUT/'source_admission.json').read_text())
    assert c['verified'] and all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    test=split((.11668,-.16665),(.116225,-.16499));assert test['radial_contribution_rad']>0 and test['angular_contribution_rad']<0
    m=mujoco.MjModel.from_xml_path(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'))
    qids=[int(m.jnt_qposadr[m.joint(n).id]) for n in ('alphaR','betaR')]
    vids=[int(m.jnt_dofadr[m.joint(n).id]) for n in ('alphaR','betaR')]
    records=[];series={};inputs={};peaks={}
    for entry in c['records']:
        directory=OUT/entry['condition'];result=json.loads((directory/'result.json').read_text())
        for row in result['runs']:
            f=directory/row['complete_trace']['path'];assert sha(f)==row['complete_trace']['sha256'];inputs[str(f.relative_to(ROOT))]=sha(f)
            with np.load(f) as z:
                t=z['trace'];post=z['post'];pre=z['pre'];cols={str(k):i for i,k in enumerate(z['columns'])}
            joint=post[:,qids[0]];speed=post[:,17+vids[0]];cmd=t[:,cols['speed_command']]
            seen=np.maximum.accumulate(cmd!=0);park=int(np.flatnonzero(seen&(cmd==0))[0]);peak=int(np.argmax(abs(joint)))
            clear=park;target_delta=None;ref=None
            if 'joint_reference_trace' in row:
                file=directory/row['joint_reference_trace']['path'];inputs[str(file.relative_to(ROOT))]=sha(file)
                with np.load(file) as z:ref=z['trace']
                clear=int(np.flatnonzero((np.arange(len(t))>=park)&(ref[:,22:30]==0).all(axis=1))[0])
                target_delta=(ref[clear,5:7]-ref[park-1,5:7]).tolist()
            def state(i):
                pose=sim.fk_joints(*post[i,qids]);radius=pose['leg_len'];angle=pose['phi5']+math.pi/2
                np.testing.assert_allclose(alpha(radius,angle),joint[i],rtol=0,atol=4e-15)
                margin=1.4-abs(float(joint[i]));outward=float(np.sign(joint[i])*speed[i])
                return dict(step=i+1,time_s=float(t[i,3]),alpha_rad=float(joint[i]),alpha_speed_rad_s=float(speed[i]),margin_rad=margin,
                    radial_length_m=radius,leg_angle_rad=angle,body_vx_m_s=float(t[i,cols['post_body_vx']]),
                    required_constant_deceleration_rad_s2=outward*outward/(2*margin) if margin>0 and outward>0 else None)
            points={name:state(i) for name,i in [('withdrawal',park),('filters_cleared',clear),('peak',peak)]}
            path=[(sim.fk_joints(*post[i,qids])['leg_len'],sim.fk_joints(*post[i,qids])['phi5']+math.pi/2) for i in range(park,peak+1)]
            contributions=[split(path[i],path[i+1]) for i in range(len(path)-1)]
            radial=sum(v['radial_contribution_rad'] for v in contributions);angular=sum(v['angular_contribution_rad'] for v in contributions)
            np.testing.assert_allclose(radial+angular,joint[peak]-joint[park],rtol=0,atol=1e-12)
            record=dict(condition=entry['condition'],case=row['seed'],points=points,reference_reset_seconds=(clear-park)*.0005,target_shift_m=target_delta,
                max_post_withdrawal_alpha_speed_rad_s=float(speed[park:peak+1].max()),path_radial_contribution_rad=radial,path_angular_contribution_rad=angular,
                design_pass=row['design_joint_passed'],task_success=row['success'])
            records.append(record);peaks[entry['condition'],row['seed']]=(points['peak']['radial_length_m'],points['peak']['leg_angle_rad'])
            if row['seed']==32500000:
                end=min(len(t),park+900);rads=[];angs=[]
                for q in post[park-100:end,qids]:
                    fk=sim.fk_joints(*q);rads.append(fk['leg_len']);angs.append(fk['phi5']+math.pi/2)
                series[entry['condition']]=dict(relative_time=t[park-100:end,3]-t[park,3],joint=joint[park-100:end],radius=np.array(rads),angle=np.array(angs))
    comparisons=[]
    for label in ('frozen_M_det','frozen_M_stochastic'):
        for case in [r['seed'] for r in json.loads((OUT/'proposal.json').read_text())['cases']]:
            comparisons.append(dict(condition=label,case=case,compared_with='joint_zero',peak_pose_difference=split(peaks['joint_zero',case],peaks[label,case])))
    assert len(records)==20 and len(comparisons)==8
    plot(series)
    atomic_json(OUT/'handover_audit.json',dict(round=326,verified=True,records=records,peak_pose_comparisons=comparisons,source_sha256={**inputs,'wheelleg_warp/audit_parking_handover.py':sha(__file__)},
        interpretation=['learnedtrajectory startsparking withlarger angleposition margin andalphaR initiallymoving inward;ascalarinitialmargin/velocity story isinsufficient',
            'mean/d targets restore originalNom within milliseconds whilebase braking changesactuallegangle;observed radial/angular motion both driveactivejoint trajectory',
            'symmetric exactIK decomposition iskinematic attribution,notcausal forceallocation orvalidatedstopping certificate',
            'all20taskfail independentlyofdesign,so newlearningrequires baseline/taskability review aswell asjointtransition handling'],
        limits='single seen engineeringcase;policyfixed here,originaltraining policyvaried;peak comparisons aredifferent peak times;required deceleration isnecessary constant-braking proxy withno certifiedavailableacceleration bound',
        model_decision='Closecommonmean referencelearning. Return tooriginalfixedNom differential-force/hip/yaw thesis scope onlywith explicitshared joint-transition controller admission andstrongsame-information baseline;no immediatenewPPO orrenamedmeanvariant.',
        new_physics=0,new_controller_queries=0,new_training_samples=0,formal5_admitted=False))
    print('PASS326:20handover streams/8exactpeak-pose decompositions,0physics/query/learning')


def plot(series):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True,layout='constrained')
    for label in ('joint_zero','joint_yaw','frozen_M_det','frozen_M_stochastic'):
        s=series[label];axes[0].plot(s['relative_time'],s['joint'],label=label)
        axes[1].plot(s['relative_time'],s['radius']*1000)
        axes[2].plot(s['relative_time'],np.rad2deg(s['angle']))
    axes[0].axhline(1.4,color='black',ls='--',lw=1,label='active design limit')
    axes[1].axhline(115,color='black',ls='--',lw=1)
    for ax in axes:ax.axvline(0,color='gray',ls=':');ax.grid(alpha=.2)
    axes[0].set_ylabel('alphaR (rad)');axes[1].set_ylabel('FK radial length (mm)');axes[2].set_ylabel('FK leg angle (deg)')
    axes[2].set_xlabel('Time after policy withdrawal (s)');axes[0].legend(fontsize=8,ncol=2)
    fig.suptitle('One seen case: frozen-policy parking diagnosis (not a performance test)',fontsize=11)
    fig.savefig(OUT/'parking_handover.png',dpi=180);fig.savefig(OUT/'parking_handover.svg');plt.close(fig)


if __name__=='__main__':run()
