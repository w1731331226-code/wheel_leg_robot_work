"""Reconstruct the frozen reward and episode-start return from accepted flat trajectories."""
import json
import numpy as np
from run_wheel_interference import OUT,STUDY,inputs
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def rewards(trace,guard,success,gamma):
    n=len(trace);assert n and guard.shape==(n,62) and 0<gamma<1
    residual=(guard[:,20:26]-guard[:,14:20])/np.array([40.]*4+[4.5]*2)
    difference=np.diff(residual,axis=0,prepend=np.zeros((1,6)))/.0005
    parts=dict(speed=.0005*np.exp(-((trace[:,44]-trace[:,43])/.25)**2),
        roll=-.0005*(trace[:,7]/.08726646)**2,pitch=-.0005*(trace[:,8]/.08726646)**2,
        yaw=-.0005*(trace[:,9]/.08726646)**2,effort=-.0005*.05*np.sum(residual**2,axis=1),
        smooth=-.0005*.0001*np.sum(difference**2,axis=1),terminal=np.zeros(n))
    parts['terminal'][-1]=10. if success else -10.
    starts=np.arange(0,n,40);discount=gamma**np.arange(len(starts))
    packets={k:np.add.reduceat(v,starts) for k,v in parts.items()}
    total=sum(parts.values());policy_reward=np.add.reduceat(total,starts).astype(np.float32)
    return dict(undiscounted=float(total.sum()),discounted_float32=float(discount@policy_reward),
        components_undiscounted={k:float(v.sum()) for k,v in parts.items()},components_discounted={k:float(discount@v) for k,v in packets.items()},
        policy_steps=len(starts),terminal_discount_weight=float(discount[-1]),
        float32_discount_quantization=float(discount@policy_reward-sum(discount@v for v in packets.values())))


def self_check():
    t=np.zeros((41,112));g=np.zeros((41,62));r=rewards(t,g,True,.5)
    assert r['policy_steps']==2 and r['terminal_discount_weight']==.5
    np.testing.assert_allclose(r['undiscounted'],10.0205,rtol=0,atol=1e-12)
    np.testing.assert_allclose(r['discounted_float32'],5.02025,rtol=0,atol=2e-7)
    g[1,24]=.45;r=rewards(t,g,False,.5)
    np.testing.assert_allclose(r['components_undiscounted']['smooth'],-.004,rtol=0,atol=1e-14)
    assert r['components_undiscounted']['terminal']==-10
    print('PASS34340substep packets/partialterminal/actor-clock discount/zeroinitial residual/smooth pulse',flush=True)


def run():
    self_check();p=inputs();assert not (OUT/'reward_audit.json').exists()
    review=json.loads((OUT/'review.json').read_text());completion=json.loads((OUT/'completion.json').read_text())
    assert review['verified'] and review['reproduction_gate_passed'] and review['completion_sha256']==sha(OUT/'completion.json')
    training=json.loads((STUDY/'proposal.json').read_text());gamma=training['ppo']['gamma'];assert not training['normalization']['norm_reward']
    assert sha(ROOT/'wheelleg_warp/native/environment.py')==p['source_sha256']['wheelleg_warp/native/environment.py']
    records={};source={};max_error=0.
    for entry in completion['records']:
        f=OUT/entry['path'];assert sha(f)==entry['sha256'];row=json.loads(f.read_text())['runs'][0];directory=f.parent
        assert row['reason']=='completed' and not row['TimeLimit.truncated'] and not row['scenario']['relative_attitude']
        assert row['scenario']['terrain']=='legacy' and row['scenario']['height_l']==row['scenario']['height_r']==0
        arrays=[]
        for field in ('complete_trace','joint_guard_trace'):
            path=directory/row[field]['path'];digest=sha(path);assert digest==row[field]['sha256']==review['raw_sha256'][str(path.relative_to(ROOT))]
            source[str(path.relative_to(ROOT))]=digest
            with np.load(path,allow_pickle=False) as z:arrays.append(z['trace'])
        result=rewards(*arrays,row['success'],gamma);error=abs(result['undiscounted']-row['episode']['r'])
        assert error<=1e-9;max_error=max(max_error,error)
        result.update(success=row['success'],yaw_peak_deg=row['peak_deg'][2],archived_episode_return=row['episode']['r'],reconstruction_error=error,
            result_sha256=sha(f),physical_steps=row['physical_steps'])
        records[entry['condition']]=result
    pairs={}
    for model in p['models']:
        label=model['label'];original=records[label+'_original'];masked=records[label+'_wheel_off']
        pairs[label]=dict(undiscounted_gain=masked['undiscounted']-original['undiscounted'],discounted_gain=masked['discounted_float32']-original['discounted_float32'],
            discounted_component_gains={k:masked['components_discounted'][k]-original['components_discounted'][k] for k in original['components_discounted']},
            task_improved=not original['success'] and masked['success'],yaw_improved=masked['yaw_peak_deg']<original['yaw_peak_deg'])
    all_aligned=all(r['discounted_gain']>0 and r['yaw_improved'] for r in pairs.values())
    atomic_json(OUT/'reward_audit.json',dict(round=343,verified=True,episodes=13,first_episode_world_steps=sum(r['physical_steps'] for r in records.values()),gamma=gamma,actor_dt_s=.02,
        reward_definition='FrozenNative after():speed-attitude-executedresidual-smooth at0.5ms,terminal±10. Sum40substeps thenVecNormalize norm_reward=False float32. Discount per20ms actorpacket,notperphysicsstep.',
        records=records,pairs=pairs,all_six_yaw_improvements_have_higher_discounted_return=all_aligned,max_episode_reconstruction_error=max_error,
        review_sha256=sha(OUT/'review.json'),training_proposal_sha256=sha(STUDY/'proposal.json'),source_sha256=source,auditor_sha256=sha(__file__),new_physics_steps=0,new_learning_samples=0,formal5_admitted=False,
        interpretation='This realised episode-start return ranking tests only these fixed deterministic trajectories. It isnotGAE/criticcalibration,theon-policy stochastic trainingdistribution,globalreward-task alignment orconvergence proof. Lower terminaldiscountweight alone doesnotprove wrongranking. No reward/gamma change oroldcandidate revival.',
        next='344 finite method-level synthesis of accepted reward/interference andtraining/exploration limits before345direction review;no extra maskruns ornewlearning without a distinct preregistered question.'))
    print('DONE343 rewardreconstruction error',max_error,'allsix discountedaligned',all_aligned,flush=True)
    for label,result in pairs.items():print(label,'discounted gain',result['discounted_gain'],flush=True)


if __name__=='__main__':run()
