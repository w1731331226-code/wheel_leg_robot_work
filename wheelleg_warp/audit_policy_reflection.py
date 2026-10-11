"""Bounded raw481 reflection/mean-response audit; no robot integration or policy updates."""
import json
import pickle
import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO
from run_wheel_interference import OUT as DIAG,inputs
from fixed_reference_force import embed
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

OUT=DIAG.parent/'policy_reflection_audit_v1'
PACKET=np.arange(39)
for left,right in ((12,14),(13,15),(16,18),(17,19),(20,21),(22,23),(24,25),(26,28),(27,29),(30,31),(32,33),(34,35),(36,37)):
    PACKET[left],PACKET[right]=right,left
MOTOR=[2,3,0,1,5,4]
ACTION=[1,0,3,2,5,4]


def reflect_raw(raw):
    x=np.asarray(raw)
    if x.ndim!=2 or x.shape[1]!=481 or not np.isfinite(x).all():raise ValueError('Finite raw481 batches required')
    y=x.copy();frames=y[:,:390].reshape(-1,10,39)
    frames[:]=x[:,:390].reshape(-1,10,39)[:,:,PACKET]
    frames[:,:,[0,2,3,5,7,38]]*=-1
    old=x[:,390:462].reshape(-1,9,8);new=y[:,390:462].reshape(-1,9,8)
    if old[:,:,6:].any():raise ValueError('This H1 contract requires zero proxy slots')
    new[:,:,:6]=old[:,:,MOTOR]
    return y


def self_check():
    x=np.random.default_rng(34531).normal(size=(4,481)).astype(np.float32);x[:,390:462].reshape(-1,9,8)[:,:,6:]=0
    np.testing.assert_array_equal(reflect_raw(reflect_raw(x)),x)
    y=reflect_raw(x);f=x[:,:390].reshape(-1,10,39);r=y[:,:390].reshape(-1,10,39)
    np.testing.assert_array_equal(r[:,:,11],f[:,:,11])
    np.testing.assert_array_equal(r[:,:,26:32],f[:,:,26:32][:,:,MOTOR])
    np.testing.assert_array_equal(r[:,:,32:38],f[:,:,32:38][:,:,ACTION])
    np.testing.assert_array_equal(y[:,462:],x[:,462:])
    latent=np.ones((4,3),np.float32);np.testing.assert_array_equal(embed(latent,'D3',4)[:,ACTION],embed(-latent,'D3',4))
    for bad in (np.zeros((2,480)),np.full((2,481),np.nan),np.ones((2,481))):
        try:reflect_raw(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid reflection contract accepted')


def morphology():
    spec=mujoco.MjSpec.from_file(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'));sim.hw.configure_spec(spec);m=spec.compile();F=np.diag([1.,-1.,1.])
    def inertia(i):
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,m.body_iquat[i]);rotation=rotation.reshape(3,3)
        return rotation@np.diag(m.body_inertia[i])@rotation.T
    pairs=[('chassis','chassis'),('legL','legR'),('legL_D','legR_D'),('kneeA_L','kneeA_R'),('kneeB_L','kneeB_R'),('wheelL','wheelR')];result=[]
    for l,r in pairs:
        i,j=m.body(l).id,m.body(r).id
        result.append(dict(left=l,right=r,mass_difference=float(abs(m.body_mass[i]-m.body_mass[j])),
            position_error_m=float(abs(F@m.body_pos[i]-m.body_pos[j]).max()),com_error_m=float(abs(F@m.body_ipos[i]-m.body_ipos[j]).max()),
            inertia_reflection_error_kgm2=float(abs(F@inertia(i)@F-inertia(j)).max())))
    return result


def run():
    self_check();p=inputs();review=json.loads((DIAG/'review.json').read_text());assert review['verified'] and not OUT.exists()
    OUT.mkdir();registration=[]
    for model in p['models']:
        file=DIAG/(model['label']+'_original')/'actor_world_0.npz';assert sha(file)==review['raw_sha256'][str(file.relative_to(ROOT))]
        with np.load(file) as z:count=len(z['trace'])
        assert count>=64;indices=np.linspace(0,count-1,64,dtype=int).tolist();assert len(set(indices))==64
        registration.append(dict(model=model,path=str(file.relative_to(ROOT)),sha256=sha(file),indices=indices))
    atomic_json(OUT/'registration.json',dict(round=345,records=registration,raw_sample_count=384,model_forward_rows=768,
        selection='All6frozen finalmodels,64uniform chronological savedrows each includingreset;no scores/seed selection',
        packet_permutation=PACKET.tolist(),sign_flip=[0,2,3,5,7,38],motor_permutation=MOTOR,action_permutation=ACTION,
        source_sha256={f:p['source_sha256'][f] for f in ('wheelleg_warp/native/environment.py','wheelleg_warp/execution_history_env.py','wheelleg_warp/route_state.py','wheelleg_warp/fixed_reference_force.py')},
        audit_sha256=sha(__file__),new_physics_steps=0,new_learning_samples=0))
    torch.set_num_threads(1);records=[]
    for item in registration:
        model=item['model'];file=ROOT/item['path']
        with np.load(file) as z:x=z['trace'][item['indices'],:481].copy()
        sx=reflect_raw(x);np.testing.assert_array_equal(reflect_raw(sx),x)
        with (ROOT/model['normalization']).open('rb') as f:norm=pickle.load(f)
        agent=PPO.load(ROOT/model['model'],device='cpu');assert agent.num_timesteps==200000
        u=embed(agent.predict(norm.normalize_obs(x.copy()),deterministic=True)[0],model['arm'],64)
        su=embed(agent.predict(norm.normalize_obs(sx.copy()),deterministic=True)[0],model['arm'],64)
        defect=u-su[:,ACTION];projected=.5*(u+su[:,ACTION]);projected_s=.5*(su+u[:,ACTION])
        np.testing.assert_array_equal(projected_s,projected[:,ACTION]);assert abs(projected).max()<=1
        np.testing.assert_array_equal(sx[0],x[0])
        if model['arm']=='D3':np.testing.assert_array_equal(projected[0],0)
        normalized_commutator=norm.normalize_obs(sx.copy())-reflect_raw(norm.normalize_obs(x.copy()))
        records.append(dict(model=model['label'],raw_rows=64,canonical_mean_equivariance_defect_RMS=float(np.sqrt(np.mean(defect**2))),
            max_defect=float(abs(defect).max()),reset_original6=u[0].tolist(),reset_projected6=projected[0].tolist(),
            raw_before_normalize_vs_naive_normalized_reflection_max_difference=float(abs(normalized_commutator).max()),
            projected_equivariance_algebra_passed=True,model_sha256=sha(ROOT/model['model']),RMS_sha256=sha(ROOT/model['normalization'])))
    atomic_json(OUT/'review.json',dict(round=345,verified=True,records=records,nominal_static_morphology=morphology(),registration_sha256=sha(OUT/'registration.json'),auditor_sha256=sha(__file__),
        model_forward_rows=768,new_physics_steps=0,new_learning_samples=0,formal5_admitted=False,
        interpretation='Rawobservation/action C2 algebra andfrozenmean non-equivariance established on384savedinputs. Projectedmeans calculatedoffline only,notappliedtorobot. Motorresidual26:32 blockswaps differfrom virtualrequest32:38 pairswaps;slot11height iseven. Symmetry mustprecede frozenRMS,notnaive normalizedswap.',
        limits='Staticnominal morphology comparison isnot full contact/terrain/control/dynamics equivariance proof. Scenario reflection requires side/friction/drive/geometry/contact-mask transformations;globalrandom roughterrain mappings unverified. Smallcompiledinertia asymmetry isreported ratherthan zeroed. Known groupaveraging isnotnewtheory orperformance advantage.'))
    print('PASS345384public states/768frozenforward rows/algebra,0robotphysics/learning',flush=True)


if __name__=='__main__':run()
