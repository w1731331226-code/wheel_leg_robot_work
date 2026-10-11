"""Finite request-order and scale audit; no native controller or physics execution."""
import itertools
import json
import numpy as np
from cartesian_pair_action import OUT, BASE, F0, H0, W0, H_SUM, request, decode
from state_estimation import leg_kinematics
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json


def slew(previous,target):
    return previous+np.clip(target-previous,-.01,.01)


def compare(legs,reset_legs,previous,target):
    filtered=slew(previous,target)
    live=request(legs,filtered)[0];fixed=request(reset_legs,filtered)[0]
    start=request(legs,previous)[0]
    wrong=slew(start,request(legs,target)[0])
    old=np.array([filtered[0],-filtered[0],filtered[1],-filtered[1],filtered[2],-filtered[2]])
    return np.array([live,fixed,wrong,old]),filtered,start


def self_check():
    legs=np.array([[0.,-.38],[0.,-.38]])
    u,filtered,start=compare(legs,legs,np.zeros(3),np.array([1.,0.,0.]))
    np.testing.assert_array_equal(u[0],u[1])
    np.testing.assert_allclose(filtered,[.01,0,0],rtol=0,atol=0)
    assert abs(u[0]-start).max()>.01
    uneven=np.array([[.04,-.115],[-.03,-.38]])
    u,_,_=compare(uneven,legs,np.zeros(3),np.array([1.,0.,0.]))
    assert abs(decode(uneven,u[0]).sum(axis=0)).max()<1e-12
    assert abs(decode(uneven,u[2]).sum(axis=0)).max()>1e-4
    np.testing.assert_array_equal(compare(uneven,legs,np.zeros(3),np.zeros(3))[0],0)
    assert np.any(compare(uneven,legs,np.ones(3),np.zeros(3))[0][0]!=0)
    print('PASS362 cold-zero, finite slew, wrong-order counterexample and mapped-rate counterexample',flush=True)


def run():
    self_check();p=json.loads((OUT/'delivery_proposal.json').read_text());assert not (OUT/'delivery_review.json').exists()
    parent=json.loads((OUT/'review.json').read_text())
    assert p['parent_review_sha256']==sha(OUT/'review.json') and p['parent_proposal_sha256']==sha(OUT/'proposal.json')
    assert parent['source_sha256']==sha(ROOT/'wheelleg_warp/cartesian_pair_action.py')
    assert all(sha(ROOT/f)==h for f,h in p['source_sha256'].items())
    data=BASE/'reflection_retention_falsification_v1';accepted=json.loads((data/'review.json').read_text())
    cache={};hashes={};all_u=[];all_motor=[];all_latent=[];indices=[];stats=[];initial_equal=0
    targets=np.array(list(itertools.product([-1.,0.,1.],repeat=3)))
    previous=np.asarray(p['fixture']['previous_latents'],float)
    for geometry_index,record in enumerate(parent['records']):
        condition=record['condition']
        if condition not in cache:
            directory=data/condition;row=json.loads((directory/'result.json').read_text())['runs'][0]
            file=directory/row['actor_trace']['path'];digest=sha(file)
            assert digest==parent['input_sha256'][str(file.relative_to(ROOT))]==accepted['raw_sha256'][str(file.relative_to(ROOT))]
            with np.load(file) as z:cache[condition]=z['trace']
            hashes[str(file.relative_to(ROOT))]=digest
        actor=cache[condition];frame=actor[record['actor_index'],351:390];reset=actor[0,351:390]
        parts=[leg_kinematics(frame[12+2*s:14+2*s].astype(float),np.zeros(2)) for s in range(2)]
        legs=np.array([x[0] for x in parts]);np.testing.assert_array_equal(legs,record['legs_xz_m'])
        reset_legs=np.array([leg_kinematics(reset[12+2*s:14+2*s].astype(float),np.zeros(2))[0] for s in range(2)])
        rate_max=wrong_max=live_error=hip_max=0.;rate_count=wrong_count=0
        for ip,prev in enumerate(previous):
            for it,target in enumerate(targets):
                u,latent,start=compare(legs,reset_legs,prev,target)
                error=float(abs(decode(legs,u[0]).sum(axis=0)).max());hip=abs(H0*(u[0,2]+u[0,3]))
                assert error<=1e-10 and hip<=H_SUM+1e-12 and abs(u[0]).max()<=1+1e-12 and u[0,4]+u[0,5]==0
                if record['actor_index']==0:
                    np.testing.assert_array_equal(u[0],u[1]);initial_equal+=1
                wrong=float(np.linalg.norm(decode(legs,u[2]).sum(axis=0)))
                rate=float(abs(u[0]-start).max());rate_count+=rate>.01+1e-12;wrong_count+=wrong>1e-10
                rate_max=max(rate_max,rate);wrong_max=max(wrong_max,wrong);live_error=max(live_error,error);hip_max=max(hip_max,hip)
                motor=np.zeros((4,6))
                for arm in range(4):
                    for side in range(2):motor[arm,2*side:2*side+2]=parts[side][3]@np.array([F0*u[arm,side],H0*u[arm,2+side]])
                    motor[arm,4:6]=W0*u[arm,4:6]
                all_u.append(u);all_motor.append(motor);all_latent.append(latent);indices.append([geometry_index,ip,it])
        stats.append(dict(geometry_index=geometry_index,condition=condition,actor_index=record['actor_index'],live_force_component_error_N=live_error,
            virtual_H_sum_peak_Nm=hip_max,mapped_step_max=rate_max,mapped_rate_exceedances=rate_count,
            wrong_order_force_sum_norm_max_N=wrong_max,wrong_order_nonzero_count=wrong_count))
    u=np.asarray(all_u);motor=np.asarray(all_motor)
    assert len(u)==p['fixture']['combinations']==55296 and initial_equal==8*27*4
    artifact=OUT/'delivery_fixtures.npz';np.savez_compressed(artifact,indices=np.array(indices),filtered_latent=np.array(all_latent),canonical=u,
        motor_request_Nm=motor,arms=np.array(['live_geometry','reset_geometry','wrong_order','original_D3']))
    rms=np.sqrt(np.mean(motor**2,axis=0));assert (rms>0).all()
    report=dict(round=362,verified=True,geometry_rows=512,fixture_combinations=len(u),initial_map_equal_comparisons=initial_equal,
        live_force_component_error_N=max(x['live_force_component_error_N'] for x in stats),
        live_H_sum_max_Nm=max(x['virtual_H_sum_peak_Nm'] for x in stats),mapped_canonical_step_max=max(x['mapped_step_max'] for x in stats),
        mapped_canonical_rate_exceedances=sum(x['mapped_rate_exceedances'] for x in stats),
        wrong_order_force_sum_max_N=max(x['wrong_order_force_sum_norm_max_N'] for x in stats),wrong_order_nonzero_count=sum(x['wrong_order_nonzero_count'] for x in stats),
        motor_grid_rms_Nm=rms.tolist(),live_to_original_D3_grid_rms_ratio=(rms[0]/rms[3]).tolist(),live_to_reset_grid_rms_ratio=(rms[0]/rms[1]).tolist(),
        geometry_statistics=stats,fixture_sha256=sha(artifact),input_sha256=hashes,proposal_sha256=sha(OUT/'delivery_proposal.json'),parent_review_sha256=sha(OUT/'review.json'),auditor_sha256=sha(__file__),
        runtime_admitted=False,new_physics_steps=0,new_model_forward_rows=0,new_learning_samples=0,formal5_admitted=False,
        interpretation='Raw virtual/VMC request arithmetic at fixed saved geometry, before all actuator/joint guard stages. Not real control-step delivery, new plant trajectories, Gaussian exploration, or policy advantage. Latent slew differs from canonical slew even without motion. Live/reset equality only at reset; later differences are intended and not claimed equal distributions.')
    atomic_json(OUT/'delivery_review.json',report)
    print('PASS362',len(u),'fixtures;initial equal',initial_equal,'mapped max step',report['mapped_canonical_step_max'],
        'rate exceeds',report['mapped_canonical_rate_exceedances'],'wrong-order force',report['wrong_order_force_sum_max_N'],'RMS ratios',report['live_to_original_D3_grid_rms_ratio'],flush=True)


if __name__=='__main__':run()
