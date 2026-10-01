"""Independent nominal pose-request projection and six normal full-episode gate."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
from native.controller import D,project_leg_angle
from native.environment import NativeEnv
from native.terrain import bank_height_115,HEIGHT_115_GEOMETRIC_MIN
from probe_height_115_margin import cases
from probe_height_angle_envelope import inverse
from probe_height_115_action_predict_loow import sha


@wp.kernel
def project(inputs:wp.array2d[D],out:wp.array2d[D]):
    w=wp.tid();a,valid=project_leg_angle(inputs[w,0],inputs[w,1],inputs[w,2])
    out[w,0]=a;out[w,1]=D(int(valid))


def run(output,coordinated=False,radial_guard=False):
    assert not output.exists(),output;wp.init();wp.set_device('cuda:0')
    inputs=np.array([[h,a,1.4] for h in np.linspace(HEIGHT_115_GEOMETRIC_MIN,.38,25) for a in (-.8,-.4,-.15,0,.15,.4,.8)])
    out=wp.zeros((len(inputs),2),dtype=D);wp.launch(project,len(inputs),[wp.array(inputs,dtype=D),out]);got=out.numpy()
    assert np.all(got[:,1]==1)
    for (h,a,cap),(projected,_) in zip(inputs,got):
        q=inverse(h,projected);assert q is not None and max(map(abs,q))<=cap+1e-7
        original=inverse(h,a)
        if original is not None and max(map(abs,original))<=cap:assert projected==a
    scenes=cases()[:6];found=[];counts=[];peaks=[]
    for enabled in (False,True):
        env=NativeEnv(n=6,scenario=scenes,bank_factory=bank_height_115,height_conditioned=True,
            height_design='range115',residual_scale=0,feasible_reference=enabled,coordinated_reference=enabled and coordinated,radial_guard=enabled and radial_guard)
        try:
            obs=env.reset();assert np.allclose(obs[:,11],-.185,atol=1e-6)
            result=[None]*6;clipped=np.zeros(6,np.int64);delta=np.zeros(6)
            for _ in range(700):
                _,_,done,infos=env.step(np.zeros((6,3),np.float32))
                if enabled:
                    d=env.diag.numpy();r=env.k['reference'].numpy()
                    assert np.all(d[:,21:23]>=HEIGHT_115_GEOMETRIC_MIN-1e-12) and np.all(d[:,21:23]<=.38+1e-12)
                    np.testing.assert_allclose(d[:,21]+d[:,22],2*r[:,2],atol=1e-12,rtol=0)
                    clipped+=(d[:,28]>0).astype(np.int64)
                    delta=np.maximum(delta,np.max(abs(d[:,23:25]-d[:,25:27]),axis=1))
                for i in np.flatnonzero(done):
                    if result[i] is None:result[i]={k:v for k,v in infos[i].items() if k!='terminal_observation'}
                if all(x is not None for x in result):break
            assert all(x is not None for x in result)
            assert all(x['physical_evidence_steps']==x['physical_steps'] for x in result)
            found.append(result);counts.append(clipped.tolist());peaks.append(delta.tolist())
            print('PANEL',enabled,'success',sum(x['success'] for x in result),'physical',sum(x['physical_safety_passed'] for x in result),flush=True)
        finally:env.close()
    result=dict(role='public_nominal_reference_projection_trial',geometry_samples=len(inputs),projection_math_pass=True,
        reference_fields=['alpha_ref','beta_ref','nominal_height','minimum_reference','mean_preserving_radial','request_joint_cap','maximum_reference'],
        projection_rule='Radial targets keep their original mean and fit declared reference interval. Angular static-equivalent position requests use connected standing IK with1.4rad design cap; retain common angular velocity feedback and unchanged wheel LQR.',
        nominal_height_m=.115,nominal_speed_targets_unchanged=True,physical_limits_unchanged=True,coordinated_reference=coordinated,radial_guard=radial_guard,
        radial_guard_rule='Common nonnegative force from public mass8kg plus nominal reflected hip inertia; nominal-target tracking expression -2k*rate-k^2*(L-target), k=200/s from5ms horizon; analytic hip headroom preserves angular requests. Not a certified dynamics bound.' if radial_guard else None,
        coordination_rule='delta_wheel=(K_wheel_v/K_hub_v)*delta_mean_hub, same virtual velocity-reference change in both LQR inputs' if coordinated else None,
        scenarios=[asdict(s) for s in scenes],baseline=found[0],candidate=found[1],
        policy_boundary_projection_counts=counts[1],maximum_requested_angle_change_rad=peaks[1],
        full_normal_gate_pass=all(r['success'] for r in found[1]),
        limitations='Projection is of static-equivalent requests, not a dynamic invariance proof.1.4rad is existing0.1rad model design reserve, not measured hardware margin. Boundary diagnostics sampled each20ms; no clipping frequency or hard real-time claim. No PPO or final holdout.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/controller.py',
            ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/probe_height_angle_envelope.py')})
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED',result['full_normal_gate_pass'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--coordinated',action='store_true');parser.add_argument('--radial-guard',action='store_true');args=parser.parse_args()
    run(args.output,args.coordinated,args.radial_guard)
