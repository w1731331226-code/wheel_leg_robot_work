"""Conventional pair-reference projection and original-control static response, no rollout."""
import json
import numpy as np
import warp as wp

from nom_yaw_filter_probe import rec
from test_nom_yaw_filter_probe import control_args
from train_height_comparison import raw_env
import wheelleg_sim as sim

OUT=rec.ROOT/'wheelleg_warp/results/paper_recovery_20261004/nom_yaw_filter_probe_v1'


def pair_projection(height,offset,lower,upper):
    if not np.isfinite([height,offset,lower,upper]).all() or not lower<upper or not lower<=height<=upper:
        raise ValueError('Invalid reference limits or height')
    left=float(np.clip(height+offset,lower,upper));right=float(np.clip(height-offset,lower,upper))
    return (left+right)/2,(left-right)/2,left,right


def unit():
    assert pair_projection(.2,.01,.1,.4)==(.2,.010000000000000009,.21000000000000002,.19)
    h,o,l,r=pair_projection(.1,.03,.1,.4)
    np.testing.assert_allclose([h,o,l,r],[.115,.015,.13,.1])
    h,o,l,r=pair_projection(.4,-.03,.1,.4)
    np.testing.assert_allclose([h,o,l,r],[.385,-.015,.37,.4])
    for height in np.linspace(.1147044660616607,.38,17):
        for offset in np.linspace(-.035,.035,17):
            h,o,l,r=pair_projection(height,offset,.1147044660616607,.38)
            assert .1147044660616607<=l<=.38 and .1147044660616607<=r<=.38
            assert abs(h-height)<=abs(offset)/2+1e-15
            room=min(h-.1147044660616607,.38-h)
            np.testing.assert_allclose([h+np.clip(offset,-room,room),h-np.clip(offset,-room,room)],[l,r],atol=1e-15)
    try:pair_projection(float('nan'),0,.1,.4)
    except ValueError:pass
    else:raise AssertionError('Invalid input accepted')


def run():
    unit();p=json.loads((OUT/'proposal.json').read_text())
    c=json.loads((OUT/'source_contract.json').read_text())
    assert all(rec.sha(rec.ROOT/n)==v for n,v in c['source_sha256'].items())
    pair_review=json.loads((OUT/'round178_pair_review.json').read_text());assert pair_review['verified'] and not pair_review['fixed_variant_qualification']
    cases=[next(x for x in p['cases'] if x['scenario']['stand_height_m']==h) for h in (.115,.16,.24,.30,.38)]
    raw=raw_env(cases,'diff3');rows=[]
    try:
        raw.reset();original=raw.k['reference'].numpy().copy()
        for degrees in (-5.,0.,5.):
            raw.reset();q=raw.data.qpos.numpy();q[:,3]=np.cos(np.deg2rad(degrees)/2);q[:,4]=np.sin(np.deg2rad(degrees)/2)
            raw.data.qpos.assign(q)
            # Set controller finite-difference caches using one original control computation,
            # then zero derivatives for this static synthetic pose. No mj_step.
            wp.launch(raw.control_kernel,raw.num_envs,control_args(raw),block_dim=32)
            state=raw.k['state'].numpy();state[:,0]=2.;state[:,3:9]=0.
            baseline=[]
            roll=np.arctan2(2*q[:,3].astype(float)*q[:,4],1-2*q[:,4].astype(float)**2)
            desired=np.clip(.3*roll,-.035,.035)
            for variant in ('fixed_mean','pair_projected_mean'):
                ref=original.copy();projected=[]
                if variant=='pair_projected_mean':
                    for w in range(raw.num_envs):
                        projected.append(pair_projection(ref[w,2],desired[w],ref[w,3],ref[w,6]))
                        ref[w,2]=projected[-1][0]
                raw.k['reference'].assign(ref);raw.k['state'].assign(state)
                wp.launch(raw.control_kernel,raw.num_envs,control_args(raw),block_dim=32)
                diag=raw.diag.numpy();ctrl=raw.data.ctrl.numpy()
                assert np.isfinite(diag).all() and np.isfinite(ctrl).all()
                assert np.all(abs(ctrl[:,:4])<=40+1e-6) and np.all(abs(ctrl[:,4:])<=4.5+1e-6)
                if projected:
                    np.testing.assert_allclose(diag[:,21:23],[x[2:] for x in projected],atol=1e-12,rtol=0)
                for w,case in enumerate(cases):
                    rows.append(dict(case=case['seed'],height_command_m=float(original[w,2]),synthetic_roll_deg=degrees,variant=variant,
                        mean_reference_m=float(ref[w,2]),requested_offset_m=float(desired[w]),
                        selected_offset_m=float((diag[w,21]-diag[w,22])/2),left_target_m=float(diag[w,21]),right_target_m=float(diag[w,22]),
                        radial_guard_reference_m=float(min(ref[w,2],sim.L_SQUAT_MIN)),
                        shorter_target_below_guard_reference=bool(min(diag[w,21],diag[w,22])<min(ref[w,2],sim.L_SQUAT_MIN)-1e-12),
                        requested_common_guard_N=float(diag[w,31]),applied_common_guard_N=float(diag[w,32]),guard_limited=bool(diag[w,37]),
                        pre_final_radial_left_N=float(diag[w,35]),pre_final_radial_right_N=float(diag[w,36]),
                        commanded_torque_Nm=ctrl[w].tolist()))
            raw.k['reference'].assign(original)
        raw.reset();assert not raw.data.time.numpy().any() and not raw.state.numpy()[:,0].any()
    finally:raw.close()
    rec.atomic_json(OUT/'round179_leg_reference_audit.json',dict(verified=True,rows=rows,recorded_static_controller_rows=30,preparatory_static_world_computations=15,
        source_sha256=rec.sha(__file__),controller_sha256=rec.sha(rec.ROOT/'wheelleg_warp/native/controller.py'),
        source_contract_sha256=rec.sha(OUT/'source_contract.json'),pair_review_sha256=rec.sha(OUT/'round178_pair_review.json'),
        facts='Final fleft/fright use each individual actual leg length and selected target. Common mean shifts therefore reach radial force; not canceled by LQR support replacement. Pair clipping can be represented through original mean/room logic, with mean shift<=|desired offset|/2<=17.5mm.',
        classification='Ordinary Euclidean box projection, not a new algorithm. Not admitted alone: raising tracking mean also raises min(mean,0.16m) protective anchor and can hit40Nm hip bounds in low-height synthetic poses.',
        limits='Synthetic root-roll poses and original control computations without physics. Reference bound is not actual leg-height error/RMS/end guarantee. Changed mean also affects radial force and feasible-angle/coordinated wheel branches, so never describe as roll-only intervention.0.16/0.24/0.30 reference identity does not resolve all observed failures.',
        new_physics_evaluations=0,training_updates=0,
        next='180 review one reference-role consistency candidate: distinguish legitimate protective lower limit from tracking mean/left/right targets before any reference-governor dynamics. Constant operating-floor choice is a hypothesis, not proven safe or a new algorithm. No intervention deployed, alpha1 remains closed; no mean-only repair/gain scan/PPO. Full contribution/5seed/independent tests/new manuscript remain open.'))
    print('PASS289 projection fixtures and30 original-controller static computations;0rollout',flush=True)
    for r in rows:
        if r['synthetic_roll_deg']==5 and r['height_command_m'] in (.115,.16,.38):print(r,flush=True)


if __name__=='__main__':run()
