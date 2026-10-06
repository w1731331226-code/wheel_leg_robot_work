"""All newly introduced primary design violations; saved trajectories only."""
import json
from collections import Counter
import numpy as np
import nominal_mean_main as main
from audit_floor_design_crossing import intervals, NAMES
from native.terrain import HeightTerrainScenario, model


def crossing(pre_q, post_q, pre_v):
    margins = 1.4-np.abs(post_q)
    bad = margins.min(axis=1) < 0
    spans = intervals(bad)
    if not spans:
        return dict(violating_samples=0, first=None, worst_margin_rad=float(margins.min()))
    i = int(spans[0][0]); j = int(margins[i].argmin())
    return dict(violating_samples=int(bad.sum()), first=i, joint=NAMES[j],
                worst_margin_rad=float(margins.min()),
                pre_margin_rad=float(1.4-abs(pre_q[i,j])),
                outward_pre_velocity_rad_s=float(np.sign(pre_q[i,j])*pre_v[i,j]),
                spans=[dict(first_step=int(a+1),last_step=int(b+1)) for a,b in spans])


def unit():
    q = np.array([[0.,0.,0.,0.],[1.39,0.,0.,0.]])
    post = q.copy(); post[1,0] = 1.41
    v = np.zeros_like(q); v[1,0] = 2.
    r = crossing(q,post,v)
    json.dumps(r,allow_nan=False)
    assert r['first']==1 and r['violating_samples']==1 and r['outward_pre_velocity_rad_s']==2
    assert crossing(q,q,v)['first'] is None
    q[:,0] *= -1; post[:,0] *= -1; v[:,0] *= -1
    assert crossing(q,post,v)['outward_pre_velocity_rad_s']==2


def run():
    unit(); p=main.verify(); out=main.OUT; root=main.ROOT
    review_file=out/'study_pair_review.json'; review=json.loads(review_file.read_text())
    assert review['verified'] and not review['formal_expansion_gate_passed']
    assert review['main_completion_sha256']==main.sha(out/'main_completion.json')
    inputs={str(review_file.relative_to(root)):main.sha(review_file)}; rows={}; old={}; cache={}
    def read(file,expected):
        assert main.sha(file)==expected;inputs[str(file.relative_to(root))]=expected
        return json.loads(file.read_text())['runs']
    for ref in p['references']['phase_result_refs']:
        if ref['label']=='B0' and ref['panel'] in ('regular','controlled'):
            file=root/ref['path']
            for row in read(file,ref['sha256']):old[(ref['panel'],row['seed'])]=(row,file.parent)
    for seed in p['seeds']:
        file=out/'runs/anchored'/str(seed)/'verification.json';v=json.loads(file.read_text())
        inputs[str(file.relative_to(root))]=main.sha(file)
        for ref in v['evaluation_records']:
            if ref['panel'] in ('regular','controlled'):
                f=out/ref['path']
                for row in read(f,ref['sha256']):rows[(seed,ref['panel'],row['seed'])]=(row,f.parent)
    def inspect(row,directory):
        key=str(directory/row['complete_trace']['path'])
        if key in cache:return cache[key]
        arrays={}
        for kind in ('complete_trace','role_trace','phase_trace'):
            f=directory/row[kind]['path'];assert main.sha(f)==row[kind]['sha256']
            inputs[str(f.relative_to(root))]=row[kind]['sha256']
            with np.load(f,allow_pickle=False) as z:
                arrays[kind]={k:z[k] for k in (('trace','pre','post','columns','state_sizes') if kind=='complete_trace' else ('trace','columns'))}
        a=arrays['complete_trace'];t,pre,post=a['trace'],a['pre'],a['post'];c={str(k):i for i,k in enumerate(a['columns'])}
        # Compile the same model for joint addresses only; no forward/step/rollout.
        cpu=model(HeightTerrainScenario(**row['scenario']))
        assert a['state_sizes'].tolist()==[cpu.nq,cpu.nv,cpu.nu]
        qi=[int(cpu.joint(n).qposadr[0]) for n in NAMES];vi=[cpu.nq+int(cpu.joint(n).dofadr[0]) for n in NAMES]
        assert len(t)==row['physical_steps'] and np.isfinite(pre).all() and np.isfinite(post).all()
        np.testing.assert_array_equal(pre[1:,:cpu.nq+cpu.nv],post[:-1,:cpu.nq+cpu.nv])
        np.testing.assert_array_equal(t[:,c['step']],np.arange(1,len(t)+1))
        margin=(1.4-np.abs(post[:,qi])).min(axis=1)
        np.testing.assert_allclose(margin,t[:,c['design_margin']],atol=1e-12,rtol=0)
        assert margin.min()==row['min_active_design_margin_rad']
        result=crossing(pre[:,qi],post[:,qi],pre[:,vi])
        assert (result['first'] is None)==row['design_joint_passed']
        role=arrays['role_trace'];phase=arrays['phase_trace']
        rc={str(k):i for i,k in enumerate(role['columns'])};pc={str(k):i for i,k in enumerate(phase['columns'])}
        for z,cols in ((role,rc),(phase,pc)):
            assert np.isfinite(z['trace']).all();np.testing.assert_array_equal(z['trace'][:,cols['step']],t[:,c['step']])
        if result['first'] is not None:
            i=result['first'];load=t[:,c['left_target_normal_N']]+t[:,c['right_target_normal_N']];hit=np.flatnonzero(load>1e-6)
            result.update(first_post_s=float(t[i,c['post_s']]),command_zero=bool(phase['trace'][i,pc['zero_command']]),
                arrival_s=row['arrival_s'],first_crossing_after_arrival=(bool(t[i,c['pre_s']]>=row['arrival_s']) if row['arrival_s'] is not None else None),
                guard_lower_m=float(phase['trace'][i,pc['guard_lower']]),
                guard_requested_N=float(role['trace'][i,rc['guard_requested']]),guard_applied_N=float(role['trace'][i,rc['guard_applied']]),
                residual_lambda=float(t[i,c['original_lambda']]),first_original_target_load_pre_s=float(t[hit[0],c['pre_s']]) if len(hit) else None,
                before_first_original_target_load=bool(i<hit[0]) if len(hit) else None,
                post_roll_pitch_yaw_rad=t[i,[c['roll'],c['pitch'],c['yaw']]].tolist(),
                filtered_virtual_residual=t[i,[c['filtered_F_left'],c['filtered_F_right'],c['filtered_H_left'],c['filtered_H_right'],c['filtered_wheel_left'],c['filtered_wheel_right']]].tolist(),
                pre_commanded_motor_torques=pre[i,cpu.nq+cpu.nv:].tolist())
        cache[key]=result;return result
    selected=[]
    for seed in p['seeds']:
        for panel in ('regular','controlled'):
            pair=review['pairs'][f'anchored/{seed}/{panel}/vs_B0']
            for failure in pair['new_component_failures']:
                if 'design' not in failure['flags']:continue
                case=failure['case'];row,d=rows[(seed,panel,case)];ref,rd=old[(panel,case)]
                assert row['scenario']==ref['scenario'] and ref['design_joint_passed'] and not row['design_joint_passed']
                selected.append(dict(seed=seed,panel=panel,case=case,height_m=row['target_leg_m'],terrain=row['scenario']['terrain'],
                    original_success=ref['success'],candidate_success=row['success'],candidate=inspect(row,d),reference=inspect(ref,rd)))
    assert len(selected)==26
    counts=dict(records=len(selected),unique_cases=len({r['case'] for r in selected}),
        first_crossing_zero_command=sum(r['candidate']['command_zero'] for r in selected),
        first_crossing_after_arrival=sum(r['candidate']['first_crossing_after_arrival'] is True for r in selected),
        first_crossing_before_recorded_original_target_load=sum(r['candidate']['before_first_original_target_load'] is True for r in selected),
        first_crossing_outward_velocity=sum(r['candidate']['outward_pre_velocity_rad_s']>0 for r in selected),
        first_crossing_guard_zero=sum(r['candidate']['guard_applied_N']==0 for r in selected),
        first_crossing_residual_unlimited=sum(r['candidate']['residual_lambda']==1 for r in selected),
        by_joint=dict(Counter(r['candidate']['joint'] for r in selected)))
    main.write(out/'round209_failure_mechanism.json',dict(verified=True,round=209,selection='Every new design flag versus B0 in both primary panels and all three anchored seeds; no outcome-picked trajectory.',
        counts=counts,records=selected,input_sha256=inputs,source_sha256=main.sha(__file__),
        dependency_sha256={n:main.sha(root/n) for n in ['wheelleg_warp/audit_floor_design_crossing.py','wheelleg_warp/native/controller.py','wheelleg_warp/native/environment.py','wheelleg_warp/nominal_packet_reference.py','wheelleg_warp/nominal_mean_policy.py']},
        new_training_or_evaluations=0,limits='Sampled pre/post states, not continuous invariance. Original-target normal witnesses do not cover every terrain support. Filtered residual is controller state,not raw actor sample. Paired classical success does not identify a unique causal channel. Kinematic zero mean is not a braking/contact safety constraint; no solver/control change or causal intervention.',
        next='Close anchored benefit branch. Before any new training, test whether an observable coupled dynamic joint-design/actuator/contact feasibility contract can preserve the strong reference. Distinguish from existing CBF/QP and uncertainty-learning methods; no new method or safety theorem is claimed.210 direction review/cleanup.'))
    print('PASS ALL NEW DESIGN CROSSINGS',counts,flush=True)


if __name__=='__main__':run()
