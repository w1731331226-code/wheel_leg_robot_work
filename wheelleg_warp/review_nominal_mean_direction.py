"""Round210 direction review and an offline public-command parking contract."""
import json
import numpy as np
import nominal_mean_main as main


def parking_mask(commands):
    commands=np.asarray(commands,dtype=float)
    if commands.ndim!=1 or not np.isfinite(commands).all():
        raise ValueError('Expected finite current-command history')
    return np.maximum.accumulate(commands!=0) & (commands==0)


def unit():
    x=[0,0,.2,.5,0,0,-.1,0]
    expected=[False,False,False,False,True,True,False,True]
    assert parking_mask(x).tolist()==expected
    assert parking_mask(-np.array(x)).tolist()==expected
    assert not parking_mask([0,0,0]).any()
    try:parking_mask([0,float('nan')])
    except ValueError:pass
    else:raise AssertionError('Invalid command accepted')


def run():
    unit();p=main.verify();out=main.OUT;root=main.ROOT;inputs={}
    def read(name):
        f=out/name;inputs[str(f.relative_to(root))]=main.sha(f);return json.loads(f.read_text())
    study=read('study_pair_review.json');delivery=read('completed_collection_audit.json');failure=read('round209_failure_mechanism.json');clean=read('round210_cleanup.json')
    assert study['verified'] and delivery['verified'] and failure['verified'] and clean['verified']
    assert not study['formal_expansion_gate_passed']
    assert delivery['study_review_sha256']==main.sha(out/'study_pair_review.json')
    assert (delivery['validated_training_policy_steps'],delivery['validated_final_evaluations'],delivery['validated_trajectory_files'])==(1200000,1254,5286)
    assert main.sha(root/'wheelleg_warp/audit_nominal_mean_failures.py')==failure['source_sha256']
    for evidence in (delivery['input_sha256'],failure['input_sha256'],failure['dependency_sha256']):
        assert all(main.sha(root/n)==s for n,s in evidence.items())
    for seed in p['seeds']:
        a=delivery['records'][f'anchored/{seed}'];b=delivery['records'][f'plain/{seed}']
        assert a['initial_weight_sha256']==b['initial_weight_sha256'] and a['initial_world_sha256']==b['initial_world_sha256']
        for arm in p['arms']:
            d=out/'runs'/arm/str(seed);r=delivery['records'][f'{arm}/{seed}']
            assert main.sha(d/'verification.json')==r['run_verification_sha256']
            t=json.loads((d/'training_verification.json').read_text())
            assert t['verified'] and (t['policy_steps'],t['epochs'],t['Adam'])==(200000,400,8000)
    hypothetical={}
    for panel in ('regular','controlled'):
        rows=[]
        for seed in p['seeds']:
            pair=study['pairs'][f'anchored/{seed}/{panel}/vs_B0']
            only=[r['case'] for r in pair['rows'] if r['candidate_flags']['design'] and sum(r['candidate_flags'].values())==1]
            rows.append(dict(seed=seed,observed_success=pair['candidate_success'],design_only_case_ids=only,
                             hypothetical_success_if_only_design_flags_removed=pair['candidate_success']+len(only)))
        hypothetical[panel]=rows
    trigger=[]
    for r in failure['records']:
        suffix=f'/case_phase_{r["case"]}.npz';prefix=f'{str(out.relative_to(root))}/runs/anchored/{r["seed"]}/evaluation/{r["panel"]}/'
        names=[n for n in failure['input_sha256'] if n.startswith(prefix) and n.endswith(suffix)]
        assert len(names)==1;f=root/names[0];inputs[names[0]]=main.sha(f)
        with np.load(f,allow_pickle=False) as z:
            t=z['trace'];cols={str(k):i for i,k in enumerate(z['columns'])}
        cmd=t[:,cols['current_command']];mask=parking_mask(cmd);first_motion=int(np.flatnonzero(cmd!=0)[0]);first_gate=int(np.flatnonzero(mask)[0])
        assert not mask[:first_motion].any() and mask[r['candidate']['first']]
        np.testing.assert_array_equal(t[:,cols['zero_command']],cmd==0)
        trigger.append(dict(seed=r['seed'],case=r['case'],first_motion_step=first_motion+1,first_parking_gate_step=first_gate+1,
                            first_design_crossing_step=r['candidate']['first']+1,startup_untouched=True))
    cases={r['case'] for r in failure['records']}
    selected=[c for panel in ('regular','controlled') for c in p[panel] if c['seed'] in cases]
    assert len(selected)==13 and len(trigger)==26
    models=[]
    for seed in p['seeds']:
        d=out/'runs/anchored'/str(seed)
        models.append(dict(seed=seed,model=str((d/'step_200000.zip').relative_to(root)),model_sha256=main.sha(d/'step_200000.zip'),
                           normalization=str((d/'step_200000.pkl').relative_to(root)),normalization_sha256=main.sha(d/'step_200000.pkl')))
    main.write(out/'round210_direction_review.json',dict(verified=True,round=210,previous_turn='progress',input_sha256=inputs,
        source_sha256=main.sha(__file__),trainer_contract_sha256=main.sha(out/'trainer_contract.json'),
        finite_original_gate_pass_count=sum(study['gates'].values()),finite_original_gate_count=len(study['gates']),
        decision='Keep candidate-benefit closed; one bounded causal parking-withdrawal probe is worthwhile before designing dynamic residual constraints. No new learning or formal5.',
        hypothetical_design_only_repair=hypothetical,hypothetical_controlled_mean= float(np.mean([r['hypothetical_success_if_only_design_flags_removed'] for r in hypothetical['controlled']])),
        hypothetical_limits='Offline flag accounting with every other metric held fixed; not achieved scores,an intervention prediction,or an upper bound on actual coupled dynamics. Yaw score depends on measured yaw/duration,not design flag.',
        startup_confound_resolved=True,command_history_trigger_checks=trigger,
        diagnostic=dict(status='registered_for_source_qualification_only',cases=selected,models=models,conditions=['original','parking_request_withdrawal'],
            total_first_episode_evaluation_budget=78,training_updates=0,trigger='Per-world seen_nonzero_current_command latch AND exact current_command==0; reset clears latch. Never read goal/arrival/contact truth to decide withdrawal.',
            change='Gate new six-dimensional residual requests before the existing temporal filter; preserve its owner/order/delay,actual phase anchor,Nominal controller and39 actor inputs.',
            admission='Reset/terminal lifecycle, startup/moving noop, switching beforefilter/currentcommand owner,all39 prefixes/modelRMS immutable/source bindings,original replay comparability first; no execution until qualified/frozen.',
            interpretation='All3 models on the common13 already-seen development cases,not26 independent cases or fresh generalization. Both conditions/all failures kept. Success informs handover only,not revival of the failed benefit gate,novelty,learning necessity or formal5;failure closes fixed withdrawal without duration/gain/seed rescue.'),
        cleanup=clean,new_training_or_evaluations=0,entire_goal_complete=False,
        next='211 source qualification of the single registered diagnostic;212 freeze/run only if admitted;213 paired causal review;214 decide a distinguishable active-motion/dynamic-feasibility method;215 direction review/cleanup. Full strong comparisons/formal5/freshID-OOD/statistics/reproducible manuscript remain open.'))
    print('PASS210 deepreview; original gates',sum(study['gates'].values()),'/',len(study['gates']),
          'hypothetical design-only controlled',[r['hypothetical_success_if_only_design_flags_removed'] for r in hypothetical['controlled']],
          'startup-safe trigger checks',len(trigger),'budget78 registered,0 consumed',flush=True)


if __name__=='__main__':run()
