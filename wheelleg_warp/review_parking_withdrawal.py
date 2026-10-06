"""Complete78 paired prefix/mechanism review; no new simulation or learning."""
import json
import subprocess
import numpy as np
import run_parking_withdrawal as runner
from score_reference_learning import load_rows
from review_floor_broad_qualification import ordered,wrap
from review_nominal_mean_study import paired
from audit_nominal_mean_failures import crossing,NAMES
from native.terrain import HeightTerrainScenario,model


def prefix(a,b,n):
    if len(a)<n or len(b)<n:return dict(exact=False,missing_prefix=True,max_abs_difference=None)
    different=np.flatnonzero(np.any((a[:n]!=b[:n]).reshape(n,-1),axis=1)) if n else []
    return dict(exact=bool(np.array_equal(a[:n],b[:n])),missing_prefix=False,first_difference_step=int(different[0])+1 if len(different) else None,
                max_abs_difference=float(np.max(np.abs(a[:n]-b[:n]))) if n else 0.)


def unit():
    x=np.array([[1.,2.],[3.,4.]]);y=x.copy()
    assert prefix(x,y,2)['exact']
    y[1,0]+=1;assert prefix(x,y,1)['exact'] and not prefix(x,y,2)['exact']
    assert not prefix(x,y[:1],2)['exact']


def run():
    unit();p=runner.verify();out=runner.OUT;root=runner.ROOT
    cfile=out/'completion.json'
    if not cfile.exists():raise RuntimeError('Incomplete78: full causal review refused')
    c=json.loads(cfile.read_text());assert c['verified'] and c['evaluations']==78 and c['training_updates']==0 and c['runner_contract_sha256']==runner.sha(out/'runner_contract.json')
    assert [(r['seed'],r['condition'],r['batch']) for r in c['records']]==[(j['seed'],j['condition'],j['batch']) for j in runner.jobs(p)]
    state=subprocess.check_output(['systemctl','--user','show','wheelleg-parking-withdrawal-v1.service','-p','MainPID','-p','SubState','-p','Result','-p','ExecMainStatus'],text=True)
    assert 'MainPID=0' in state and 'SubState=exited' in state and 'Result=success' in state and 'ExecMainStatus=0' in state
    inputs={str(cfile.relative_to(root)):runner.sha(cfile)};datasets={};paths={};traces=total_bytes=0
    for m in p['models']:
        for condition in p['conditions']:
            rows=[];indices=[]
            for r in c['records']:
                if (r['seed'],r['condition'])!=(m['seed'],condition):continue
                f=out/r['path'];assert runner.sha(f)==r['sha256'];inputs[str(f.relative_to(root))]=r['sha256']
                d=load_rows(f,[p['cases'][i] for i in r['indices']]);rows+=d['runs'];indices+=r['indices']
                assert r['verified'] and r['model_RMS_immutable']
                assert runner.parking_logs(f.parent,condition)['parking_trace_files']==len(d['runs'])
                for row in d['runs']:
                    paths[(m['seed'],condition,row['seed'])]=f.parent
                    for kind in ('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace'):
                        file=f.parent/row[kind]['path'];assert runner.sha(file)==row[kind]['sha256'];inputs[str(file.relative_to(root))]=row[kind]['sha256'];traces+=1;total_bytes+=file.stat().st_size
            datasets[(m['seed'],condition)]=wrap(ordered(rows,indices,13))
    cpu=model(HeightTerrainScenario(**p['cases'][0]['scenario']));qi=[int(cpu.joint(n).qposadr[0]) for n in NAMES];vi=[cpu.nq+int(cpu.joint(n).dofadr[0]) for n in NAMES]
    pairs={};case_pairs=[]
    for m in p['models']:
        seed=m['seed'];before=datasets[(seed,'original')];after=datasets[(seed,'parking_request_withdrawal')]
        pairs[str(seed)]=paired(before,after)
        for a,b in zip(before['runs'],after['runs']):
            def load(row,condition,kind):
                file=paths[(seed,condition,row['seed'])]/row[kind]['path']
                with np.load(file,allow_pickle=False) as z:return {k:z[k] for k in z.files}
            pa=load(a,'original','parking_trace')['trace'];pb=load(b,'parking_request_withdrawal','parking_trace')['trace']
            hit=np.flatnonzero(pb[:,4]==1);g=int(hit[0]) if len(hit) else min(len(pa),len(pb))
            checks={};a_full=load(a,'original','complete_trace');b_full=load(b,'parking_request_withdrawal','complete_trace')
            assert a_full['state_sizes'].tolist()==b_full['state_sizes'].tolist()==[cpu.nq,cpu.nv,cpu.nu]
            for field in ('trace','pre','post'):checks['complete/'+field]=prefix(a_full[field],b_full[field],g)
            for kind in ('gyro_trace','role_trace','phase_trace'):
                checks[kind]=prefix(load(a,'original',kind)['trace'],load(b,'parking_request_withdrawal',kind)['trace'],g)
            checks['parking_request_and_memory']=prefix(pa,pb,g)
            exact=bool(len(hit) and all(x['exact'] for x in checks.values()))
            original=crossing(a_full['pre'][:,qi],a_full['post'][:,qi],a_full['pre'][:,vi]);withdrawn=crossing(b_full['pre'][:,qi],b_full['post'][:,qi],b_full['pre'][:,vi])
            assert (original['first'] is None)==a['design_joint_passed'] and (withdrawn['first'] is None)==b['design_joint_passed']
            assert original['worst_margin_rad']==a['min_active_design_margin_rad'] and withdrawn['worst_margin_rad']==b['min_active_design_margin_rad']
            case_pairs.append(dict(seed=seed,case=a['seed'],height_m=a['target_leg_m'],trigger_step=g+1 if len(hit) else None,
                recorded_prefix_exact=exact,prefix_checks=checks,original_crossing=original,withdrawal_crossing=withdrawn,
                original_flags=pairs[str(seed)]['rows'][p['cases'].index(next(x for x in p['cases'] if x['seed']==a['seed']))]['original_flags'],
                original_success=a['success'],withdrawal_success=b['success'],original_physical=a['physical_safety_passed'],withdrawal_physical=b['physical_safety_passed'],
                design_violation_removed=bool(not a['design_joint_passed'] and b['design_joint_passed']),new_design_violation=bool(a['design_joint_passed'] and not b['design_joint_passed'])))
    assert len(case_pairs)==39 and traces==390
    summary=dict(paired_records=39,exact_recorded_pretrigger_prefix=sum(r['recorded_prefix_exact'] for r in case_pairs),
        original_design_violations=sum(r['original_crossing']['first'] is not None for r in case_pairs),
        withdrawal_design_violations=sum(r['withdrawal_crossing']['first'] is not None for r in case_pairs),
        removed_design_violations=sum(r['design_violation_removed'] for r in case_pairs),new_design_violations=sum(r['new_design_violation'] for r in case_pairs),
        original_success=[datasets[(m['seed'],'original')]['summary']['success_count'] for m in p['models']],
        withdrawal_success=[datasets[(m['seed'],'parking_request_withdrawal')]['summary']['success_count'] for m in p['models']])
    qualified=[r for r in case_pairs if r['recorded_prefix_exact']]
    summary['causally_qualified_subset']=dict(pairs=len(qualified),design_violations_removed=sum(r['design_violation_removed'] for r in qualified),
        original_success=sum(r['original_success'] for r in qualified),withdrawal_success=sum(r['withdrawal_success'] for r in qualified),
        original_success_lost=sum(r['original_success'] and not r['withdrawal_success'] for r in qualified),
        original_physical=sum(r['original_physical'] for r in qualified),withdrawal_physical=sum(r['withdrawal_physical'] for r in qualified))
    summary['unqualified_prefix_pairs']=39-len(qualified)
    runner.write(out/'round213_pair_review.json',dict(verified=True,round=213,summary=summary,pairs=pairs,case_pairs=case_pairs,
        original_replay_comparisons={str(r['seed']):r['original_replay_comparison'] for r in c['records'] if r['condition']=='original'},
        trajectory_files=traces,trajectory_bytes=total_bytes,first_episode_physics_steps=c['first_episode_physics_steps'],evaluation_queue_wall_s=c['evaluation_queue_wall_s'],terminal_unit=state,
        input_sha256=inputs,source_sha256=runner.sha(__file__),runner_contract_sha256=runner.sha(out/'runner_contract.json'),new_training_or_evaluations=0,
        candidate_benefit_reopened=False,formal5_admitted=False,entire_goal_complete=False,
        limits='Finite39 new pairs on13 seen development cases/3 frozenmodels. Exact recorded prefix is required percase;solver warmstart is not recorded. No cross-study bitwise equivalence,global safety,independent generalization,novelty or training advantage. All old/replay and prefix mismatches retained,not tolerance-tuned or dropped.',
        next='214 use exact qualified case evidence to decide dynamic handover versus active-motion/design feasibility method;215 direction review/cleanup. Full original manuscript/formal5/freshID-OOD/strongcomparisons remain open.'))
    print('PASS213 full78 delivery/pairs',summary,'raw',traces,total_bytes,flush=True)


if __name__=='__main__':run()
