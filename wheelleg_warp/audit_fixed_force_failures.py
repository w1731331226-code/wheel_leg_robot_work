"""Offline gate/path/request witnesses for the closed fixed-force study; no rollouts."""
import json
from collections import Counter
import numpy as np
from analyze_reward_failures import flags
from train_fixed_force_study import OUT
from score_reference_learning import load_rows
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
import wheelleg_sim as sim


def first(mask):
    rows=np.flatnonzero(mask)
    return int(rows[0]) if len(rows) else None


def lateral_gap(points,box,radius):
    position=np.asarray(box['position']);extent=abs(np.asarray(box['rotation']))@np.asarray(box['size'])
    near=abs(points[:,0]-position[0])<=extent[0]+radius
    return None if not near.any() else float((abs(points[near,1]-position[1])-extent[1]-radius).min())


def self_check():
    assert first(np.array([False,True,True]))==1 and first(np.zeros(3,bool)) is None
    assert first(abs(np.array([5.,-5.,5.001]))>5)==2
    box=dict(position=[1.,0.,0.],size=[.25,.035,.01],rotation=np.eye(3).tolist())
    assert lateral_gap(np.array([[1.,.2,0.]]),box,.05)>0
    assert lateral_gap(np.array([[1.,0.,0.]]),box,.05)<0
    assert lateral_gap(np.array([[0.,.2,0.]]),box,.05) is None
    print('PASS339 strictcross/missingevent/projected disjoint-overlap-noapproach;0physics',flush=True)


def run():
    self_check();destination=OUT/'failure_audit.json';assert not destination.exists()
    p=json.loads((OUT/'proposal.json').read_text());review=json.loads((OUT/'evaluation_review.json').read_text())
    assert review['verified'] and review['candidate_benefit_branch_closed'] and not review['candidate_qualification_passed']
    assert review['proposal_sha256']==sha(OUT/'proposal.json') and review['completion_sha256']==sha(OUT/'models/completion.json')
    baseline_review=json.loads((OUT/'baseline_raw_review.json').read_text());raw_inputs={};result_inputs={};data={};paths={};geometry={}
    assert review['baseline_raw_review_sha256']==sha(OUT/'baseline_raw_review.json') and baseline_review['completion_sha256']==sha(OUT/'baseline/completion.json')
    assert review['evaluation_source_sha256']==sha(OUT/'evaluation_source.json')
    source=json.loads((OUT/'evaluation_source.json').read_text());assert all(sha(ROOT/f)==h for f,h in source['source_sha256'].items())
    conditions=[f'{arm}_{seed}' for seed in p['seeds'] for arm in p['arms']]+[r['label'] for r in p['classical_conditions']]
    failure_counts={};total=0
    for kind in ('models','baseline'):
        c=json.loads((OUT/kind/'completion.json').read_text())
        for entry in c['records']:
            job=next(j for j in p['eval_jobs'] if j['panel']==entry['panel'] and j['batch']==entry['batch'])
            path=OUT/kind/entry['path'];assert sha(path)==entry['sha256'];result=load_rows(path,job['cases'])
            result_inputs[str(path.relative_to(ROOT))]=sha(path)
            gf=path.parent/'geometry.json';assert sha(gf)==entry['checked']['geometry_sha256'];geometry[str(path.parent)]=json.loads(gf.read_text())
            result_inputs[str(gf.relative_to(ROOT))]=sha(gf)
            for w,row in enumerate(result['runs']):
                key=(entry['condition'],entry['panel'],row['seed']);assert key not in data
                f=flags(row);assert row['success']==(not any(f.values()));data[key]=row;paths[key]=(path.parent,w)
                count=failure_counts.setdefault(entry['condition']+'/'+entry['panel'],Counter());count.update(k for k,v in f.items() if v)
                count['failures']+=not row['success'];count['total']+=1;total+=1
    assert total==1476
    pairs={}
    for seed in p['seeds']:
        label=f'D3_{seed}'
        for panel in ('regular','controlled','legacy'):
            cases=[case for job in p['eval_jobs'] if job['panel']==panel for case in job['cases']]
            for reference in [f'V6_{seed}']+[r['label'] for r in p['classical_conditions']]:
                lost=[];gained=[]
                for case in cases:
                    x=data[label,panel,case['seed']];y=data[reference,panel,case['seed']]
                    if y['success'] and not x['success']:lost.append(case['seed'])
                    if x['success'] and not y['success']:gained.append(case['seed'])
                assert len(gained)-len(lost)==sum(data[label,panel,c['seed']]['success']-data[reference,panel,c['seed']]['success'] for c in cases)
                pairs[f'{label}/{panel}--{reference}']=dict(lost=lost,gained=gained)
    selected=[('controlled',case['seed']) for job in p['eval_jobs'] if job['panel']=='controlled' for case in job['cases']]
    for panel in ('regular','legacy'):
        cases=[case for job in p['eval_jobs'] if job['panel']==panel for case in job['cases']]
        candidates=[]
        for case in cases:
            row=data['D3_33532',panel,case['seed']];failed={k for k,v in flags(row).items() if v}
            target=failed=={'terrain_contact'} if panel=='regular' else failed=={'attitude','yaw'} and max(row['peak_deg'][:2])<=5
            if target and data['V6_33532',panel,case['seed']]['success'] and data['guard_B0',panel,case['seed']]['success']:candidates.append(case['seed'])
        assert candidates;selected.append((panel,candidates[0]))
    assert len(selected)==42
    plot_cases=[selected[0],selected[-1]]
    witnesses=[];series={};steps=0;statistics={}
    for panel,case in selected:
        for label in conditions:
            key=(label,panel,case);row=data[key];directory,w=paths[key];geom=geometry[str(directory)]
            def read(field):
                f=directory/row[field]['path'];digest=sha(f);expected=review['raw_sha256'] if label.startswith(('D3_','V6_')) else baseline_review['raw_sha256']
                assert digest==row[field]['sha256']
                # Baseline receipt lists dense archives; frozen result SHA also binds the other stream hashes.
                if label.startswith(('D3_','V6_')) or str(f.relative_to(ROOT)) in expected:assert digest==expected[str(f.relative_to(ROOT))]
                raw_inputs[str(f.relative_to(ROOT))]=digest
                with np.load(f,allow_pickle=False) as z:return z['trace'].copy(),z['columns'].tolist() if 'columns' in z.files else None
            t,columns=read('complete_trace');g,_=read('joint_guard_trace');parking,_=read('parking_trace');actor,_=read('actor_trace')
            col={name:i for i,name in enumerate(columns)};n=len(t);assert n==row['physical_steps'] and g.shape==(n,62) and parking.shape==(n,30)
            assert actor.shape==(int(np.ceil(n/40)),968);steps+=n
            yaw=np.rad2deg(t[:,col['yaw']]);np.testing.assert_allclose(abs(yaw).max(),row['peak_deg'][2],rtol=0,atol=1e-12)
            speed=t[:,col['speed_command']];moving=speed!=0;started=np.maximum.accumulate(moving)
            stop=first(started & ~moving);cross=first(abs(yaw)>5)
            contacts=first((t[:,col['left_target_candidates']]+t[:,col['right_target_candidates']])>0)
            approach=None;gaps=[];radius=max(sim.hw.WHEEL_RADIUS,sim.hw.WHEEL_WIDTH/2)
            for box in geom['boxes'][w]:
                extent=abs(np.asarray(box['rotation']))@np.asarray(box['size']);position=np.asarray(box['position'])
                event=first(row['scenario']['speed']/abs(row['scenario']['speed'])*(t[:,col['body_x']]-position[0])>=-extent[0]-radius)
                if event is not None:approach=event if approach is None else min(approach,event)
                if box['geom'] in [i for i,name in enumerate(geom['geom_names']) if name in ('bump_L','bump_R')]:
                    mask=1 if geom['geom_names'][box['geom']]=='bump_L' else 2
                    if row['required_contact_mask'] & mask:
                        gap=[lateral_gap(t[:,[col[f'pre_{side}_{a}'] for a in 'xyz']],box,radius) for side in ('left','right')]
                        gaps.append(dict(geom=box['geom'],name=geom['geom_names'][box['geom']],left_right_separation_lower_bound_m=gap,
                            sampled_lateral_disjoint=all(v is not None and v>0 for v in gap)))
            def event(index):
                if index is None:return None
                k=index//40
                return dict(step=index+1,pre_s=float(t[index,col['pre_s']]),post_s=float(t[index,col['post_s']]),body_y_m=float(t[index,col['body_y']]),yaw_deg=float(yaw[index]),
                    route_estimate_m=float(actor[k,389]),packet_yaw_deg=float(np.rad2deg(actor[k,353])),policy_request6=actor[k,962:].tolist(),
                    nominal_motor6=g[index,2:8].tolist(),executed_residual_motor6=g[index,8:14].tolist(),issued_motor6=g[index,20:26].tolist(),
                    original_lambda=float(t[index,col['original_lambda']]),during_motion=bool(moving[index]))
            nominal=(g[:,6]-g[:,7])/2;residual=(g[:,12]-g[:,13])/2
            active=moving & (abs(nominal)>1e-6) & (abs(residual)>1e-6)
            window=moving if approach is None else moving & (np.arange(n)<approach)
            result=dict(condition=label,panel=panel,case=case,success=row['success'],failure_flags=[k for k,v in flags(row).items() if v],yaw_peak_deg=row['peak_deg'][2],
                body_y_peak_m=float(abs(t[:,col['body_y']]).max()),final_body_y_m=float(t[-1,col['body_y']]),approach=event(approach),first_target_contact=event(contacts),
                first_yaw5=event(cross),parking_handover=event(stop),required_bump_gaps=gaps,all_required_bumps_sampled_lateral_disjoint=bool(gaps) and all(x['sampled_lateral_disjoint'] for x in gaps),
                guard_nominal_corrected=int((g[:,60]>1e-12).sum()),guard_residual_reduced=int((g[:,27]<1-1e-9).sum()),
                original_lambda_limited_fraction=float(np.mean(t[moving,col['original_lambda']]<1-1e-9)),
                preapproach_wheel_residual_mean_Nm=float(np.mean(residual[window])) if window.any() else None,
                preapproach_wheel_residual_RMS_Nm=float(np.sqrt(np.mean(residual[window]**2))) if window.any() else None,
                preapproach_leg_residual_RMS_Nm=np.sqrt(np.mean(g[window,8:12]**2,axis=0)).tolist() if window.any() else None,
                opposite_nominal_wheel_residual_steps=int(((nominal*residual<0)&active).sum()),nonzero_nominal_and_residual_steps=int(active.sum()),
                source_result_sha256=result_inputs[str((directory/'result.json').relative_to(ROOT))])
            witnesses.append(result);st=statistics.setdefault(label,Counter());st['traces']+=1;st['yaw_cross_before_target_contact']+=cross is not None and (contacts is None or cross<contacts)
            st['yaw_cross_during_motion']+=cross is not None and bool(moving[cross]);st['no_target_contact']+=contacts is None
            st['required_bumps_sampled_disjoint']+=result['all_required_bumps_sampled_lateral_disjoint']
            if (panel,case) in plot_cases and label in ('D3_33532','V6_33532','guard_B0','guard_B1_route','guard_B1_withdraw'):
                series[key]=dict(time=t[:,col['post_s']],yaw=yaw,y=t[:,col['body_y']],x=t[:,col['body_x']],left=t[:,[col['post_left_x'],col['post_left_y']]],
                                 nominal=nominal,residual=residual,boxes=geom['boxes'][w])
        print('AUDITED339',panel,case,'matched9',flush=True)
    assert len(witnesses)==378
    atomic_json(destination,dict(round=339,verified=True,original_gate_rows=1476,unique_development_cases=164,overlapping_failure_counts={k:dict(v) for k,v in failure_counts.items()},pairs=pairs,
        selection_rule='All40controlled plus first registered D3_33532 regular terrain-contact-only loss and first legacy yaw-only loss,each also V6_33532/guardB0 success. Everyselectedcase readsall9conditions;not a new performance test.',
        selected_cases=[dict(panel=a,case=b) for a,b in selected],dense_trajectories=378,dense_physical_steps_read=steps,witnesses=witnesses,statistics={k:dict(v) for k,v in statistics.items()},
        evaluation_review_sha256=sha(OUT/'evaluation_review.json'),baseline_raw_review_sha256=sha(OUT/'baseline_raw_review.json'),proposal_sha256=sha(OUT/'proposal.json'),auditor_sha256=sha(__file__),
        result_sha256=result_inputs,raw_sha256=raw_inputs,new_physics_steps=0,new_learning_samples=0,formal5_admitted=False,candidate_benefit_branch_closed=True,
        source_observation='Dense reward includes speed/attitude/executed-residual/smooth and terminal fulltask ±10. No dense route-y or terrain-preview term. Route-y isavailable in public481input,not proofpolicyusesit.',
        limits='Descriptive saved-trajectory timing/action associations,not causalchannel intervention orinsufficienttraining proof. Lateraldisjoint uses tirebounding sphere and geomAABB at recorded preposes;negative gap doesnot prove contact. Public packetroute/yaw isdelayed;postpose/firstcontact clocks differby0.5ms. No counterfactual policy ornewcase selection fortraining.',
        next='340 five-round direction/cleanup plus ten-round whole-framework audit:decide whether a finite frozen-channel mechanism test isworthwhile;closedcandidate never rescued by tuning/epochs.'))
    plot(series,plot_cases);print('DONE3391476gate rows/378dense matched witnesses;0physics/learning',flush=True)


def plot(series,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,(panel,case) in enumerate(cases):
        for (label,p,c),s in series.items():
            if (p,c)!=(panel,case):continue
            axes[row,0].plot(s['x'],s['y'],label=label,linewidth=1)
            axes[row,1].plot(s['time'],s['yaw'],linewidth=1)
            axes[row,2].plot(s['time'],s['residual'],linewidth=1)
        axes[row,0].set(xlabel='Body x (m)',ylabel='Body y (m)',title=f'{panel}: {case}')
        axes[row,1].axhline(5,color='black',linestyle='--',linewidth=.7);axes[row,1].axhline(-5,color='black',linestyle='--',linewidth=.7)
        axes[row,1].set(xlabel='Time (s)',ylabel='Yaw (deg)');axes[row,2].set(xlabel='Time (s)',ylabel='Executed wheel residual half-difference (Nm)')
    axes[0,0].legend(fontsize=8);fig.suptitle('Two deterministically selected seen cases: saved trajectories, no causal intervention')
    fig.tight_layout();fig.savefig(OUT/'failure_witnesses.png',dpi=160);fig.savefig(OUT/'failure_witnesses.svg');plt.close(fig)


if __name__=='__main__':
    import sys
    self_check() if '--self-check' in sys.argv else run()
