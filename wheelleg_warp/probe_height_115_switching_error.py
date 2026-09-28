"""冻结旧G/误差界，前瞻采集新动作序列；不调参、不作闭环安全声明。"""
import json
from pathlib import Path
import numpy as np
from probe_height_115_action_predict_loow import ROOT, DT, sha, compare_start
from probe_height_115_braking_budget import ACTIVE, barriers
from probe_height_115_live_common_local import simulate, common_basis
from probe_height_115_radial_authority import torque_box
from native.terrain import HeightTerrainScenario, model


def run():
    folder = ROOT/'wheelleg_warp/results'
    output = folder/'height_115_switching_error_20260929'
    assert not output.exists()
    inputs = [folder/n/'verification.json' for n in (
        'height_115_action_predict_1nm_single_graph_20260929',
        'height_115_acceleration_error_20260929','height_115_local_states_20260928')]
    fit, bound, archive = [json.loads(p.read_text()) for p in inputs]
    windows_path = inputs[2].parent/'windows.npz'
    assert sha(windows_path)==archive['windows_sha256']
    windows=np.load(windows_path)
    # Fixed before results: ± sequences, command delays 0/.5/2ms, 12ms before archived failure.
    directions=np.array([[0,0,1],[1,-1,1],[0,0,-1],[-1,1,-1],
                         [1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[1,1,-1],[-1,-1,1]],float)
    directions/=np.abs(directions).sum(axis=1,keepdims=True)
    output.mkdir();results=[]
    for world in range(6):
        event_id=fit['selected_event_ids'][world];event=archive['events'][event_id]
        scenario=HeightTerrainScenario(**event['scenario']);m=model(scenario)
        qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE])
        va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
        start=round(event['reference_s']/DT)-24
        old=windows[f'event_{event_id}_pre'];row=old[np.argmin(abs(old[:,0]-start*DT))]
        assert abs(row[0]-start*DT)<2e-6
        cols=archive['columns'];qold=row[cols['qpos_start']:cols['qpos_start']+m.nq]
        scale=.8/np.max(abs(common_basis(m,qold,np.zeros(m.nv))),axis=0)
        schedule=np.zeros((40,7,3));metadata=[dict(sign=0,delay_steps=0)]
        for arm,(sign,delay) in enumerate([(s,d) for d in (0,1,4) for s in (1,-1)],1):
            metadata.append(dict(sign=sign,delay_steps=delay))
            for t in range(delay,40):schedule[t,arm]=sign*directions[(t-delay)//4]*scale
        records,starts,stats,_=simulate([scenario],[start],np.zeros((7,3)),schedule=schedule)
        default_check=None
        if world==0:
            legacy,_,_,_=simulate([scenario],[start],np.zeros((7,3)))
            qerr=float(abs(legacy[0]['post']-records[0]['post']).max())
            verr=float(abs(legacy[0]['pre'][:,m.nq:m.nq+m.nv]-records[0]['pre'][:,m.nq:m.nq+m.nv]).max())
            assert qerr<=2e-6 and verr<=1e-6,(qerr,verr)
            default_check=dict(zero_arm_qpos_max_error=qerr,zero_arm_qvel_max_error=verr)
        gain=np.array(fit['folds'][world]['gain']);b=bound['folds'][world]
        upper=np.array(b['positive_error_bound_rad_s2']);lower=np.array(b['negative_error_bound_rad_s2'])
        arrays={k:np.stack([r[k] for r in records]) for k in ('pre','post','post_velocity','applied')}
        arrays['schedule']=schedule;arrays['starts']=starts
        arrays['previous_velocity']=np.stack([r['previous_qvel'] for r in records])
        arrays['contact_raw']=records[0]['contact_raw']
        np.savez_compressed(output/f'world_{world}.npz',**arrays)
        peak=0.;match=0.;box=0.;age_rows=[]
        for arm,r in enumerate(records):
            compare_start(starts[arm],starts[0],m.nq,m.nv,f'world{world} arm{arm}')
            assert np.all(r['pre'][:,-1]==1)
            assert np.array_equal(r['pre'][1:,m.nq:m.nq+m.nv],r['post_velocity'][:-1])
            for t in range(40):
                q=r['pre'][t,:m.nq];v=r['pre'][t,m.nq:m.nq+m.nv]
                nominal=r['pre'][t,m.nq+m.nv:m.nq+m.nv+6]
                delta=common_basis(m,q,v)@schedule[t,arm]
                peak=max(peak,float(abs(delta).max()))
                match=max(match,float(abs(r['applied'][t]-nominal-delta).max()))
                box=max(box,float((abs(r['applied'][t])-torque_box(m,v)).max()))
            for age in (0,1,4):
                errors=[];radial=[];switched=[]
                for t in range(age,40):
                    k=t-age;q=r['pre'][k,qa];v=r['pre'][k,m.nq+va]
                    prev=r['previous_qvel'][va] if k==0 else r['pre'][k-1,m.nq+va]
                    prior=np.zeros(3) if k==0 else schedule[k-1,arm]
                    predicted=(v-prev)/DT+gain@(schedule[t,arm]-prior)
                    measured=(r['post_velocity'][t,va]-r['pre'][t,m.nq+va])/DT
                    errors.append(predicted-measured)
                    bs,_=barriers(q,v,predicted,gain,(0.,0.))
                    now,_=barriers(r['pre'][t,qa],r['pre'][t,m.nq+va],np.zeros(4),gain,(0.,0.))
                    nxt,_=barriers(r['post'][t,qa],r['post_velocity'][t,va],np.zeros(4),gain,(0.,0.))
                    radial.append([bs[j]['a0']-(nxt[j]['rate']-now[j]['rate'])/DT for j in range(2)])
                    switched.append(not np.array_equal(schedule[t,arm],np.zeros(3) if t==0 else schedule[t-1,arm]))
                e=np.array(errors);excess=np.maximum(e-upper,-e-lower);miss=np.any(excess>1e-9,axis=1)
                switched=np.array(switched)
                age_rows.append(dict(arm=arm,observation_age_ms=age*.5,steps=len(e),
                    uncovered_steps=int(miss.sum()),switch_steps=int(switched.sum()),
                    uncovered_switch_steps=int((miss&switched).sum()),
                    max_bound_excess_rad_s2=float(max(0,excess.max())),
                    max_abs_joint_error_rad_s2=float(abs(e).max()),
                    max_optimistic_radial_error_m_s2=float(np.max(radial)),
                    contact_changed_vs_zero=sum(a!=z for a,z in zip(r['contacts'],records[0]['contacts']))))
        assert peak<=1.+1e-6 and match<=1e-5 and box<=1e-6,(peak,match,box)
        results.append(dict(world=world,start_step=start,arms=metadata,rows=age_rows,
            default_path_check=default_check,
            peak_extra_motor_Nm=peak,motor_match_max_error_Nm=match,motor_box_excess_Nm=box,
            kernel_clipping_events=float(stats[:,1].sum()),trace_sha256=sha(output/f'world_{world}.npz')))
        print('world',world,'age0 misses',sum(r['uncovered_steps'] for r in age_rows if r['observation_age_ms']==0),flush=True)
    report=dict(role='prospective_switching_sequences_frozen_model_validation',
        new_scenarios=False,new_action_sequences=True,training=False,final_holdout_opened=False,
        control_safety_claim=False,observation_age_is_retrospective=True,
        command_delays_ms=[0,.5,2],observation_ages_ms=[0,.5,2],
        protocol='Six public worlds; start failure-12ms; 40 steps; zero plus ± fixed sequence with three command delays. Frozen prior LOOW gains and signed bounds; no refitting.',
        limitations='Delayed commands are open-loop probes, not delayed closed-loop braking. Radial errors are measured but lack an independently calibrated bound. Observation-age audit uses old measurements and known issued action, without state extrapolation.',
        worlds=results,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs+[windows_path]},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),
            ROOT/'wheelleg_warp/probe_height_115_live_common_local.py',ROOT/'wheelleg_warp/probe_height_115_braking_budget.py',
            ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
    (output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    run()
