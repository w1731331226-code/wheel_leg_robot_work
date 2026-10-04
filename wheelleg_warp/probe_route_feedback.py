"""One registered known-origin route-reference classical control experiment.

Existing PD and kinematic lookahead are not claimed as novel control theory.
Keep the old task gates, actuator limits and 0.3Nm request authority.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,json,math
import numpy as np
from route_state import RouteState
from train_height_comparison import raw_env,summary,b1_action
from native.terrain import HeightTerrainScenario
from training_contract import digest
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'wheelleg_warp/results/paper_recovery_20261004'
COL=['end_s','packet_yaw_rad','packet_gyro_z_rad_s','command_m_s','estimated_y_m',
     'reference_yaw_rad','wheel_request_normalized','end_true_y_m','terminal']


def actions(obs,candidate,arm):
    obs=np.asarray(obs,dtype=np.float64)
    if obs.ndim!=2 or obs.shape[1]!=39 or not np.isfinite(obs).all() or arm not in (0,1,-1):
        raise ValueError('Expected finite39D route packet and registered arm')
    kp=.4+candidate['kp'];kd=2.+candidate['kd'];tau=kd/kp
    moving=abs(obs[:,9])>.05
    ref=np.zeros(len(obs))
    ref[moving]=-arm*np.sign(obs[moving,9])*np.arctan2(obs[moving,38],abs(obs[moving,9])*tau)
    ref=np.clip(ref,-math.radians(3),math.radians(3))
    schedule=1+candidate['roll_gain']*np.minimum(abs(obs[:,0])/math.radians(2),1)
    torque=-schedule*(candidate['kp']*obs[:,2]+candidate['kd']*obs[:,5])+kp*ref
    result=np.zeros((len(obs),3),np.float32);result[:,2]=np.clip(torque/.3,-1,1)
    assert np.isfinite(result).all() and np.max(abs(result))<=1
    return result,ref


def self_check(candidate):
    obs=np.random.default_rng(117).normal(size=(64,39));obs[:,38]=0
    result,ref=actions(obs,candidate,1)
    np.testing.assert_array_equal(result,np.stack([b1_action(o,candidate) for o in obs]))
    assert not ref.any()
    obs=np.zeros((4,39));obs[:,9]=[.7,-.7,.7,0];obs[:,38]=[.01,.01,-.01,.01]
    result,ref=actions(obs,candidate,1)
    assert ref[0]<0 and ref[1]>0 and ref[2]>0 and ref[3]==0
    mirrored=obs.copy();mirrored[:,[0,2,5,38]]*=-1
    np.testing.assert_array_equal(actions(mirrored,candidate,1)[0][:,2],-result[:,2])
    np.testing.assert_array_equal(actions(obs,candidate,-1)[1],-ref)
    obs[:,38]=1e6;assert np.max(abs(actions(obs,candidate,1)[1]))<=math.radians(3)


def collect(cases,candidate,arm,out,label):
    raw=raw_env(cases,'diff3');env=RouteState(raw,'route')
    rows=[None]*len(cases);pending=np.ones(len(cases),bool);traces=[[] for _ in cases]
    try:
        obs=env.reset();ids=raw.ids.numpy()
        deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
        for _ in range(deadline):
            before=obs.copy();action,ref=actions(before,candidate,arm)
            obs,_,done,infos=env.step(action)
            q=raw.data.qpos.numpy();state=raw.state.numpy();stopped=raw.stopped_q.numpy() if done.any() else None
            for w in np.flatnonzero(pending):
                end=stopped[w] if done[w] else q[w]
                time=infos[w]['duration_s'] if done[w] else float(state[w,0]*.0005)
                traces[w].append([time,*before[w,[2,5,9,38]].tolist(),ref[w],float(action[w,2]),float(end[1]),float(done[w])])
                if done[w]:
                    length=float(np.mean([sim.fk_joints(float(end[ids[2*s]]),float(end[ids[2*s+1]]))['leg_len'] for s in range(2)]))
                    rows[w]=dict(**cases[w],**{k:v for k,v in infos[w].items() if k!='terminal_observation'},final_mean_fk_leg_m=length)
                    pending[w]=False
            if not pending.any():break
        assert not pending.any()
        arrays=[np.asarray(t) for t in traces]
        np.savez_compressed(out/(label+'_trace.npz'),columns=np.array(COL),offsets=np.cumsum([0]+[len(t) for t in arrays]),trace=np.concatenate(arrays))
        result=dict(summary=summary(rows),physical=sum(r['physical_safety_passed'] for r in rows),design=sum(r['design_joint_passed'] for r in rows),runs=rows,
            mean_case_max_abs_y_m=float(np.mean([abs(t[:,7]).max() for t in arrays])),max_reference_rad=float(max(abs(t[:,5]).max() for t in arrays)))
        atomic_json(out/(label+'.json'),result)
        return {k:v for k,v in result.items() if k!='runs'}
    finally:env.close()


def run(out):
    prior=json.loads((S/'yaw_sector_v1/registration.json').read_text());candidate=prior['classical'];self_check(candidate)
    engine=json.loads((S/'route_interface_v1/verification.json').read_text());assert engine['verified']
    assert all(digest(ROOT/n)==v for n,v in engine['source_sha256'].items())
    controlled=[]
    for height in [.115,.16,.24,.30,.38]:
        for direction in [1,-1]:
            for side in ['left','right']:
                for obstacle in [.01,.02]:
                    seed=4600000+len(controlled)
                    scene=HeightTerrainScenario(speed=direction*.7,mass=7.,height_l=obstacle if side=='left' else 0.,height_r=obstacle if side=='right' else 0.,
                        center=1.8,offset=0.,mu_l=.8,mu_r=.8,drive_difference=0.,delay_ms=0.,terrain='legacy',terrain_seed=seed,relative_attitude=False,stand_height_m=height)
                    controlled.append(dict(seed=seed,scenario=asdict(scene)))
    sets=dict(regular=prior['cases'],controlled=controlled)
    out.mkdir(parents=True,exist_ok=False)
    reg=dict(version='route-feedback-development-v1',sets=sets,candidate=candidate,arms={'B1':0,'correct':1,'reverse':-1},columns=COL,
        formula='psi_ref=clip(-arm*sign(command)*atan2(estimated_y,abs(command)*tau),+-3deg) if abs(command)>.05 else0; B1rawPD+(0.4+B1kp)*psi_ref, then original+-0.3Nm request clip',
        tau_s=(2.+candidate['kd'])/(.4+candidate['kp']),
        derivation='Small-angle y_dot~command*yaw, overdamped inner yaw kd*yaw_dot+kp*yaw=kp*reference has nominal time constant tau=kd/kp; lookahead abs(command)*tau.3deg reference leaves2deg of original5deg yaw budget; not a dynamic safety margin guarantee.',
        inputs='Same39D packet and causal estimated y for all arms; no hidden geometry/parameters/current position. True y recorded only after action for analysis.',
        budget_episodes=216,training_steps=0,development_only=True,source_sha256={**engine['source_sha256'],'wheelleg_warp/probe_route_feedback.py':digest(__file__)},
        primary='Complete task success, preserve each B1 success, physical/design nondegradation separately per panel; include all40 controlled cases and regular32, no gain/case selection.',
        continue_gate='Correct preserves each B1 task success, completed trajectory and physical/design pass in each panel; more controlled successes; lower controlled mean max|y| than both B1 and reverse; regular Jpsi<=B1+.05deg. Failure stops this fixed law, not all route information research.',
        limitations='Kinematic/overdamped approximation ignores contact/actuator coupling and saturation. Existing PD/path reference is not a new algorithm; no contact/global safety or PPO superiority claim.',
        old_gate_or_final_used=False,no_retry_or_gain_sweep=True)
    atomic_json(out/'registration.json',reg);results={};records=[];completed=0
    try:
        for panel,cases in sets.items():
            for label,arm in reg['arms'].items():
                job=panel+'_'+label;atomic_json(out/'progress.json',dict(status='running',completed_episodes=completed,pending_job=job))
                assert all(digest(ROOT/n)==v for n,v in reg['source_sha256'].items())
                results[job]=collect(cases,candidate,arm,out,job)
                records.append(dict(job=job,episodes=len(cases),result_sha256=digest(out/(job+'.json')),trace_sha256=digest(out/(job+'_trace.npz'))))
                completed+=len(cases);atomic_json(out/'completed_jobs.json',dict(completed_episodes=completed,records=records))
                print('COMPLETED',completed,job,results[job],flush=True)
        preservation={}
        for panel in sets:
            b=json.loads((out/(panel+'_B1.json')).read_text());c=json.loads((out/(panel+'_correct.json')).read_text())
            preservation[panel]=all((not rb['success'] or rc['success'])
                and (rb['reason']!='completed' or rc['reason']=='completed')
                and (not rb['physical_safety_passed'] or rc['physical_safety_passed'])
                and (not rb['design_joint_passed'] or rc['design_joint_passed'])
                for rb,rc in zip(b['runs'],c['runs']))
        b,c,r=[results['controlled_'+label] for label in ['B1','correct','reverse']]
        cb,cc=[results['regular_'+label]['summary'] for label in ['B1','correct']]
        gate=all(preservation.values()) and c['summary']['success_count']>b['summary']['success_count'] and c['mean_case_max_abs_y_m']<min(b['mean_case_max_abs_y_m'],r['mean_case_max_abs_y_m']) and cc['complete'] and cb['complete'] and cc['mean_yaw_score_deg']<=cb['mean_yaw_score_deg']+.05
        assert completed==216
        atomic_json(out/'result.json',dict(completed_episodes=completed,training_steps=0,results=results,preservation=preservation,fixed_law_continue_gate=gate,registration_sha256=digest(out/'registration.json')))
        atomic_json(out/'progress.json',dict(status='complete',completed_episodes=completed))
    except BaseException as e:
        atomic_json(out/'interruption.json',dict(completed_episodes=completed,error=str(e),pending_job_consumption_unknown=True,silently_resumable=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:self_check(dict(kp=.4,kd=.3,roll_gain=0));print('PASS originalB1 equivalence, direction/mirror and reference bound')
    else:
        if args.output is None:parser.error('--output required')
        run(args.output)
