"""Independent full-stream receipt of the repaired Cartesian delivery queue."""
import json
import numpy as np
import mujoco
from run_cartesian_repair import OUT,PARENT,inputs
from review_yaw_sector import ROOT,sha
from score_reference_learning import load_rows
from complete_contact_witness import check_files
from run_phase_support_qualification import logs
from parking_withdrawal_probe import check_log
from review_fixed_force_evaluation import chain_minima,physical_metrics
from analyze_reward_failures import flags
from cartesian_pair_action import request,F0,H0,W0
from check_cartesian_pair_kernel import bounds
from state_estimation import leg_kinematics
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

FIELDS=('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace','joint_guard_trace','actor_trace')


def pairs(a):
    return np.column_stack((a[:,0],-a[:,0],a[:,1],-a[:,1],a[:,2],-a[:,2]))


def projection(base,residual,limit):
    ratio=np.full_like(residual,np.inf)
    np.divide(limit-base,residual,out=ratio,where=residual>0)
    np.divide(-limit-base,residual,out=ratio,where=residual<0)
    return np.clip(np.minimum(1,ratio.min(axis=1)),0,1)


def check_memory(actor,c,parking):
    starts=np.arange(len(actor))*40
    np.testing.assert_array_equal(actor[:,383:389],pairs(c[starts,1:4]).astype(np.float32))
    np.testing.assert_array_equal(c[0,1:4],0)
    np.testing.assert_array_equal(c[1:,1:4],c[:-1,4:7])
    np.testing.assert_array_equal(c[:,4:7],c[:,1:4]+np.clip(parking[:,11:17:2]-c[:,1:4],-.01,.01))
    np.testing.assert_array_equal(parking[:,17:23],pairs(c[:,1:4]))
    np.testing.assert_array_equal(parking[:,23:29],pairs(c[:,4:7]))


def check_history(actor,pre):
    latest=actor[:,351:390];scale=np.array([40.]*4+[4.5]*2);count=len(actor)
    means=np.array([pre[40*i:40*(i+1),33:39].astype(float).mean(axis=0)/scale for i in range(count-1)])
    for i in range(count):
        for j in range(10):
            np.testing.assert_array_equal(actor[i,39*j:39*(j+1)],latest[max(0,i-9+j)])
        np.testing.assert_array_equal(actor[i,471:481],np.arange(10)>=9-i)
        for j in range(9):
            k=i-9+j;offset=390+8*j
            np.testing.assert_allclose(actor[i,offset:offset+6],means[k] if k>=0 else 0,rtol=0,atol=1e-6)
            np.testing.assert_array_equal(actor[i,offset+6:offset+8],0)
            assert actor[i,462+j]==(1 if k>=0 else 0)


def self_check():
    actor=np.zeros((1,968),np.float32);c=np.zeros((40,38));park=np.zeros((40,30));check_memory(actor,c,park)
    bad=actor.copy();bad[0,383]=.1
    try:check_memory(bad,c,park)
    except AssertionError:pass
    else:raise AssertionError('Canonical/foreign memory accepted as latent memory')
    np.testing.assert_array_equal(projection(np.zeros((1,6)),np.zeros((1,6)),np.ones((1,6))),1)
    np.testing.assert_allclose(projection(np.ones((1,6))*.9,np.ones((1,6))*.2,np.ones((1,6))),.5,atol=1e-15)
    actor[0,480]=1;check_history(actor,np.zeros((40,39)))
    actor[0,396]=1
    try:check_history(actor,np.zeros((40,39)))
    except AssertionError:pass
    else:raise AssertionError('Private proxy history accepted')
    print('PASS366 wrong latent memory/proxy rejection; accepted-Nom zero and active projection',flush=True)


def run():
    self_check();p=inputs();assert not (OUT/'review.json').exists()
    a=json.loads((OUT/'runtime_admission.json').read_text());done=json.loads((OUT/'runtime/completion.json').read_text());receipt=json.loads((OUT/'run_receipt.json').read_text())
    assert a['verified'] and all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    assert receipt['round']==365 and receipt['status']=='complete' and receipt['scoped_bindings_restored']
    assert receipt['result_sha256']==sha(OUT/'runtime/completion.json') and done['round']==receipt['engine_generation_round']==364
    assert done['verified'] and done['evaluations']==len(done['records'])==20 and done['new_learning_samples']==done['new_model_forward_rows']==0
    assert done['proposal_sha256']==sha(OUT/'runtime_proposal.json') and done['admission_sha256']==sha(OUT/'runtime_admission.json')
    assert [r['condition'] for r in done['records']]==[r['label'] for r in p['conditions']]
    spec=mujoco.MjSpec.from_file(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'));sim.hw.configure_spec(spec);model=spec.compile()
    qids=np.array([model.jnt_qposadr[model.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
    vids=np.array([model.jnt_dofadr[model.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')])
    hashes={};initial={};base={};details=[];steps=contacts=actor_count=cart_steps=zero_count=0
    for job,entry in zip(p['conditions'],done['records']):
        directory=OUT/'runtime'/job['label'];file=directory/'result.json';assert sha(file)==entry['sha256']
        assert entry['arm']==job['arm'] and entry['profile']==job['profile'] and entry['case']==job['case']['seed']
        row=load_rows(file,[job['case']])['runs'][0];assert row['success']==entry['success']==(not any(flags(row).values()))
        checked=check_files(directory,[job['case']],{row['seed']:row});checked.pop('original_label_differences')
        assert checked==entry['checked'] and logs(directory,[job['case']])==row['physical_steps'];contacts+=checked['contacts']
        streams={}
        for field in FIELDS:
            path=directory/row[field]['path'];assert sha(path)==row[field]['sha256'];hashes[str(path.relative_to(ROOT))]=sha(path)
            with np.load(path) as z:streams[field]={k:z[k].copy() for k in z.files}
        with np.load(directory/'initial.npz') as z:snap={k:z[k].copy() for k in z.files}
        assert sha(directory/'initial.npz')==entry['initial_sha256'] and sha(directory/'terminal.npz')==entry['terminal_sha256']
        assert {'q','v','ctrl','time','warm','sensors','param','reference','controller_memory','guard_memory','frames','inputs','elapsed','valid','geom_xpos','geom_xmat'}==set(snap)
        key=row['seed']
        if key not in initial:initial[key]=snap
        else:
            for name in snap:np.testing.assert_array_equal(snap[name],initial[key][name])
        full=streams['complete_trace'];t,pre,post=full['trace'],full['pre'],full['post'];n=len(t);steps+=n
        assert n==entry['first_episode_world_steps']<=p['per_job_first_episode_steps_cap']
        np.testing.assert_array_equal(snap['q'][0],pre[0,:17]);np.testing.assert_array_equal(snap['v'][0],pre[0,17:33])
        lengths,loop=chain_minima(post[:,:17],model);physical=physical_metrics(pre,post,model)
        assert min(lengths)>=HEIGHT_115_GEOMETRIC_MIN and physical[0]>=0 and max(physical[1:])<=1e-6
        np.testing.assert_allclose(lengths,[row['min_actual_A_leg_m'],row['min_actual_B_leg_m']],atol=1e-12,rtol=0)
        np.testing.assert_allclose(loop,row['max_loop_error_m'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(physical,[row['min_eight_joint_margin_rad'],row['max_actual_torque_excess_Nm'],row['max_command_torque_excess_Nm']],atol=1e-12,rtol=0)
        margin=1.4-abs(post[:,qids]).max(axis=1);np.testing.assert_array_equal(margin,t[:,24]);assert margin.min()==row['min_active_design_margin_rad']>=0
        with np.load(directory/'terminal.npz') as z:
            np.testing.assert_array_equal(z['terminal_q'][0],post[-1,:17]);np.testing.assert_array_equal(z['terminal_v'][0],post[-1,17:33])
        g=streams['joint_guard_trace']['trace'];assert g.shape==(n,62)
        with np.load(directory/'guard_prefix.npz') as z:np.testing.assert_array_equal(g,z['trace'])
        np.testing.assert_array_equal(g[:,50:54],pre[:,qids]);np.testing.assert_array_equal(g[:,54:58],pre[:,17+vids[:4]])
        np.testing.assert_array_equal(g[:,20:26],pre[:,33:39])
        park=streams['parking_trace']['trace'];check_log(park,'parking_request_withdrawal')
        actor=streams['actor_trace']['trace'];assert actor.shape==(int(np.ceil(n/40)),968) and len(actor)==entry['actor_rows'];actor_count+=len(actor)
        np.testing.assert_array_equal(actor[:,:481],actor[:,481:962]);np.testing.assert_array_equal(actor[0,:390],snap['frames'].reshape(-1))
        times=np.arange(len(actor))*.02
        latent=np.zeros((len(actor),3)) if job['profile']=='zero' else .15*np.column_stack((np.sin(np.pi*times),np.cos(np.pi*times),np.sin(2*np.pi*times)))
        np.testing.assert_array_equal(actor[:,962:],pairs(latent).astype(np.float32))
        np.testing.assert_array_equal(park[:,5:11],actor[np.arange(n)//40,962:].astype(float))
        starts=np.arange(len(actor))*40
        np.testing.assert_array_equal(actor[:,363:367],pre[starts][:,qids].astype(np.float32))
        np.testing.assert_array_equal(actor[:,367:373],pre[starts][:,17+vids].astype(np.float32))
        check_history(actor,pre)
        item=dict(condition=job['label'],arm=job['arm'],profile=job['profile'],success=row['success'],flags=flags(row),physical_steps=n,
            peak_deg=row['peak_deg'],guard_nominal_corrected=int((g[:,60]>1e-12).sum()),guard_residual_reduced=int((g[:,27]<1-1e-9).sum()))
        if job['arm']=='B0':base[key]=(streams,row)
        else:
            topology=json.loads((directory/'cartesian_topology.json').read_text())
            assert topology['accepted_nominal_mode']==1 and topology['candidate_kernel_sha256']==sha(ROOT/'wheelleg_warp/cartesian_pair_kernel_v2.py')
            assert all(topology['counts'][k]==80 for k in ('control','guard','prepare','finish','after'))
            for field in ('cartesian_trace','cartesian_terminal'):
                path=directory/row[field]['path'];assert sha(path)==row[field]['sha256'];hashes[str(path.relative_to(ROOT))]=sha(path)
            with np.load(directory/row['cartesian_trace']['path']) as z:c=z['trace'];clock=z['clock']
            assert c.shape==(n,38);cart_steps+=n;np.testing.assert_array_equal(clock,t[:,1:4]);np.testing.assert_array_equal(c[:,0],1);np.testing.assert_array_equal(c[:,33],0)
            check_memory(actor,c,park)
            np.testing.assert_array_equal(t[:,47:53],c[:,7:13]);np.testing.assert_array_equal(g[:,26],c[:,14]);np.testing.assert_array_equal(t[:,53],g[:,26]*g[:,27])
            with np.load(directory/'cartesian_initial.npz') as z:
                np.testing.assert_array_equal(z['latent'],0);np.testing.assert_array_equal(z['shadow'],0);reference=z['reference'][0].copy()
            reset_frame=snap['frames'][0,-1]
            reset_geometry=np.array([leg_kinematics(reset_frame[12+2*s:14+2*s].astype(float),np.zeros(2))[0] for s in range(2)])
            np.testing.assert_allclose(reference,reset_geometry.ravel(),rtol=0,atol=1e-7)
            expected_u=np.empty((n,6));expected_motor=np.empty((n,6));expected_alpha=np.empty(n);geometry_error=0.
            for i in range(n):
                parts=[leg_kinematics(pre[i,qids[2*s:2*s+2]],np.zeros(2)) for s in range(2)]
                geometry=np.array([x[0] for x in parts]) if job['arm']=='CartLive3' else reset_geometry
                geometry_error=max(geometry_error,float(abs(geometry.ravel()-c[i,34:38]).max()))
                u,alpha=request(geometry,c[i,4:7]);expected_u[i]=u;expected_alpha[i]=alpha
                for s in range(2):expected_motor[i,2*s:2*s+2]=parts[s][3]@np.array([F0*u[s],H0*u[2+s]])
                expected_motor[i,4:6]=W0*u[4:6]
            u_error=float(abs(expected_u-c[:,7:13]).max());motor_error=float(abs(expected_motor-c[:,15:21]).max())
            assert geometry_error<=1e-7 and u_error<=1e-6 and motor_error<=1e-6
            np.testing.assert_allclose(expected_alpha,c[:,13],rtol=0,atol=1e-7)
            limit=bounds(pre[:,17+vids]);np.testing.assert_allclose(limit,c[:,27:33],rtol=0,atol=1e-12)
            lam=projection(g[:,2:8],expected_motor,limit);np.testing.assert_allclose(lam,c[:,14],rtol=0,atol=1e-7)
            np.testing.assert_allclose(g[:,8:14],c[:,14,None]*c[:,15:21],rtol=0,atol=1e-12)
            candidate=np.clip(g[:,2:8]+c[:,14,None]*c[:,15:21],-c[:,27:33],c[:,27:33]).astype(np.float32)
            np.testing.assert_array_equal(candidate,c[:,21:27].astype(np.float32))
            final=np.clip(g[:,14:20]+g[:,27,None]*g[:,8:14],-c[:,27:33],c[:,27:33]).astype(np.float32)
            np.testing.assert_array_equal(final,pre[:,33:39].astype(np.float32))
            with np.load(directory/row['cartesian_terminal']['path']) as z:
                np.testing.assert_array_equal(z['latent'],c[-1,4:7]);np.testing.assert_array_equal(z['shadow'][16:22],pairs(c[-1:,4:7])[0]);np.testing.assert_array_equal(z['canonical'],c[-1,7:13])
                np.testing.assert_array_equal(z['reference'],reference)
            with np.load(directory/'cartesian_buffers.npz') as z:
                np.testing.assert_array_equal(z['latent'],0);np.testing.assert_array_equal(z['shadow'],0);np.testing.assert_array_equal(z['reference'][0],reference)
            item.update(canonical_replay_error=u_error,motor_replay_error_Nm=motor_error,geometry_error_m=geometry_error,
                withdrawn_substeps=int((park[:,4]==1).sum()),candidate_projected_substeps=int((c[:,14]<1-1e-12).sum()))
            if job['profile']=='zero':
                old,old_row=base[key];assert flags(row)==flags(old_row)
                for field in FIELDS:
                    assert streams[field].keys()==old[field].keys()
                    for name in streams[field]:np.testing.assert_array_equal(streams[field][name],old[field][name])
                zero_count+=1
            for name in ('cartesian_initial.npz','cartesian_buffers.npz','cartesian_topology.json'):hashes[str((directory/name).relative_to(ROOT))]=sha(directory/name)
        for name in ('result.json','initial.npz','terminal.npz','guard_prefix.npz','geometry.json','parking_contract.json'):
            hashes[str((directory/name).relative_to(ROOT))]=sha(directory/name)
        details.append(item);print('RECEIVED366',job['label'],n,'physical steps',flush=True)
    assert steps==done['first_episode_world_steps'] and actor_count==done['actor_rows']<=p['actor_row_budget'] and zero_count==8
    assert sum(e['actual_graph_world_steps'] for e in done['records'])==done['actual_graph_world_steps']<=p['actual_graph_world_step_budget'] and len(done['FD_calls'])<=p['constructor_FD_budget']
    assert sha(PARENT/'runtime/failure.json')==p['parent_failure_sha256']
    atomic_json(OUT/'review.json',dict(round=366,verified=True,evaluations=20,first_episode_world_steps=steps,contacts_checked=contacts,actor_rows=actor_count,
        cartesian_substeps_replayed=cart_steps,zero_exact_pairs=zero_count,physical_passed=20,design_passed=20,task_successes=sum(x['success'] for x in details),
        details=details,raw_sha256=hashes,proposal_sha256=sha(OUT/'runtime_proposal.json'),admission_sha256=sha(OUT/'runtime_admission.json'),completion_sha256=sha(OUT/'runtime/completion.json'),
        run_receipt_sha256=sha(OUT/'run_receipt.json'),reviewer_sha256=sha(__file__),original_failed_queue_preserved=True,
        new_physics_steps=0,new_FD=0,new_model_forward_rows=0,new_learning_samples=0,formal5_admitted=False,
        limits='Engineering delivery on four seen, zero-delay cases with scripted targets; not learning benefit, multiworld curriculum admission, delayed-sensor generalization or stability certification.',
        next='367 decide a concrete method/learning prequalification using the accepted interface and matched CartReset3; no automatic PPO from engineering pass.370 direction/cleanup/framework.'))


if __name__=='__main__':run()
