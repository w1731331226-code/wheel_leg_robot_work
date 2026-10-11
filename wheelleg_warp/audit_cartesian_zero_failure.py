"""Receive the stopped zero pair and isolate projection semantics without a rerun."""
import ast
import json
import numpy as np
import warp as wp
from native.controller import D,V6,project_bounds
from cartesian_pair_action import OUT
from review_yaw_sector import ROOT,sha
from analyze_reward_failures import flags
from dashboard.live_env import atomic_json
from task_mode_recorder import COL


@wp.kernel
def modes(base:wp.array2d[D],limit:wp.array2d[D],result:wp.array2d[D]):
    w=wp.tid();b=V6();bounds=V6();inward=V6();zero=V6()
    for j in range(6):
        b[j]=base[w,j];bounds[j]=limit[w,j];inward[j]=-D(.01)*wp.sign(b[j])
    a,e0=project_bounds(b,zero,zero,bounds,1,False,False,False,0)
    c,e1=project_bounds(b,zero,zero,bounds,1,False,False,False,1)
    d,e2=project_bounds(b,inward,zero,bounds,1,False,True,False,0)
    e,e3=project_bounds(b,inward,zero,bounds,1,False,True,False,1)
    result[w,0]=a;result[w,1]=c;result[w,2]=d;result[w,3]=e
    result[w,4]=D(e0+e1+e2+e3)


def differing_columns(a,b):
    assert a.shape==b.shape and a.ndim==2
    return np.flatnonzero(np.any(a!=b,axis=0)).tolist()


def run():
    assert differing_columns(np.zeros((2,3)),np.array([[0,1,0],[0,0,0]]))==[1]
    queue=OUT/'runtime';assert (queue/'failure.json').exists() and not (queue/'completion.json').exists()
    assert not (OUT/'zero_failure_audit.json').exists()
    admission=json.loads((OUT/'runtime_admission.json').read_text())
    assert all(sha(ROOT/f)==h for f,h in admission['source_sha256'].items())
    failure=json.loads((queue/'failure.json').read_text());progress=json.loads((queue/'progress.json').read_text())
    assert failure['condition']['label']=='6301003_CartLive3_zero' and len(failure['records'])==progress['evaluations']==1
    directories=[queue/'6301003_B0_zero',queue/'6301003_CartLive3_zero']
    rows=[json.loads((p/'result.json').read_text())['runs'][0] for p in directories]
    assert all(r['physical_safety_passed'] and r['design_joint_passed'] and r['nominal_correction_enabled'] for r in rows)
    assert flags(rows[0])==flags(rows[1]) and rows[0]['physical_steps']==rows[1]['physical_steps']==14994
    hashes={};differences={};arrays={}
    for field in ('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace','joint_guard_trace','actor_trace'):
        pair=[]
        for directory,row in zip(directories,rows):
            file=directory/row[field]['path'];assert sha(file)==row[field]['sha256'];hashes[str(file.relative_to(ROOT))]=sha(file)
            with np.load(file) as z:pair.append({k:z[k].copy() for k in z.files})
        assert pair[0].keys()==pair[1].keys()
        arrays[field]=pair
        for name in pair[0]:
            a,b=pair[0][name],pair[1][name];assert a.shape==b.shape
            if not np.array_equal(a,b):
                assert a.ndim==2
                differences[field+'/'+name]=dict(columns=differing_columns(a,b),elements=int((a!=b).sum()))
    assert differences=={'complete_trace/trace':dict(columns=[53],elements=140),'joint_guard_trace/trace':dict(columns=[26],elements=140)}
    assert COL[53]=='original_lambda'
    a,b=[x['trace'] for x in arrays['complete_trace']]
    changed=a[:,53]!=b[:,53]
    preclip=np.any(abs(np.clip(a[:,31:33],-4.5,4.5))>a[:,58:60]+1e-9,axis=1)
    np.testing.assert_array_equal(changed,preclip)
    assert all(r['base_infeasible_steps']==140 for r in rows)
    for name in ('initial.npz','terminal.npz'):
        with np.load(directories[0]/name) as x,np.load(directories[1]/name) as y:
            assert set(x.files)==set(y.files)
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    node=ast.parse((ROOT/'wheelleg_warp/cartesian_pair_kernel.py').read_text())
    calls=[x for x in ast.walk(node) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='project_bounds']
    assert len(calls)==1 and isinstance(calls[0].args[-1],ast.Constant) and calls[0].args[-1].value==0
    n=int(changed.sum());base=np.zeros((n,6));bounds=np.full((n,6),40.)
    base[:,4:6]=a[changed,33:35];bounds[:,4:6]=a[changed,58:60]
    assert np.all(abs(base)<=bounds+1e-12)
    wp.init();wp.set_device('cuda:0');result=wp.zeros((n,5),dtype=D)
    wp.launch(modes,n,[wp.array(base,dtype=D),wp.array(bounds,dtype=D),result])
    values=result.numpy();np.testing.assert_array_equal(values,np.tile([0,1,0,1,0],(n,1)))
    sample=OUT/'projection_mode_witness.npz'
    np.savez_compressed(sample,step_indices=np.flatnonzero(changed),clock=a[changed,2:4],accepted_wheel_nominal=base[:,4:6],
        wheel_bounds=bounds[:,4:6],projection_results=values,fields=np.array(['mode0_zero','mode1_zero','mode0_inward','mode1_inward','error_sum']))
    for directory in directories:
        for name in ('result.json','initial.npz','terminal.npz'):hashes[str((directory/name).relative_to(ROOT))]=sha(directory/name)
    atomic_json(OUT/'zero_failure_audit.json',dict(round=364,verified=True,queue_failed=True,complete_episodes_written=2,accepted_queue_records=1,remaining_unrun=18,
        actual_graph_world_steps=failure['actual_graph_world_steps'],actor_rows=failure['actor_rows'],FD_calls=len(failure['FD_calls']),
        physical_steps_compared=29988,physical_and_design_passed=2,full_task_flags_equal=True,
        physical_qv_commands_contacts_sensors_actor_equal=True,common_initial_and_terminal_arrays_equal=True,differences=differences,
        first_difference_pre_s=float(a[np.flatnonzero(changed)[0],2]),preclip_wheel_flag_support_exact=True,
        residual_limited_steps=[r['residual_limited_steps'] for r in rows],mean_residual_lambda=[r['mean_residual_lambda'] for r in rows],
        cause='Candidate hardcodes projection_mode=0 although upstream shared nominal-correction path forces effective mode1 after accepting/clipping Nom. Raw project_clipped_base=False is not the effective mode. The historical invalid_base flag persists for diagnosis and must not disable residuals on accepted Nom.',
        source_scope='reference_role_control.control_step lines180-199 and its nominal_correction_enabled=1 call; native.controller.project_bounds branch. No source or failure gate modified.',
        projection_probe_rows=n,projection_helper_evaluations=4*n,synthetic_inward_limit='Only demonstrates helper semantics on accepted wheel Nom and supplied limits; not a Cartesian policy action or rollout.',
        witness_sha256=sha(sample),raw_sha256=hashes,auditor_sha256=sha(__file__),admission_sha256=sha(OUT/'runtime_admission.json'),failure_sha256=sha(queue/'failure.json'),
        new_physics_steps=0,new_FD=0,new_model_forward_rows=0,new_learning_samples=0,
        next='365 direction/cleanup: require a versioned accepted-Nom mode repair and flagged-Nom unit regression before any new separately registered queue. Original20 queue remains failed; do not exclude lambda or silently resume it.'))
    print('CONFIRMED364 physical equality;140 lambda mismatches exactly match preclip wheel flags;140x4 projection helper outputs [0,1,0,1];no rerun',flush=True)


if __name__=='__main__':run()
