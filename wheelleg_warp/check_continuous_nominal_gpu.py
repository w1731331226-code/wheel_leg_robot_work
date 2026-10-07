"""Source reversal and GPU preparation checks; no actual controller queries."""
import ast
import json
import numpy as np
import warp as wp
import continuous_nominal_map as mapping
import continuous_nominal_control as candidate
from native.controller import D
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json

OUT=mapping.OUT


def run():
    assert not (OUT/'GPU_source_admission.json').exists()
    p=json.loads((OUT/'proposal.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    manifest=json.loads((OUT/'table_manifest.json').read_text())
    assert manifest['proposal_sha256']==sha(OUT/'proposal.json')
    assert mapping.COPY.read_text()==mapping.expected_source()
    source=mapping.COPY.read_text()
    for extra in (mapping.DOMAIN_EXTRA,mapping.THETA_EXTRA,mapping.FEED_EXTRA):
        assert source.count(extra)==1
        source=source.replace(extra,'')
    assert source==mapping.ORIGINAL.read_text()
    assert ast.dump(ast.parse(source))==ast.dump(ast.parse(mapping.ORIGINAL.read_text()))
    wp.init();device=wp.get_device('cuda:0')
    wp.load_module(candidate,device=device)
    h,v,t=mapping.arrays(device)
    before=[x.numpy().copy() for x in (h,v,t)]
    records=[dict(height=float(height),speed=float(speed),kind='node')
             for height in before[0] for speed in before[1]]
    review=json.loads((ROOT/p['validation_review']['path']).read_text())
    records.extend(dict(height=r['height'],speed=r['speed'],kind='nonfitvalidation') for r in review['records'])
    records.extend(dict(height=float(x),speed=0.,kind='zero') for x in np.linspace(.115,.38,19))
    records.extend(dict(height=a,speed=b,kind='invalid') for a,b in (
        (.11499,0.),(.38001,.7),(.2,-1.00001),(.2,1.00001),
        (float('nan'),.7),(.2,float('nan')),(float('inf'),0.),(.2,float('inf'))))
    records.append(dict(height=.24,speed=.7,kind='inactive'))
    n=len(records);base=np.random.default_rng(281).normal(size=(n,16))
    base[:,2]=[r['height'] for r in records]
    command=np.array([r['speed'] for r in records])
    active=np.ones(n,np.int32);active[-1]=0
    base_gpu=wp.array(base,dtype=D,device=device)
    command_gpu=wp.array(command,dtype=D,device=device)
    active_gpu=wp.array(active,dtype=int,device=device)
    reference=wp.zeros((n,21),dtype=D,device=device)
    wp.launch(mapping.prepare,n,[base_gpu,command_gpu,active_gpu,h,v,t,reference],device=device)
    result=reference.numpy()
    np.testing.assert_array_equal(result[:,:16],base)
    maximum=0.
    for i,r in enumerate(records):
        if r['kind']=='invalid':
            assert result[i,20]==-1.
            np.testing.assert_array_equal(result[i,16:20],0.)
        elif r['kind']=='inactive' or r['speed']==0.:
            np.testing.assert_array_equal(result[i,16:],0.)
        else:
            cpu=mapping.sample(*before,r['height'],r['speed'])
            difference=float(abs(cpu-result[i,16:20]).max())
            maximum=max(maximum,difference)
            np.testing.assert_allclose(result[i,16:20],cpu,rtol=0,atol=1e-12)
            assert result[i,20]==1.
    for array,saved in zip((h,v,t),before):np.testing.assert_array_equal(array.numpy(),saved)
    np.testing.assert_array_equal(base_gpu.numpy(),base)
    np.testing.assert_array_equal(command_gpu.numpy(),command)
    np.savez_compressed(OUT/'GPU_preparation_checks.npz',base=base,command=command,active=active,prepared=result,
                        height_knots=before[0],speed_knots=before[1],delta_table=before[2])
    atomic_json(OUT/'GPU_source_admission.json',dict(
        verified=True,proposal_sha256=sha(OUT/'proposal.json'),table_manifest_sha256=sha(OUT/'table_manifest.json'),
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (
            mapping.COPY,ROOT/'wheelleg_warp/continuous_nominal_map.py',ROOT/'wheelleg_warp/check_continuous_nominal_gpu.py')},
        preparation_array_sha256=sha(OUT/'GPU_preparation_checks.npz'),prepared_rows=n,
        valid_nodes=25,nonfit_validation_rows=20,zero_rows=19,invalid_rows=8,inactive_rows=1,
        GPU_CPU_delta_max_difference=maximum,source_reversal_exact=True,readonly_table_and_inputs=True,
        GPU_candidate_compiled=True,invalid_flag_preboot_rejection_source_checked=True,
        zero_packet_disabled_exact=True,zero_controller_runtime_equality_pending=True,
        controller_queries=0,integration_steps=0,transitionFD_calls=0,optimization_calls=0,new_training_samples=0,
        registered50static_queries_admitted=True,production_admitted=False,
        limits='Compilation/source/GPUprepare only. No controlkernel launched. Runtime zero output/diag/memory preservation and physicaltorque gaps checked only in subsequent50queries; no dynamics/task/PPO admission.'))
    print('PASS281 GPUcompile/source reversal/73preparedrows; CPUmaxdifference',maximum,
          ';0controller/physics/FD/PPO',flush=True)


if __name__=='__main__':
    run()
