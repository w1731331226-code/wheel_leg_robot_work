"""V2 regression for a valid accepted Nom retaining a pre-clip invalid flag."""
import ast
import copy
import json
import numpy as np
import warp as wp
from cartesian_pair_kernel import D,deliver
from cartesian_pair_kernel_v2 import deliver_accepted_nominal
from cartesian_pair_action import OUT,request
from native.controller import fk
from state_estimation import leg_kinematics
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run():
    old=next(n for n in ast.parse((ROOT/'wheelleg_warp/cartesian_pair_kernel.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='deliver')
    new=next(n for n in ast.parse((ROOT/'wheelleg_warp/cartesian_pair_kernel_v2.py').read_text()).body if isinstance(n,ast.FunctionDef))
    old=copy.deepcopy(old);old.name=new.name
    call=next(n for n in ast.walk(old) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='project_bounds')
    assert call.args[-1].value==0;call.args[-1].value=1
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
    wp.init();wp.set_device('cuda:0');n=4
    q=np.zeros((n,4),np.float32);v=np.zeros((n,6),np.float32);base=np.tile([.25]*4+[-.5,4.],(n,1)).astype(np.float32)
    target=np.zeros((n,6),np.float32);target[2:]=[.1,-.1,-.2,.2,.3,-.3]
    leg=leg_kinematics(np.zeros(2),np.zeros(2))[0];reference=np.tile(np.r_[leg,leg],(n,1))
    report=[];saved={}
    for mode in (0,1):
        outputs=[]
        for name,kernel in [('v1',deliver),('v2',deliver_accepted_nominal)]:
            diag=np.zeros((n,38));diag[:,:6]=base;diag[:,12]=1;diag[:,13]=[0,1,0,1]
            lat=wp.zeros((n,3),dtype=D);state=wp.zeros((n,30),dtype=D);shadow=wp.zeros((n,22),dtype=D)
            ctrl=wp.array(base);dg=wp.array(diag,dtype=D);trace=wp.zeros((1,n,38),dtype=D)
            wp.launch(kernel,n,[0,mode,wp.array(q),wp.array(v),wp.array([0,1,2,3,0,1,2,3,4,5],dtype=int),
                wp.array(target),wp.ones(n,dtype=int),lat,wp.array(reference,dtype=D),state,shadow,ctrl,dg,trace])
            result=dict(ctrl=ctrl.numpy().copy(),diag=dg.numpy().copy(),trace=trace.numpy()[0].copy())
            np.testing.assert_array_equal(result['diag'][:,13],[0,1,0,1]);assert not result['trace'][:,33].any()
            outputs.append(result)
            for field,value in result.items():saved[f'{name}_mode{mode}_{field}']=value
        a,b=outputs
        np.testing.assert_array_equal(a['trace'][:,14],[1,0,1,0]);np.testing.assert_array_equal(b['trace'][:,14],1)
        np.testing.assert_array_equal(b['ctrl'][:2],base[:2])
        np.testing.assert_array_equal(b['ctrl'][2],b['ctrl'][3]);assert np.any(b['ctrl'][3]!=base[3])
        np.testing.assert_array_equal(a['ctrl'][[0,2]],b['ctrl'][[0,2]])
        np.testing.assert_array_equal(a['trace'][:,7:21][:,:6],b['trace'][:,7:21][:,:6])
        report.append(dict(mode=mode,old_lambda=a['trace'][:,14].tolist(),new_lambda=b['trace'][:,14].tolist(),
            historical_flag_preserved=True,unflagged_commands_unchanged=True,cold_zero_preserves_nominal=True,feasible_nonzero_not_suppressed=True))
    path=OUT/'nominal_mode_repair_fixture.npz';assert not path.exists();np.savez_compressed(path,**saved)
    atomic_json(OUT/'nominal_mode_repair_review.json',dict(round=364,verified=True,reports=report,
        only_algorithmic_change='effective project_bounds mode0 -> mode1 for the accepted-Nom interface; v1 source untouched',
        source_sha256={f:sha(ROOT/f) for f in ['wheelleg_warp/cartesian_pair_kernel.py','wheelleg_warp/cartesian_pair_kernel_v2.py','wheelleg_warp/test_cartesian_nominal_mode.py']},
        fixture_sha256=sha(path),real_cuda_kernel_row_evaluations=16,new_physics_steps=0,new_FD=0,new_learning_samples=0,
        installed=False,old_runtime_admission_unchanged=True,old_queue_still_failed=True,
        next='365 review a separately versioned repaired adapter/admission/protocol; no retry of the failed original queue.'))
    print('PASS364 v2 AST-only projection-mode change; live/reset x zero/nonzero x historical flag0/1; v1 preserved; no physics',flush=True)


if __name__=='__main__':run()
