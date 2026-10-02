"""Same-state CPU/native decision parity, request bounds, resets and native rollout."""
from pathlib import Path
from dataclasses import replace
import argparse,json,importlib.util,sys
import numpy as np
import warp as wp
from native.environment import NativeEnv,reset_rows
from native.shared_reference import load,update
from probe_height_115_margin import cases
from probe_height_115_action_predict_loow import sha
ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'wheelleg_warp/results/twentythird_round_virtual_reference_20261002/experiment.py'


def run(output):
    assert not output.exists();output.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('reference_fixture_core',CORE)
    core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
    ref,effort,ends,metric,cut=load()
    scenes=cases()[:6]+[replace(s,mass=7.5,mu_l=.6,mu_r=1.,drive_difference=.03,delay_ms=10.) for s in cases()[4:6]]
    env=NativeEnv.height115_candidate(n=8,scenario=scenes,residual_scale=0,shared_reference=True)
    rows=[None]*8;peak=0.;decisions=0
    try:
        env.reset();offset=env.reference_offset
        feature=wp.zeros((8,15),dtype=core.D);shadow=wp.zeros_like(env.k['state']);request=wp.zeros_like(env.nominal_correction)
        for step in range(700):
            wp.launch(core.features,8,[env.data.qpos,env.data.qvel,env.data.sensordata,env.ids,env.k['heights'],env.k['angles'],env.k['state'],feature])
            x=feature.numpy();state=env.state.numpy();param=env.param.numpy();memory=env.k['state'].numpy();old=env.nominal_correction.numpy()
            expected=old.copy();context=memory[:,offset:offset+4].copy()
            for w in range(8):
                if state[w,1]<0:expected[w]=0;context[w]=0;continue
                target=np.zeros(6);d=int(param[w,0]<0)
                if param[w,13]<cut[0] and abs(param[w,0])>cut[1]:
                    if context[w,3]==0:context[w,1]=x[w,3];context[w,3]=1
                    if memory[w,13]>0:context[w,2]=1
                    if context[w,2]==0 and abs(context[w,1])>1e-6:
                        phase=int(context[w,0]);choices=np.arange(phase,min(phase+2,int(ends[d]))+1)
                        query=x[w,:6].copy();query[2]=0;query[3]*=ref[d,0,3]/context[w,1]
                        delta=ref[d,choices]-query;cost=np.einsum('ni,ij,nj->n',delta,metric,delta)
                        phase=int(choices[np.argmin(cost)]);context[w,0]=phase
                        target[:4]=(x[w,7:15].reshape(2,2,2)@effort[d,phase,:2]).reshape(4);target[4:]=effort[d,phase,2]
                        target/=max(1.,float(abs(target).max()));decisions+=1
                delta=target-old[w];expected[w]=np.clip(old[w]+delta*min(1.,.1/max(abs(delta).sum(),1e-30)),-1,1)
            wp.copy(shadow,env.k['state']);wp.copy(request,env.nominal_correction)
            args=env.shared_reference_args.copy();args[6]=shadow;args[-1]=request
            wp.launch(update,8,args)
            got=request.numpy();peak=max(peak,float(abs(got-expected).max()))
            assert peak<=1e-5,peak
            np.testing.assert_allclose(shadow.numpy()[:,offset:offset+4],context,rtol=0,atol=1e-12)
            assert abs(got).max()<=1 and abs(got-old).sum(axis=1).max()<=.1+1e-12
            _,_,done,infos=env.step(np.zeros((8,3),np.float32))
            for w in np.flatnonzero(done):
                if rows[w] is None:rows[w]={k:v for k,v in infos[w].items() if k!='terminal_observation'}
                assert not env.k['state'].numpy()[w,offset:].any() and not env.nominal_correction.numpy()[w].any()
            if all(r is not None for r in rows):break
        assert all(r is not None for r in rows) and all(r['success'] for r in rows)
        assert decisions>0
        try:env.set_nominal_correction(np.zeros((8,6)))
        except ValueError:pass
        else:raise AssertionError('Automatic reference manually overwritten')
        memory=env.k['state'].numpy();memory[:,-4:]=1;env.k['state'].assign(memory)
        env.mask.assign(np.array([1]+[0]*7,np.int32));wp.launch(reset_rows,8,env.reset_args)
        assert not env.k['state'].numpy()[0,-4:].any() and np.all(env.k['state'].numpy()[1:,-4:]==1)
        env.reset();assert not env.k['state'].numpy()[:,-4:].any()
        result=dict(passed=True,episodes=rows,native_cpu_request_peak_Nm=peak,decisions=decisions,
            same_state_phase_parity=True,masked_auto_full_reset_passed=True,manual_override_rejected=True,
            learning=False,full_admission=False,
            source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/shared_reference.py',ROOT/'wheelleg_warp/native/shared_reference.npz')})
        (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS native shared reference, CPU decisions, eight full tasks and resets',peak,flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
