"""首个可观察接触后：同图七臂1Nm单步响应与局部仿射制动诊断。"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from probe_height_115_action_predict_loow import ROOT, DT, sha, compare_start
from probe_height_115_braking_budget import ACTIVE, barriers
from probe_height_115_live_common_local import common_basis, simulate
from probe_height_115_radial_authority import torque_box
from native.terrain import model, HeightTerrainScenario


def run(peak_nm=1.):
    assert peak_nm in (1.,2.)
    folder=ROOT/'wheelleg_warp/results'
    output=folder/('height_115_contact_response_20260929' if peak_nm==1. else 'height_115_contact_response_2nm_20260929')
    assert not output.exists()
    source=folder/'height_115_switching_error_20260929'
    paths=[folder/n/'verification.json' for n in ('height_115_contact_budget_20260929',
        'height_115_action_predict_1nm_single_graph_20260929','height_115_local_states_20260928')]
    budget,fit,archive=[json.loads(p.read_text()) for p in paths]
    original=json.loads((source/'verification.json').read_text())
    output.mkdir();rows=[]
    for row in budget['rows']:
        world=row['world'];path=source/f'world_{world}.npz'
        assert sha(path)==row['trace_sha256']
        old=np.load(path);offset=row['first_contact_step']+1
        start=original['worlds'][world]['start_step']+offset
        event=archive['events'][fit['selected_event_ids'][world]]
        m=model(HeightTerrainScenario(**event['scenario']))
        q=old['pre'][0,offset,:m.nq];v=old['pre'][0,offset,m.nq:m.nq+m.nv]
        B=common_basis(m,q,v);eps=peak_nm/np.max(abs(B),axis=0)
        coeff=np.zeros((7,3))
        for k in range(3):coeff[1+2*k,k]=eps[k];coeff[2+2*k,k]=-eps[k]
        schedule=np.zeros((40,7,3));schedule[0]=coeff
        rec,starts,stats,_=simulate([HeightTerrainScenario(**event['scenario'])],[start],np.zeros((7,3)),schedule=schedule)
        qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
        np.savez_compressed(output/f'world_{world}.npz',coefficients=coeff,starts=starts,
            previous_velocity=rec[0]['previous_qvel'],
            **{key:np.stack([r[key] for r in rec]) for key in ('pre','post','post_velocity','applied')},
            contact_raw=rec[0]['contact_raw'])
        qerr=float(abs(rec[0]['pre'][0,:m.nq]-q).max())
        verr=float(abs(rec[0]['pre'][0,m.nq:m.nq+m.nv]-v).max())
        assert qerr<=2e-6 and verr<=1e-6,(qerr,verr)
        q=rec[0]['pre'][0,:m.nq];v=rec[0]['pre'][0,m.nq:m.nq+m.nv];B=common_basis(m,q,v)
        nominal=rec[0]['pre'][0,m.nq+m.nv:m.nq+m.nv+6];limit=torque_box(m,v)
        rsv=fit['folds'][world]['reserves'];reserve=(rsv['actual_A_length_m'],rsv['eight_joint_margin_rad'])
        oldgain=np.array(fit['folds'][world]['gain']);acc=[];rad=[];match=0.
        base,_=barriers(q[qa],v[va],np.zeros(4),oldgain,reserve)
        for arm,r in enumerate(rec):
            compare_start(starts[arm],starts[0],m.nq,m.nv,f'world{world} arm{arm}')
            assert r['pre'][0,-1]==1
            assert np.max(abs(r['applied'][0])-limit)<=1e-6
            match=max(match,float(abs(r['applied'][0]-nominal-B@coeff[arm]).max()))
            acc.append((r['post_velocity'][0,va]-v[va])/DT)
            nxt,_=barriers(r['post'][0,qa],r['post_velocity'][0,va],np.zeros(4),oldgain,reserve)
            rad.append([(nxt[j]['rate']-base[j]['rate'])/DT for j in range(2)])
        acc=np.array(acc);rad=np.array(rad)
        assert match<=1e-5 and np.max(abs(B@coeff.T))<=peak_nm+1e-5
        gain=np.stack([(acc[1+2*k]-acc[2+2*k])/(2*eps[k]) for k in range(3)],axis=1)
        radialgain=np.stack([(rad[1+2*k]-rad[2+2*k])/(2*eps[k]) for k in range(3)],axis=1)
        new,_=barriers(q[qa],v[va],acc[0],gain,reserve)
        for j in range(2):new[j]['a0']=rad[0,j];new[j]['g']=radialgain[j]
        A=np.concatenate([B,-B,B,-B]);rhs=np.r_[np.full(12,peak_nm),limit-nominal,limit+nominal]
        closing=[];constraint=[];limits=[]
        for b in new:
            assert b['h']>0
            if b['rate']>=0:continue
            required=b['rate']**2/(2*b['h'])
            sol=linprog(-b['g'],A_ub=A,b_ub=rhs,bounds=[(None,None)]*3,method='highs')
            assert sol.success and np.max(A@sol.x-rhs)<1e-6
            closing.append(dict(name=b['name'],required=required,
                measured_local_affine_max=float(b['a0']+b['g']@sol.x),maximizing_action=sol.x.tolist()))
            constraint.append(-b['g']);limits.append(b['a0']-required)
        coupled=linprog(np.zeros(3),A_ub=np.r_[A,np.array(constraint)],b_ub=np.r_[rhs,limits],bounds=[(None,None)]*3,method='highs')
        if coupled.success:assert np.max(np.r_[A,np.array(constraint)]@coupled.x-np.r_[rhs,limits])<1e-6
        previous=(v[va]-rec[0]['previous_qvel'][va])/DT
        result=dict(world=world,start_step=start,start_q_error=qerr,start_v_error=verr,
            motor_match_error_Nm=match,first_step_contact_pairs_equal=[rec[a]['contacts'][0]==rec[0]['contacts'][0] for a in range(7)],
            measured_active_acceleration=acc.tolist(),measured_radial_acceleration=rad.tolist(),
            local_gain=gain.tolist(),local_radial_gain=radialgain.tolist(),
            old_gain_response_max_error_rad_s2=float(abs(coeff@oldgain.T-(acc-acc[0])).max()),
            local_affine_response_max_error_rad_s2=float(abs(coeff@gain.T-(acc-acc[0])).max()),
            local_affine_radial_max_error_m_s2=float(abs(coeff@radialgain.T-(rad-rad[0])).max()),
            previous_acceleration_baseline_error_rad_s2=(previous-acc[0]).tolist(),
            local_affine_coupled_feasible=bool(coupled.success),closing_barriers=closing,
            coupled_action=None if not coupled.success else coupled.x.tolist(),
            trace_sha256=sha(output/f'world_{world}.npz'))
        rows.append(result)
        print(world,'radial',rad.tolist(),'feasible',coupled.success,flush=True)
    result=dict(role='paired_first_contact_one_step_response',peak_nm=peak_nm,rows=rows,
        original_1nm_source_git_revision='887733d',
        limitations=['Only first 0.5ms pulse scored; remaining helper rollout is unscored.',
            'Local gain and zero baseline use future paired simulation results: diagnostic oracle, not online estimator.',
            'Central finite differences fit these six axis actions; combined-action envelope unvalidated.',
            'Incremental motor trust box is not full hardware authority; no continuous safety claim.'],
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[source/'verification.json']},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_live_common_local.py',
            ROOT/'wheelleg_warp/probe_height_115_braking_budget.py',ROOT/'wheelleg_warp/native/controller.py',ROOT/'wheelleg_warp/native/environment.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--peak-nm',type=float,choices=(1.,2.),default=1.)
    run(parser.parse_args().peak_nm)
