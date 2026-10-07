"""Build only frozen20anchor deltas; no validation fit, solver or gain construction."""
import json
import numpy as np
from review_yaw_sector import ROOT, sha
from dashboard.live_env import atomic_json
from motion_balanced_nominal import reference_values, sim, ml

OUT = ROOT/'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_map_v1'


def sample(h, v, table, height, command):
    """CPU reference for subsequent GPU interpolation checks, never runtime control."""
    h,v,table=map(np.asarray,(h,v,table))
    if (h.ndim!=1 or v.ndim!=1 or len(h)<2 or len(v)<2 or table.shape!=(len(h),len(v),4)
            or not np.isfinite(h).all() or not np.isfinite(v).all() or not np.isfinite(table).all()
            or not np.all(np.diff(h)>0) or not np.all(np.diff(v)>0)):
        raise ValueError('Finite increasing axes and NxMx4 table required')
    if not np.isfinite([height,command]).all() or not h[0]<=height<=h[-1] or not v[0]<=command<=v[-1]:
        raise ValueError('Outside registered public height/command domain')
    if command==0.:return np.zeros(4)
    i=min(int(np.searchsorted(h,height,side='right'))-1,len(h)-2)
    j=min(int(np.searchsorted(v,command,side='right'))-1,len(v)-2)
    a=(height-h[i])/(h[i+1]-h[i]);b=(command-v[j])/(v[j+1]-v[j])
    return (1-a)*((1-b)*table[i,j]+b*table[i,j+1])+a*((1-b)*table[i+1,j]+b*table[i+1,j+1])


def run():
    assert not (OUT/'table_manifest.json').exists() and not (OUT/'delta_table.npz').exists()
    p=json.loads((OUT/'proposal.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    h=np.array(p['height_knots']);v=np.array(p['speed_knots']);table=np.zeros((5,5,4))
    frozen=ROOT/p['frozen_bank']['path'];assert sha(frozen)==p['frozen_bank']['sha256']
    with np.load(frozen,allow_pickle=False) as z:
        np.testing.assert_array_equal(h,z['heights'])
        static_feed=z['feed'].copy();static_angles=z['angles'].copy()
    m,_=sim.load_model(ml.XML,True)
    assert (m.nq,m.nv,m.nu)==(17,16,6) and abs(m.body_mass.sum()-7.)<1e-12
    records=[]
    for r in p['fit_anchors']:
        path=ROOT/r['path'];assert sha(path)==r['sha256']
        assert sha(ROOT/r['cold_review_path'])==r['cold_review_sha256']
        with np.load(path,allow_pickle=False) as z:
            feed,theta,matrix,motor=reference_values(m,z['q'],z['ctrl'])
        np.testing.assert_allclose(matrix@feed,motor,rtol=0,atol=1e-12)
        i=int(np.flatnonzero(h==r['height'])[0]);j=int(np.flatnonzero(v==r['speed'])[0])
        table[i,j,:3]=feed-static_feed[i];table[i,j,3]=theta-static_angles[i]
        records.append(dict(**r,feed=feed.tolist(),theta=float(theta)))
    assert len(records)==20
    np.testing.assert_array_equal(table[:,2],0.)
    for r in records:
        i=int(np.flatnonzero(h==r['height'])[0]);delta=sample(h,v,table,r['height'],r['speed'])
        np.testing.assert_allclose(static_feed[i]+delta[:3],r['feed'],rtol=0,atol=1e-12)
        assert abs(static_angles[i]+delta[3]-r['theta'])<=1e-12
    for height in np.linspace(.115,.38,19):np.testing.assert_array_equal(sample(h,v,table,height,0.),0.)
    # Linear synthetic field must reproduce exact bilinear coordinates, including boundaries.
    synthetic=np.zeros_like(table)
    for i,height in enumerate(h):
        for j,speed in enumerate(v):synthetic[i,j]=[speed,height*speed,2*speed,3*height*speed]
    for height,speed in ((.115,-1.),(.38,1.),(.24,-.35),(.1375,.42),(.3,0.)):
        np.testing.assert_allclose(sample(h,v,synthetic,height,speed),
                                   [speed,height*speed,2*speed,3*height*speed],rtol=0,atol=1e-15)
    for height,speed in ((.1149,0.),(.3801,.5),(.2,1.01),(.2,-1.01),(.2,float('nan'))):
        try:sample(h,v,table,height,speed)
        except ValueError:pass
        else:raise AssertionError('Invalid domain accepted')
    validation=ROOT/p['validation_review']['path']
    assert sha(validation)==p['validation_review']['sha256']
    validations=json.loads(validation.read_text())['records']
    assert not {r['sha256'] for r in validations}.intersection({r['sha256'] for r in records})
    path=OUT/'delta_table.npz'
    np.savez_compressed(path,height_knots=h,speed_knots=v,delta=table,static_feed=static_feed,static_angles=static_angles)
    atomic_json(OUT/'table_manifest.json',dict(
        verified=True,proposal_sha256=sha(OUT/'proposal.json'),builder_sha256=sha(__file__),
        table_sha256=sha(path),fit_records=records,validation_points_fit=0,
        anchor_reconstruction_passed=True,zero_speed_exact0=True,linear_interpolation_and_domain_checks=True,
        optimization_calls=0,integration_steps=0,transitionFD_calls=0,controller_queries=0,new_training_samples=0,
        GPU_source_admitted=False,
        limits='CPU table/algebra only. Validation records only checked for disjointness, never fitted. GPU copy/interpolation/50static queries pending. Not continuous equilibrium/stability or new method qualification.'))
    print('PASS280 frozen20anchor table/reconstruction/zero-line/linear-domain; no validationfit/solve/physics/FD/PPO',flush=True)


if __name__=='__main__':
    run()
