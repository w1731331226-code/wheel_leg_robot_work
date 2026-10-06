"""Unchanged collision geometry and data sufficiency, no physics integration."""
import json
import sys
import numpy as np
import mujoco
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from native.terrain import HeightTerrainScenario,model
from review_yaw_sector import sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/task_mode_witness_v1'


def support_extent(radii,rotation,direction):
    a=np.asarray(radii,float); R=np.asarray(rotation,float); n=np.asarray(direction,float)
    if a.shape!=(3,) or R.shape!=(3,3) or n.shape!=(3,) or not all(np.isfinite(x).all() for x in (a,R,n)) or np.any(a<=0) or not np.isclose(n@n,1.) or not np.allclose(R.T@R,np.eye(3),atol=1e-6):
        raise ValueError('Expected positive ellipsoid radii, orthogonal rotation and unit direction')
    return float(np.linalg.norm(a*(R.T@n)))


def unit():
    radii=[.05,.0275,.05]; I=np.eye(3)
    assert support_extent(radii,I,[0,1,0])==.0275 and support_extent(radii,I,[0,0,1])==.05
    theta=.3; R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    assert np.isclose(support_extent(radii,R,[0,0,1]),.05)
    # Centre60mm from a35mm half-lane can still have2.5mm lateral overlap.
    assert .060-support_extent(radii,I,[0,1,0])-.035<0
    assert .064-support_extent(radii,I,[0,1,0])-.035>0
    try: support_extent([0,.02,.05],I,[0,1,0])
    except ValueError: pass
    else: raise AssertionError('Invalid shape accepted')


def run():
    unit(); proposal=json.loads((OUT/'proposal.json').read_text()); contract=json.loads((OUT/'source_contract.json').read_text())
    assert contract['proposal_sha256']==sha(OUT/'proposal.json') and all(sha(ROOT/n)==v for n,v in contract['source_sha256'].items())
    geometries=[]
    for case in proposal['cases']:
        m=model(HeightTerrainScenario(**case['scenario'])); wheels=[]
        for side in ('L','R'):
            name='wheel_collide_'+side; g=m.geom(name).id; b=int(m.geom_bodyid[g]); chain=[]
            assert m.geom_type[g]==mujoco.mjtGeom.mjGEOM_ELLIPSOID and np.allclose(m.geom_pos[g],0) and np.allclose(m.geom_quat[g],[1,0,0,0])
            assert np.isclose(m.geom_size[g,0],m.geom_size[g,2])
            while b:
                start=int(m.body_jntadr[b]); length=int(m.body_jntnum[b])
                joints=list(range(start,start+length)) if start>=0 else []
                if any(m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE for j in joints): break
                assert np.allclose(m.body_quat[b],[1,0,0,0])
                for j in joints:
                    assert m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and np.allclose(abs(m.jnt_axis[j]),[0,1,0])
                    chain.append(dict(joint=m.joint(j).name,axis=m.jnt_axis[j].tolist()))
                b=int(m.body_parentid[b])
            assert b>0
            wheels.append(dict(name=name,geom_id=g,type='ellipsoid',radii_m=m.geom_size[g].tolist(),full_width_m=2*float(m.geom_size[g,1]),local_quaternion=m.geom_quat[g].tolist(),local_offset=m.geom_pos[g].tolist(),joint_chain=chain,root_free_body=m.body(b).name,condim=int(m.geom_condim[g]),friction=m.geom_friction[g].tolist()))
        geometries.append(dict(case=case['seed'],wheels=wheels))
    allsizes=[w['radii_m'] for e in geometries for w in e['wheels']]
    assert all(np.array_equal(x,allsizes[0]) for x in allsizes)
    checklist=[
        dict(claim='Post-time ellipsoid support extents/AABB',status='derivable with bounded rounding assumptions',reason='Post roll/pitch/yaw plus verified zero-offset identity-quat ellipsoid. Circular x/z section makes Y-axis chain/wheel spin irrelevant to shape orientation. Not contact/load certificate.'),
        dict(claim='Solver-time exact whole-tyre pose/clearance',status='not fully recorded',reason='Pre wheel centres exist, pre body quaternion/Euler and all joint states not recorded. Post orientation cannot be substituted for pre force time.'),
        dict(claim='A positive target contact normal/vertical-normal witness',status='available',reason='Force API/unit and time labels validated. It may be an edge/front contact; normal component is not total friction-inclusive support.'),
        dict(claim='Complete per-face/per-geometry support duty',status='incomplete',reason='Only per-wheel sums and maximum-normal/maximum-vertical witnesses. A smaller simultaneous top contact can be hidden; vertical witness distance is not separately stored.'),
        dict(claim='Whole centred passage and mixed-event progression',status='not qualified',reason='Legacy centre path is only a prerequisite; mixed target/event mapping and ground/airborne operational rules pending.'),
        dict(claim='Original task physical/design/success labels',status='retained',reason='No geometry/controller/task changes or new labels from this audit.')]
    result=dict(verified=True,compiled_static_models=41,wheel_geometries=82,geometry=geometries,
        invariant_radii_m=allsizes[0],invariant_full_width_m=.055,legacy_box_half_width_m=.035,neutral_lateral_overlap_limit_m=.0625,
        support_formula='rho(n)=norm(diag(radii)*R.T*n). For a=c, rotations about localY leave ellipsoid unchanged. Projection overlap is necessary geometric possibility, not actual collision/support/full traversal.',
        evidence_checklist=checklist,source_sha256=sha(__file__),proposal_sha256=sha(OUT/'proposal.json'),source_contract_sha256=sha(OUT/'source_contract.json'),
        new_physics_rollouts=0,training_updates=0,
        limits='CPU model compilation only, not simulated replay. GPU float32 geometry/pose rounding and sampled post orientation do not make exact continuous or solver-time contact claims.',
        next='167 operational definitions and missing-evidence handling, original labels intact; no model repair/learning based on geometry alone;170 deep review/cleanup.')
    atomic_json(OUT/'wheel_geometry_evidence_audit.json',result)
    print('PASS41 static models/82 ellipsoids: radii',allsizes[0],'fullwidth55mm; time-layer/support gaps explicit;0rollout',flush=True)


if __name__=='__main__': run()
