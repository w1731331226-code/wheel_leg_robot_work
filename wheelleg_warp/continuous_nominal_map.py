"""GPU resident readonly bilinear delta preparation for isolated control copy."""
import json
import numpy as np
import warp as wp
from native.controller import D
from review_yaw_sector import ROOT, sha
from build_continuous_nominal_map import OUT, sample

ORIGINAL=ROOT/'wheelleg_warp/reference_role_control.py'
COPY=ROOT/'wheelleg_warp/continuous_nominal_control.py'
ACTIVE='    if active[w]==0:return\n'
THETA='    theta_eq=(D(1)-ratio)*angles[index]+ratio*angles[index+1]\n'
SUPPORT='    support=(D(1)-ratio)*feed[index,2]+ratio*feed[index+1,2]\n'
DOMAIN_EXTRA=(
    '    if reference.shape[1]==21 and reference[w,20]<D(0):\n'
    '        for j in range(6):ctrl[w,j]=0.\n'
    '        diagnostic[w,14]=D(2)\n'
    '        return\n')
THETA_EXTRA=(
    '    if reference.shape[1]==21 and reference[w,20]==D(1) and cmd!=D(0):\n'
    '        theta_eq=theta_eq+reference[w,19]\n')
FEED_EXTRA=(
    '    if reference.shape[1]==21 and reference[w,20]==D(1) and cmd!=D(0):\n'
    '        wheel=wheel+reference[w,16];hub=hub+reference[w,17];support=support+reference[w,18]\n')


def expected_source():
    text=ORIGINAL.read_text()
    assert all(text.count(s)==1 for s in (ACTIVE,THETA,SUPPORT))
    return text.replace(ACTIVE,ACTIVE+DOMAIN_EXTRA).replace(THETA,THETA+THETA_EXTRA).replace(SUPPORT,SUPPORT+FEED_EXTRA)


def arrays(device):
    manifest=json.loads((OUT/'table_manifest.json').read_text())
    assert manifest['verified'] and manifest['validation_points_fit']==0
    assert sha(OUT/'delta_table.npz')==manifest['table_sha256']
    with np.load(OUT/'delta_table.npz',allow_pickle=False) as z:
        h,v,t=z['height_knots'].copy(),z['speed_knots'].copy(),z['delta'].copy()
    p=json.loads((OUT/'proposal.json').read_text())
    np.testing.assert_array_equal(h,p['height_knots'])
    np.testing.assert_array_equal(v,p['speed_knots'])
    sample(h,v,t,h[0],0.)
    np.testing.assert_array_equal(t[:,2],0.)
    return [wp.array(x,dtype=D,device=device) for x in (h,v,t)]


@wp.kernel
def prepare(base:wp.array2d[D],command:wp.array[D],active:wp.array[int],
            heights:wp.array[D],speeds:wp.array[D],table:wp.array3d[D],
            reference:wp.array2d[D]):
    w=wp.tid()
    for j in range(16):reference[w,j]=base[w,j]
    for j in range(16,21):reference[w,j]=D(0)
    if active[w]==0:return
    h=base[w,2];v=command[w]
    if (not wp.isfinite(h) or not wp.isfinite(v) or h<heights[0] or
        h>heights[heights.shape[0]-1] or v<speeds[0] or v>speeds[speeds.shape[0]-1]):
        reference[w,20]=D(-1)
        return
    if v==D(0):return
    i=int(0);k=int(0)
    for j in range(1,heights.shape[0]-1):
        if h>heights[j]:i=j
    for j in range(1,speeds.shape[0]-1):
        if v>speeds[j]:k=j
    a=(h-heights[i])/(heights[i+1]-heights[i])
    b=(v-speeds[k])/(speeds[k+1]-speeds[k])
    for j in range(4):
        reference[w,16+j]=(D(1)-a)*((D(1)-b)*table[i,k,j]+b*table[i,k+1,j])+a*((D(1)-b)*table[i+1,k,j]+b*table[i+1,k+1,j])
    reference[w,20]=D(1)


if __name__=='__main__':
    assert not COPY.exists()
    COPY.write_text(expected_source())
