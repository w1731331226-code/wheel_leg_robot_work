"""Algebraic checks only: no simulator, training or held-out evaluation."""
import json
import math
from pathlib import Path


def transpose(a):
    return list(map(list, zip(*a)))


def product(a, b):
    return [[sum(x*y for x,y in zip(row,col)) for col in transpose(b)] for row in a]


def close(a, b):
    return all(math.isclose(x,y,abs_tol=1e-12) for r,s in zip(a,b) for x,y in zip(r,s))


def run():
    d = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    c = [[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]]
    theta = math.pi/6
    mixed = {}
    for name,sign in [('N3_plus',1),('N3_minus',-1)]:
        n = [[math.cos(theta)*d[i][j]+sign*math.sin(theta)*c[i][j] if j<2 else d[i][j]
              for j in range(3)] for i in range(6)]
        assert close(product(transpose(n),n), [[2,0,0],[0,2,0],[0,0,2]])
        assert [r[2] for r in n] == [r[2] for r in d]
        # A nonzero common component proves this is not a rotation inside span(D).
        common = product(transpose(c),n)
        assert abs(common[0][0]) > .9 and abs(common[1][1]) > .9
        mixed[name] = n
    swapped = [mixed['N3_plus'][i] for i in [1,0,3,2,5,4]]
    assert close(swapped,[[-x for x in row] for row in mixed['N3_minus']])
    data = dict(status='algebra_verified_not_training_ready',theta_degrees=30,
                virtual_order=['F_L','F_R','T_L','T_R','wheel_L','wheel_R'],
                M1=[[r[2]] for r in d], M3=d, **mixed,
                shared_column_gram=[[2,0,0],[0,2,0],[0,0,2]],
                maximum_leg_component=math.cos(theta)+math.sin(theta),
                note='Equal normalized virtual column energy, not identical per-motor covariance or saturation.')
    path = Path(__file__).with_name('basis.json')
    if path.exists():
        assert json.loads(path.read_text()) == data
    else:
        path.write_text(json.dumps(data,indent=2)+'\n')
    print('PASS: rank 3, equal Gram matrices, non-differential spans, shared wheel authority, left/right mirror pair')


if __name__ == '__main__':
    run()
