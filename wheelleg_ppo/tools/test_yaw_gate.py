"""Synthetic-only checks: does not open scenario manifests or real evaluation logs."""
from copy import deepcopy
import math

from score_yaw_gate import assess, REFERENCES, SEEDS


def fixture():
    scenario = dict(center=2., offset=0., speed=1.)
    gate = [dict(seed=i, scenario=scenario) for i in range(64)]
    regression = [dict(name=str(i), scenario=scenario) for i in range(28)]
    def rows(cases, score):
        return [dict(case, success=True, reason='completed', physical_steps=12000,
                     duration_s=6., arrival_s=4., rms_deg=[0.,0.,score*math.sqrt(7.625/6)],
                     peak_deg=[.1,.1,2.], velocity_rmse=.05, stop_distance_m=.1, tail_speed_m_s=.001)
                for case in cases]
    payload = dict(protocol='yaw-precision-v2', gate={}, regression={'B0':{'fixed':rows(regression,1.)}})
    for method in (*REFERENCES,'M3'):
        payload['gate'][method] = {seed:rows(gate,.8 if method=='M3' else 1.)
                                  for seed in (('fixed',) if method in ('B0','B1') else SEEDS)}
    payload['regression']['M3'] = {seed:rows(regression,.8) for seed in SEEDS}
    return payload, gate, regression


if __name__ == '__main__':
    original, gate, regression = fixture()
    assert assess(original,gate,regression)['passed']
    def changed(change, error=None):
        p = deepcopy(original); change(p)
        if error == 'invalid':
            try: assess(p,gate,regression)
            except ValueError: return
            raise AssertionError('Invalid data accepted')
        result = assess(p,gate,regression)
        assert not result['passed'], error
        assert any(error in text for text in result['failures']), result
    changed(lambda p:p['gate'].pop('N3-'), 'invalid')
    changed(lambda p:p['gate']['M3'].pop('1611'), 'invalid')
    changed(lambda p:p['gate']['M3']['1609'].pop(), 'invalid')
    changed(lambda p:p['gate']['M3']['1609'].__setitem__(0, p['gate']['M3']['1609'][1]), 'invalid')
    changed(lambda p:p['gate']['M3']['1609'][0].update(scenario=dict(center=1.,offset=0.,speed=1.)), 'invalid')
    changed(lambda p:p['gate']['B0'].update({'1609':p['gate']['B0']['fixed']}), 'invalid')
    changed(lambda p:p['gate']['M3']['1609'][0].update(success=False,reason='fall',rms_deg=[0,0,0]), 'incomplete gate')
    changed(lambda p:p['gate']['M3']['1609'][0].update(velocity_rmse=.1), 'velocity regression')
    # A longer but complete task must still respect the arrival constraint.
    changed(lambda p:p['gate']['M3']['1609'][0].update(arrival_s=5.,duration_s=7.,physical_steps=14000), 'arrival regression')
    changed(lambda p:p['regression']['M3']['1609'][0].update(success=False), 'nominal success')
    changed(lambda p:p['regression']['M3']['1609'][0].update(peak_deg=[.21,.1,2.]), 'nominal roll/pitch')
    changed(lambda p:p['regression']['M3']['1609'][0].update(velocity_rmse=.1), 'nominal velocity')
    changed(lambda p:p['gate']['M3']['1609'][0].update(rms_deg=[0,0,float('nan')]), 'invalid')
    changed(lambda p:p['gate']['M3']['1609'][0].update(peak_deg=[.1,.1,6.]), 'invalid')
    # Losing one B0-success case cannot be offset by succeeding on another case.
    p=deepcopy(original)
    p['gate']['B0']['fixed'][0]['success']=False
    for method in REFERENCES[1:]:
        for rows in p['gate'][method].values(): rows[0]['success']=False
    for rows in p['gate']['M3'].values(): rows[1]['success']=False
    assert any('lost B0 success' in x for x in assess(p,gate,regression)['failures'])
    # The newly added N3 baseline must participate in best-reference selection.
    changed(lambda p:[r.update(rms_deg=[0,0,.7*math.sqrt(7.625/6)])
                       for rows in p['gate']['N3-'].values() for r in rows], 'effect')
    # Mean advantage is insufficient when a paired seed reverses direction.
    p=deepcopy(original)
    for seed,rows in p['gate']['M3'].items():
        for row in rows: row['rms_deg']=[0,0,(1.1 if seed=='1609' else .6)*math.sqrt(7.625/6)]
    result=assess(p,gate,regression)
    assert any('paired direction' in x for x in result['failures'])
    assert not any('effect:' in x for x in result['failures'])
    for base_score, candidate_score, passed in ((1.,.85,True),(.3,.25,True),(.2,.16,False),(1.,.9,False)):
        p=deepcopy(original)
        for method,runs in p['gate'].items():
            for rows in runs.values():
                for row in rows:
                    row['rms_deg']=[0,0,(candidate_score if method=='M3' else base_score)*math.sqrt(7.625/6)]
        assert assess(p,gate,regression)['passed'] == passed
    p=deepcopy(original)
    for method in REFERENCES:
        for rows in p['gate'][method].values():
            for row in rows: row['rms_deg']=[0,0,0]
    result=assess(p,gate,regression)
    assert not result['passed'] and result['relative_improvement'] is None
    # A completed task failure stays in the score and denominator.
    p=deepcopy(original)
    for runs in p['gate'].values():
        for rows in runs.values(): rows[0]['success']=False
    result=assess(p,gate,regression)
    assert result['passed'] and math.isclose(result['method_scores']['M3'],.8)
    print('PASS: synthetic v2 effect, paired seeds, non-degradation, early termination and data integrity checks')
