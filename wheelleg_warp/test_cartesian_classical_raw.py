"""CPU counterexamples and schema fixtures; never receive a real584 queue."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import review_cartesian_classical_raw as r

YAW=dict(kp=.4,kd=.3,roll_gain=0,arm=1)


class RawTests(unittest.TestCase):
    def test_formula_and_yaw(self):
        x=np.zeros((1,481),np.float32);x[0,351]=.2;x[0,360]=.7;x[0,389]=.01
        a,d=r.classic(x,1,YAW);np.testing.assert_allclose(a[0,:2],[0,-1],rtol=0,atol=1e-7)
        expected=np.float32(.8*-np.arctan2(float(x[0,389]),float(x[0,360])*2.3/.8)/.3)
        self.assertEqual(a[0,2],expected);self.assertAlmostEqual(d['DeltaF'][0],3.4335)
        a2,_=r.classic(x,2,YAW);self.assertLessEqual(abs(a2).max(),1)

    def test_mirror_unused_fields(self):
        x=np.zeros((1,481),np.float32);x[0,[351,353,354,356,360,389]]=[.02,.01,.03,.04,.7,.015]
        a,_=r.classic(x,.5,YAW);mirror=x.copy();mirror[0,[351,353,354,356,389]]*=-1
        b,_=r.classic(mirror,.5,YAW);np.testing.assert_allclose(b,-a,rtol=0,atol=1e-7)
        other=x.copy();other[:,:351]=999;other[:,390:]=123
        np.testing.assert_array_equal(r.classic(other,.5,YAW)[0],a)

    def test_invalid_formula(self):
        for x,g in ((np.zeros((1,480)),0),(np.full((1,481),np.nan),1),(np.zeros((1,481)),True),(np.zeros((1,481)),3)):
            with self.assertRaises((AssertionError,ValueError)):r.classic(x,g,YAW)

    def test_mapping_and_counterexamples(self):
        actor=np.zeros((1,968),np.float32);c=np.zeros((1,38));g=np.zeros((1,62));p=np.zeros((1,30));c[:,0]=g[:,0]=p[:,0]=1
        c[:,13:15]=1;g[:,26:28]=1;c[:,27:33]=r.frozen.limits(np.zeros((1,6)));reference=r.frozen.geometry(np.zeros((1,2,2)))[0].reshape(4);c[:,34:38]=reference
        full=dict(pre=np.zeros((1,39)),trace=np.zeros((1,112)),ids=np.array([7,10,12,15,6,9,11,14,8,13]+[0]*9));full['trace'][:,53]=1
        diag=r.classic(actor[:,:481],0,YAW)[1]
        r.cart_case(actor,c,g,p,full,reference,0,0,YAW,diag)
        for fault in ('geometry','wheel_speed','withdraw','request','force'):
            cc,gg,pp,ff,dd=copy.deepcopy((c,g,p,full,diag))
            if fault=='geometry':cc[0,34]+=.001
            if fault=='wheel_speed':ff['pre'][0,17+8]=1000
            if fault=='withdraw':pp[0,4]=1
            if fault=='request':pp[0,5]=.1
            if fault=='force':dd['DeltaF'][0]=1
            with self.assertRaises(AssertionError):r.cart_case(actor,cc,gg,pp,ff,reference,0,0,YAW,dd)

    def test_header_missing_shape_hash(self):
        model={'x':np.zeros(1)};meta=dict(model_schema='classical_actual_model_v1',array_header={'x':dict(shape=[1],dtype='float64',sha256='wrong')})
        with self.assertRaises(AssertionError):r.model_view(model,meta,0,r.ROOT/'wheelleg_ppo/xml/wheelleg.xml')
        with self.assertRaises(AssertionError):r.finite(np.zeros((2,3)),(2,4))
        with self.assertRaises(AssertionError):r.finite(np.array([None],object))

    def test_missing_complete_and_false_admission(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)
            with self.assertRaises(FileNotFoundError):r.review(path,out=path)
            (path/'completion.json').write_text(json.dumps(dict(verified=False,episodes=584,records=[{}]*68)))
            with self.assertRaises(AssertionError):r.review(path,out=path)
            with self.assertRaises(ValueError):r.file_under(path,'../escape')

    def test_actual_delivery_container_and_partial_preservation(self):
        # Actual new producer file container -> independent history receiver, synthetic data only.
        from test_cartesian_classical_history_receipt import fixture
        from cartesian_classical_history_receipt import review_case
        from run_cartesian_classical_baseline import Actor
        for delay in (0,20):
            args=fixture(delay);delivery=args['delivery'];actor=Actor([args['case']],dict(gamma=0),YAW)
            actor.delivery[0]=[{n:a[k].copy() for n,a in delivery.items()} for k in range(2)]
            actor.delivery[0].append(dict(prewait_public38=np.ones(38),prewait_count=np.array(66)))
            with TemporaryDirectory() as folder:
                directory=Path(folder);errors=[];actor.save(directory,errors);self.assertEqual(errors,[])
                saved=r.arrays(directory/'delivery_0.npz');self.assertEqual(saved['returned_raw481'].shape,(2,481))
                self.assertTrue((directory/'delivery_partial_0_2_prewait_public38.npz').exists())
                args['delivery']=saved;self.assertTrue(review_case(**args)['verified'])


if __name__=='__main__':unittest.main(verbosity=2)
