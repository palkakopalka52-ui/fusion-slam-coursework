import tempfile
import unittest
from pathlib import Path
import numpy as np
from study_neuro import DampingAdvisor, episode, solve, TRAIN, VALID, TEST
from slam_lab import Problem


class AdvisorChecks(unittest.TestCase):
    def test_feature_input_and_reset(self):
        class Probe:
            def reset(self):
                self.seen=[]
            def propose(self, values):
                assert values.shape==(6,) and np.isfinite(values).all()
                self.seen.append(values)
                return .002
        graph,_=episode(410)
        probe=Probe()
        first=solve(graph,policy=probe)
        n=len(probe.seen)
        again=solve(graph,policy=probe)
        self.assertEqual(len(probe.seen),n)
        np.testing.assert_array_equal(first.x,again.x)
        self.assertTrue(np.all(np.diff(first.history)<0))

    def test_contraction_and_roundtrip(self):
        net=DampingAdvisor(9,recurrent=True,cap=.75)
        x=np.linspace(-1.,1.,6)
        a,b=np.ones(9),-np.ones(9)
        diff=np.max(np.abs(net.advance(x,a)-net.advance(x,b)))
        self.assertLessEqual(diff,np.max(abs(net.params['u']))*np.max(abs(a-b)))
        net.reset();ref=[net.propose(x),net.propose(x)]
        with tempfile.TemporaryDirectory() as d:
            ckpt=Path(d)/'test.npz';net.save(ckpt)
            other=DampingAdvisor.load(ckpt)
            np.testing.assert_allclose(ref,[other.propose(x),other.propose(x)],atol=1e-14)
        self.assertTrue(all(1e-5<=v<=10 for v in ref))

    def test_invalid_proposal_and_guard(self):
        class Broken:
            def reset(self): pass
            def propose(self, x): return np.nan
        g,_=episode(412)
        sol=solve(g,policy=Broken())
        self.assertTrue(np.all(np.diff(sol.history)<0))
        g,_=episode(1620,'false_loop')
        pb=Problem(g)
        x=pb.initial()+np.random.default_rng(2).normal(0,3,pb.dim)
        refused=solve(g,policy='fixed',initial_lambda=1e-7,x0=x,budget=1)
        self.assertEqual((refused.status,refused.rejects,refused.solves),('budget',1,1))
        np.testing.assert_array_equal(refused.x,x)

    def test_route_split(self):
        self.assertFalse(set(TRAIN)&set(VALID) or set(TRAIN)&set(TEST) or set(VALID)&set(TEST))


if __name__=='__main__':
    unittest.main()
