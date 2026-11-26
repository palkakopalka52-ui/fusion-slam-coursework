import unittest
from dataclasses import replace
import numpy as np
from slam_lab import Problem, compose, relative, solve, validate, wrap
from slam_lab.bench import episode


class GraphChecks(unittest.TestCase):
    def setUp(self):
        self.graph, self.truth = episode(194)

    def test_frames(self):
        a,b = np.array([3.,-.4,1.8]),np.array([.6,.01,2.1])
        np.testing.assert_allclose(relative(a,compose(a,b)),b,atol=1e-12)
        self.assertAlmostEqual(float(wrap(np.deg2rad(179)-np.deg2rad(-179))),np.deg2rad(-2))

    def test_block_jacobians(self):
        p = Problem(self.graph)
        x = p.initial()
        _, analytic = p.linearize(x)
        h = 1e-6
        numeric = np.stack([(p.linearize(x+h*d,False)[0]-p.linearize(x-h*d,False)[0])/(2*h)
                            for d in np.eye(len(x))],axis=1)
        np.testing.assert_allclose(analytic,numeric,atol=2e-6,rtol=2e-6)

    def test_bad_measurements(self):
        f = self.graph.factors[0]
        for bad in [replace(f,i=-1),replace(f,sigma=(1.,0.,1.)),replace(f,z=(np.nan,1.,2.)),
                    replace(f,j=self.graph.poses),replace(f,z=(2.,)),replace(f,kind='gyro',z=(0.,),sigma=(.1,),dt=0)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate(replace(self.graph,factors=(bad,)))
        p=Problem(self.graph)
        x=p.initial()
        obs=next(f for f in self.graph.factors if f.kind=='beacon' and f.i==0)
        x[3*(self.graph.poses-1)+2*obs.j:3*(self.graph.poses-1)+2*obs.j+2]=0
        with self.assertRaisesRegex(ValueError,'zero distance'):
            p.linearize(x)

    def test_anchor_guard_and_bias(self):
        e=solve(self.graph)
        np.testing.assert_equal(e.poses[0],[0.,0.,0.])
        self.assertTrue(np.all(np.diff(e.history)<0))
        self.assertLess(abs(e.bias-self.truth.bias),.045)
        self.assertEqual(len(e.history)-1+e.rejects,e.solves)

    def test_linear_contraction(self):
        B = np.array([[1.,0.],[-1.,1.],[0.,1.]])
        H = B.T@B
        eig = np.linalg.eigvalsh(H)
        alpha = 2./sum(eig)
        q=max(abs(1-alpha*eig))
        x = np.zeros(2); target=np.array([.3,1.1])
        for _ in range(12):
            new=x-alpha*(H@(x-target))
            self.assertLessEqual(np.linalg.norm(new-target),q*np.linalg.norm(x-target)+1e-12)
            x=new


if __name__=='__main__':
    unittest.main()
