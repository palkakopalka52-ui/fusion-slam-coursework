import unittest
from dataclasses import replace
import numpy as np
from slam_lab import COURSE_HUBER, Problem, camera_delta_to_base, compose, relative, solve, validate, wrap
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

    def test_pnp_in_optical_axes(self):
        # Optical Z points forward, X right, Y down. A lever arm also moves during yaw.
        mounting = np.array([[0,0,1,.21],[-1,0,0,.03],[0,-1,0,.14],[0,0,0,1.]])
        a = .4
        body = np.array([[np.cos(a),-np.sin(a),0,.35],[np.sin(a),np.cos(a),0,-.08],
                         [0,0,1,0],[0,0,0,1]])
        T21 = np.linalg.inv(mounting) @ np.linalg.inv(body) @ mounting
        np.testing.assert_allclose(camera_delta_to_base(T21,mounting),[.35,-.08,a],atol=1e-12)
        with self.assertRaises(ValueError):
            camera_delta_to_base(T21, np.diag([-1,1,1,1]))

    def test_whitening_and_course_scope(self):
        from slam_lab import Factor, Graph
        factors = (Factor('wheel',0,1,(0,0,0),(.5,1,1)),Factor('vo',0,1,(0,0,0),(.5,1,1)))
        pb = Problem(Graph(2,0,factors), COURSE_HUBER)
        r,_ = pb.linearize(np.array([2.,0,0,0]))
        np.testing.assert_allclose(r,[4,0,0,4,0,0])
        cost,weights = pb.loss(r,2.)
        np.testing.assert_allclose(weights,[1,.5])
        self.assertAlmostEqual(cost,14.)

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
