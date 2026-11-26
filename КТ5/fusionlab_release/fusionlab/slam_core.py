"""Small, transparent batch 2-D factor-graph SLAM core for a Jupyter workshop.

No simulation engine, ROS, or learned estimator.  The measurement generator
below is synthetic; ground truth is used ONLY for verification/benchmarks.
Conventions: world poses (x [m], y [m], yaw [rad]); body x points forwards,
body y to the left; landmark bearings measured counter-clockwise from body x.
The first robot pose is fixed to (0, 0, 0) to remove planar gauge freedom.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class Factor:
    kind: str             # wheel, visual, loop, gyro, landmark, bias_prior
    i: int                # pose index (or 0 for bias prior)
    j: int                # pose index or landmark index
    z: np.ndarray         # observed quantities; no ground truth here
    sigma: np.ndarray     # per-component standard deviations in native units
    dt: float = 1.0
    tag: str = ""         # diagnostic only: never read by the solver


@dataclass
class Dataset:
    """Sensor factors may be used without ground truth in a future data adapter."""
    factors: List[Factor]
    nposes: int
    nlandmarks: int
    seed: int = -1
    true_gyro_bias: float | None = None
    truth_poses: np.ndarray | None = None
    truth_landmarks: np.ndarray | None = None


def relative_pose(pi, pj):
    """Pose of j in frame i, as [body-frame dx, dy, wrapped dyaw]."""
    dx, dy = pj[:2] - pi[:2]
    c, s = np.cos(pi[2]), np.sin(pi[2])
    return np.array([c*dx + s*dy, -s*dx + c*dy, wrap(pj[2]-pi[2])])


def integrate_relative(p, z):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([p[0] + c*z[0] - s*z[1],
                     p[1] + s*z[0] + c*z[1],
                     wrap(p[2] + z[2])])


def range_bearing(pose, landmark):
    d = landmark - pose[:2]
    r = np.linalg.norm(d)
    if r < 1.e-10:
        raise ValueError("Landmark too close to robot for bearing")
    return np.array([r, wrap(np.arctan2(d[1], d[0]) - pose[2])])


def make_dataset(seed=23, nposes=18, false_loop=True, outliers=True):
    """Generate independent sensor noises; not a physics-accurate simulator.

    Circular unicycle-like route, returning to the start. The generator uses
    exactly known landmark IDs; data association is out of the core scope.
    """
    if nposes < 8:
        raise ValueError("Need at least 8 poses")
    rng = np.random.default_rng(seed)
    ts = np.linspace(0., 2.*np.pi, nposes)
    radius = 2.2
    poses = np.stack([radius*np.sin(ts), radius*(1.-np.cos(ts)), wrap(ts)], axis=1)
    landmarks = np.array([[-1.2, 1.0], [1.1, -0.6], [3.7, 1.2],
                          [3.2, 3.9], [1.1, 5.1], [-1.3, 3.8]], float)
    factors = []
    gyro_bias = 0.027  # radians/second, constant; here dt=1 second

    for i in range(nposes-1):
        z = relative_pose(poses[i], poses[i+1])
        wheel_sig = np.array([.075, .20, .060])
        wheel = z.copy() + rng.normal(0, wheel_sig)
        wheel[1] = rng.normal(0, wheel_sig[1])  # nonholonomic dy~0, soft assumption
        wheel[2] = wrap(wheel[2])
        factors.append(Factor("wheel", i, i+1, wheel, wheel_sig))
        if rng.random() >= .25:
            sig = np.array([.040, .040, .027])
            visual = z.copy() + rng.normal(0, sig)
            visual[2] = wrap(visual[2])
            factors.append(Factor("visual", i, i+1, visual, sig))
        yaw = wrap(z[2] + gyro_bias + rng.normal(0, .025))
        factors.append(Factor("gyro", i, i+1, np.array([yaw]),
                              np.array([.025]), dt=1.0))

    zloop = relative_pose(poses[0], poses[-1])
    factors.append(Factor("loop", 0, nposes-1,
                          zloop + rng.normal(0, [.07, .07, .040]),
                          np.array([.07, .07, .040]), tag="verified_loop"))
    # The false image match below is an intentionally misassociated loop closure.
    # Robust estimation can help; real pipelines must verify loop candidates.
    if false_loop:
        factors.append(Factor("loop", 0, nposes//2,
                              rng.normal(0, [.05, .05, .03]),
                              np.array([.12, .12, .075]), tag="false_loop"))

    for i, pose in enumerate(poses):
        for j, landmark in enumerate(landmarks):
            z = range_bearing(pose, landmark)
            if z[0] > 4.3 or rng.random() < .25:
                continue
            sig = np.array([.10, .045])
            obs = z + rng.normal(0, sig)
            bad = outliers and rng.random() < .09
            if bad:
                obs += np.array([rng.choice([-1., 1.])*rng.uniform(.8,1.8),
                                 rng.choice([-1., 1.])*rng.uniform(.15,.55)])
            obs[0] = max(.05, obs[0])
            obs[1] = wrap(obs[1])
            factors.append(Factor("landmark", i, j, obs, sig,
                                  tag="injected_outlier" if bad else ""))
    factors.append(Factor("bias_prior", 0, 0, np.array([0.]), np.array([.12])))
    return Dataset(factors=factors, nposes=len(poses),
                   nlandmarks=len(landmarks), seed=int(seed),
                   true_gyro_bias=gyro_bias,
                   truth_poses=poses, truth_landmarks=landmarks)


def wheel_dead_reckoning(data):
    out = np.zeros((data.nposes,3))
    for f in data.factors:
        if f.kind == "wheel":
            out[f.j] = integrate_relative(out[f.i], f.z)
    return out


def initial_state(data):
    """Initialize poses using wheel odometry and landmarks using observations.

    Do not access hidden truth fields except for array lengths.
    """
    poses = wheel_dead_reckoning(data)
    landmark_candidates = [[] for _ in range(data.nlandmarks)]
    for f in data.factors:
        if f.kind == "landmark":
            p = poses[f.i]
            direction = p[2] + f.z[1]
            point = p[:2] + f.z[0]*np.array([np.cos(direction),np.sin(direction)])
            landmark_candidates[f.j].append(point)
    landmarks = []
    for j, items in enumerate(landmark_candidates):
        if not items:
            raise ValueError(f"Landmark {j} was not observed")
        landmarks.append(np.median(np.asarray(items), axis=0))
    return pack_state(poses, np.array(landmarks), bias=0.)


def pack_state(poses, landmarks, bias=0.0):
    poses = np.asarray(poses, float)
    landmarks = np.asarray(landmarks, float)
    if poses.shape[1] != 3 or landmarks.shape[1] != 2:
        raise ValueError("Wrong pose/landmark dimensions")
    if not np.allclose(poses[0], 0., atol=1e-12):
        raise ValueError("First pose must be fixed at the origin")
    return np.r_[poses[1:].ravel(), landmarks.ravel(), float(bias)]


def unpack_state(x, nposes, nlandmarks):
    x = np.asarray(x, float)
    count = 3*(nposes-1)
    end = count + 2*nlandmarks
    if x.size != end+1:
        raise ValueError("Wrong state vector size")
    poses = np.vstack((np.zeros(3), x[:count].reshape(-1,3)))
    landmarks = x[count:end].reshape(-1,2)
    return poses, landmarks, float(x[-1])


def _jac_relative(pi, pj):
    dx,dy=pj[:2]-pi[:2]
    c,s=np.cos(pi[2]),np.sin(pi[2])
    u,v=c*dx+s*dy,-s*dx+c*dy
    ji=np.array([[-c,-s, v],[s,-c,-u],[0.,0.,-1.]])
    jj=np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
    return ji,jj


def _jac_landmark(p, lm):
    dx, dy = lm - p[:2]
    q = dx*dx+dy*dy
    if q < 1e-12:
        raise ValueError("Landmark measurement singular at zero range")
    r = np.sqrt(q)
    jp = np.array([[-dx/r,-dy/r,0.],[dy/q,-dx/q,-1.]])
    jl = np.array([[dx/r,dy/r],[-dy/q,dx/q]])
    return jp, jl


def _slice_pose(i):
    return None if i == 0 else slice(3*(i-1),3*i)


def _slice_landmark(j, nposes):
    k=3*(nposes-1)+2*j
    return slice(k,k+2)


def residual_jacobian(x, data, use_visual=True, use_landmarks=True,
                      use_gyro=True, use_loop=True, jac=True):
    """Whitened residual, analytic Jacobian, and per-factor row blocks.

    Tightly related components of an observation share one Huber block.
    """
    poses,lms,bias=unpack_state(x,data.nposes,data.nlandmarks)
    nd=x.size
    residuals=[]; matrices=[]; blocks=[]; active=[]
    cursor=0
    for f in data.factors:
        if f.kind == "visual" and not use_visual:
            continue
        if f.kind == "loop" and (not use_visual or not use_loop):
            continue
        if f.kind == "landmark" and not use_landmarks:
            continue
        if f.kind == "gyro" and not use_gyro:
            continue
        if f.kind == "bias_prior" and not use_gyro:
            continue
        jrow = np.zeros((len(f.z),nd)) if jac else None
        if f.kind in ("wheel", "visual", "loop"):
            pred=relative_pose(poses[f.i],poses[f.j])
            r=pred-f.z
            r[2]=wrap(r[2])
            if jac:
                ji,jj=_jac_relative(poses[f.i],poses[f.j])
                if f.i>0: jrow[:,_slice_pose(f.i)]=ji
                if f.j>0: jrow[:,_slice_pose(f.j)]=jj
        elif f.kind == "landmark":
            pred=range_bearing(poses[f.i],lms[f.j])
            r=pred-f.z
            r[1]=wrap(r[1])
            if jac:
                ji,jl=_jac_landmark(poses[f.i],lms[f.j])
                if f.i>0: jrow[:,_slice_pose(f.i)]=ji
                jrow[:,_slice_landmark(f.j,data.nposes)]=jl
        elif f.kind == "gyro":
            r=np.array([wrap(poses[f.j,2]-poses[f.i,2]+bias*f.dt-f.z[0])])
            if jac:
                if f.i>0: jrow[0,_slice_pose(f.i).start+2]=-1.
                if f.j>0: jrow[0,_slice_pose(f.j).start+2]=1.
                jrow[0,-1]=f.dt
        elif f.kind == "bias_prior":
            r=np.array([bias-f.z[0]])
            if jac: jrow[0,-1]=1.
        else:
            raise ValueError("Unknown factor kind: "+f.kind)
        residuals.append(r/f.sigma)
        if jac: matrices.append(jrow/f.sigma[:,None])
        blocks.append(slice(cursor,cursor+len(r)))
        active.append(f)
        cursor+=len(r)
    rval=np.concatenate(residuals)
    return rval,(np.vstack(matrices) if jac else None),blocks,active


def objective(resid, blocks, factors, robust=True, threshold=2.5):
    val=0.
    for block,f in zip(blocks,factors):
        norm=np.linalg.norm(resid[block])
        if robust and f.kind in ("visual", "loop", "landmark") and norm>threshold:
            val+=threshold*(norm-.5*threshold)
        else:
            val+=.5*norm*norm
    return float(val)


def irls_weights(r,blocks,factors,robust=True,threshold=2.5):
    w=np.ones_like(r)
    if robust:
        for sl,f in zip(blocks,factors):
            if f.kind in ("visual", "loop", "landmark"):
                n=np.linalg.norm(r[sl])
                if n>threshold:
                    w[sl]=threshold/n
    return w


def solve(data, x0=None, robust=True, use_visual=True,
          use_landmarks=True, use_gyro=True, use_loop=True,
          max_iter=40, lam0=1.e-3, gtol=5.e-6):
    """Adaptive damped GN + block-Huber IRLS. This is a local optimizer.

    Initial pose anchored by parameterization. Returns status, objective history,
    state and estimated trajectory/map; never accesses ground truth positions.
    """
    x=initial_state(data) if x0 is None else np.asarray(x0,float).copy()
    args=dict(use_visual=use_visual,use_landmarks=use_landmarks,
              use_gyro=use_gyro,use_loop=use_loop)
    lam=float(lam0); hist=[]; accepted=0; status="max_iter"
    for iteration in range(max_iter):
        r,j,blocks,factors=residual_jacobian(x,data,**args)
        val=objective(r,blocks,factors,robust)
        hist.append(val)
        w=np.sqrt(irls_weights(r,blocks,factors,robust))
        jw=j*w[:,None]; rw=r*w
        h=jw.T@jw
        grad=jw.T@rw
        if np.linalg.norm(grad,np.inf)<gtol:
            status="gradient_small";break
        diag=np.maximum(h.diagonal(),1.)
        succeeded=False
        for trial in range(15):
            step=np.linalg.solve(h+lam*np.diag(diag),-grad)
            if not np.all(np.isfinite(step)):
                raise FloatingPointError("Nonfinite optimization step")
            if np.linalg.norm(step)<1.e-9*(1.+np.linalg.norm(x)):
                status="step_small";succeeded=True;break
            cand=x+step
            # Bearing is undefined if a tested landmark coincides with a pose.
            # Reject such a trial rather than allowing a singular Jacobian on
            # the next iteration; no invented direction is inserted.
            if use_landmarks:
                cp, cl, _ = unpack_state(cand, data.nposes, data.nlandmarks)
                if any(f.kind == "landmark" and
                       np.sum((cp[f.i,:2]-cl[f.j])**2) < 1.e-10
                       for f in data.factors):
                    lam=min(lam*7.,1.e12)
                    continue
            # All pose yaw angles live in S1; use local Euclidean coordinates.
            for idx in range(2,3*(data.nposes-1),3):
                cand[idx]=wrap(cand[idx])
            rr,_,bb,ff=residual_jacobian(cand,data,jac=False,**args)
            newval=objective(rr,bb,ff,robust)
            if np.isfinite(newval) and newval<val:
                x=cand; lam=max(lam/3.,1.e-9)
                accepted+=1; succeeded=True;break
            lam=min(lam*7.,1.e12)
        if status=="step_small":break
        if not succeeded:
            status="stalled";break
    r,j,blocks,factors=residual_jacobian(x,data,**args)
    hist.append(objective(r,blocks,factors,robust))
    poses,lms,bias=unpack_state(x,data.nposes,data.nlandmarks)
    return dict(x=x,poses=poses,landmarks=lms,bias=bias,
                objective=hist[-1],history=np.array(hist),
                status=status,steps=accepted,lambda_final=lam,
                active_factors=len(factors))


def metrics(estimate, data, include_map=True):
    if data.truth_poses is None or data.truth_landmarks is None:
        raise ValueError("Ground truth required for benchmarks; never for solving")
    p=estimate["poses"]; t=data.truth_poses
    if len(p)!=len(t): raise ValueError("Different trajectory lengths")
    e=np.linalg.norm(p[:,:2]-t[:,:2],axis=1)
    rpe=np.linalg.norm(np.diff(p[:,:2],axis=0)-np.diff(t[:,:2],axis=0),axis=1)
    yaw=wrap(p[:,2]-t[:,2])
    ans=dict(ate_rmse_m=float(np.sqrt(np.mean(e**2))),
             rpe_xy_rmse_m=float(np.sqrt(np.mean(rpe**2))),
             yaw_rmse_rad=float(np.sqrt(np.mean(yaw**2))),
             gyro_bias_error_rad_step=float(estimate["bias"]-data.true_gyro_bias))
    if include_map:
        le=np.linalg.norm(estimate["landmarks"]-data.truth_landmarks,axis=1)
        ans["landmark_rmse_m"]=float(np.sqrt(np.mean(le**2)))
    return ans


def visual_coverage(data):
    return sum(f.kind=="visual" for f in data.factors)/float(data.nposes-1)


def neumann_demo(A,b,num_iterations=25):
    """Fixed point for SPD normal equations; provable contraction in l2."""
    H=A.T@A
    rhs=A.T@b
    eigen=np.linalg.eigvalsh(H)
    if eigen[0]<=0: raise ValueError("Fix the gauge first: SPD required")
    step=1.0/eigen[-1]
    alpha=max(abs(1.-step*eigen[0]),abs(1.-step*eigen[-1]))
    x=np.zeros(H.shape[0]); it=[x.copy()]
    for _ in range(num_iterations):
        x=x-step*(H@x-rhs)
        it.append(x.copy())
    return np.asarray(it),alpha,np.linalg.solve(H,rhs)


def previous_vo_camera_to_body_delta(T_new_old_camera,T_body_camera):
    """Bridge from old notebook estimate_pair output T_Cj_Ci.

    Both homogeneous transforms must be in meters. A fixed, calibrated
    T_body_camera is essential; never assume optical and body axes coincide.
    Returns [dx,dy,dyaw] of body j w.r.t. body i in frame Bi.
    """
    camera=np.asarray(T_new_old_camera,float)
    bc=np.asarray(T_body_camera,float)
    if camera.shape!=(4,4) or bc.shape!=(4,4):
        raise ValueError("Expected two 4x4 homogeneous transforms")
    b_i_b_j=bc@np.linalg.inv(camera)@np.linalg.inv(bc)
    angle=np.arctan2(b_i_b_j[1,0],b_i_b_j[0,0])
    return np.array([b_i_b_j[0,3],b_i_b_j[1,3],wrap(angle)])
