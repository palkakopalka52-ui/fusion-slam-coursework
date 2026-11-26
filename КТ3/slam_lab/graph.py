"""Planar factor graph. Coordinates are metres, angles radians, gyro bias rad/s."""
from dataclasses import dataclass
import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rotate(v, angle):
    x, y = np.moveaxis(np.asarray(v), -1, 0)
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c*x - s*y, s*x + c*y), axis=-1)


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.r_[rotate(b[:2] - a[:2], -a[2]), wrap(b[2] - a[2])]


def compose(a, step):
    a, step = np.asarray(a), np.asarray(step)
    return np.r_[a[:2] + rotate(step[:2], a[2]), wrap(a[2] + step[2])]


def camera_delta_to_base(T21, T_BC):
    """Project a PnP transform Cj <- Ci to a planar motion Bi <- Bj (x, y, yaw)."""
    def rigid(value):
        M = np.asarray(value, dtype=float)
        if (M.shape != (4, 4) or not np.isfinite(M).all()
                or not np.allclose(M[3], [0, 0, 0, 1], atol=1e-7, rtol=0)):
            raise ValueError('expected a finite homogeneous transform')
        R = M[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or np.linalg.det(R) <= 0:
            raise ValueError('expected a proper rotation')
        return M

    camera, mounting = rigid(T21), rigid(T_BC)
    step = mounting @ np.linalg.inv(camera) @ np.linalg.inv(mounting)
    if np.hypot(step[0, 0], step[1, 0]) < 1e-8:
        raise ValueError('body heading has no horizontal projection')
    return np.array([step[0, 3], step[1, 3], np.arctan2(step[1, 0], step[0, 0])])


@dataclass(frozen=True)
class Factor:
    kind: str
    i: int
    j: int
    z: tuple
    sigma: tuple
    dt: float = 1.


@dataclass(frozen=True)
class Graph:
    poses: int
    landmarks: int
    factors: tuple


SIZES = {'wheel': 3, 'vo': 3, 'loop': 3, 'gyro': 1, 'beacon': 2}
COURSE_HUBER = ('vo', 'loop', 'beacon')


def validate(graph):
    if not isinstance(graph.poses, (int, np.integer)) or graph.poses < 2:
        raise ValueError('poses must be an integer >= 2')
    if not isinstance(graph.landmarks, (int, np.integer)) or graph.landmarks < 0:
        raise ValueError('landmark count must be a nonnegative integer')
    for k, f in enumerate(graph.factors):
        limit = graph.landmarks if f.kind == 'beacon' else graph.poses
        indices = all(isinstance(v, (int, np.integer)) for v in (f.i, f.j))
        dim = SIZES.get(f.kind, 0)
        if not dim or not indices or not (0 <= f.i < graph.poses and 0 <= f.j < limit):
            raise ValueError(f'factor {k}: invalid kind or index')
        if f.kind != 'beacon' and f.i == f.j:
            raise ValueError(f'factor {k}: identical pose indices')
        z, sigma = np.asarray(f.z, float), np.asarray(f.sigma, float)
        if z.shape != (dim,) or sigma.shape != (dim,) or not np.isfinite(np.r_[z, sigma]).all() or np.any(sigma <= 0):
            raise ValueError(f'factor {k}: invalid value, dimension or sigma')
        if (f.kind == 'beacon' and z[0] <= 0) or (f.kind == 'gyro' and (not np.isfinite(f.dt) or f.dt <= 0)):
            raise ValueError(f'factor {k}: invalid range or gyro interval')


def wheel_path(graph):
    steps = {f.i: f.z for f in graph.factors if f.kind == 'wheel' and f.j == f.i + 1}
    if len(steps) != graph.poses-1:
        raise ValueError('initialization requires a continuous wheel chain')
    p = np.zeros((graph.poses, 3))
    for i in range(graph.poses-1):
        p[i+1] = compose(p[i], steps[i])
    return p


class Problem:
    """Compile the measurement graph once. No reference poses or landmark map are accepted."""
    def __init__(self, graph, robust_kinds=None):
        validate(graph)
        robust_kinds = set(SIZES if robust_kinds is None else robust_kinds)
        if robust_kinds.difference(SIZES):
            raise ValueError('unknown robust factor kind')
        self.graph = graph
        self.loss_evaluations = 0
        self.dim = 3*(graph.poses-1) + 2*graph.landmarks + 1
        self.groups = []
        self.blocks = []
        robust_blocks = []
        start = 0
        for kind, width in SIZES.items():
            items = [f for f in graph.factors if f.kind == kind]
            if not items:
                continue
            end = start + len(items)*width
            rows = np.arange(start, end).reshape(-1, width)
            self.groups.append((kind, rows, np.array([f.i for f in items]), np.array([f.j for f in items]),
                                np.array([f.z for f in items]), np.array([f.sigma for f in items]),
                                np.array([f.dt for f in items])))
            self.blocks.extend([width] * len(items))
            robust_blocks.extend([kind in robust_kinds] * len(items))
            start = end
        self.rows = start
        self.starts = np.cumsum([0]+self.blocks[:-1])
        self.robust_blocks = np.asarray(robust_blocks, dtype=bool)

    def initial(self):
        poses = wheel_path(self.graph)
        samples = [[] for _ in range(self.graph.landmarks)]
        for f in self.graph.factors:
            if f.kind == 'beacon':
                d, a = f.z
                samples[f.j].append(poses[f.i,:2] + rotate([d, 0.], a + poses[f.i,2]))
        if any(not s for s in samples):
            raise ValueError('unobserved landmark')
        landmarks = np.array([np.median(s, axis=0) for s in samples])
        return np.r_[poses[1:].ravel(), landmarks.ravel(), 0.]

    def unpack(self, x):
        split = 3*(self.graph.poses-1)
        return np.r_[np.zeros(3), x[:split]].reshape(-1,3), x[split:-1].reshape(-1,2), x[-1]

    def linearize(self, x, jac=True):
        poses, marks, bias = self.unpack(x)
        residual = np.empty(self.rows)
        # Work with absolute pose column numbers, then drop the three anchored columns.
        J = np.zeros((self.rows, self.dim+3)) if jac else None
        for kind, rows, i, j, z, sigma, dt in self.groups:
            ri = rows[:,0]
            ca, sa = np.cos(poses[i,2]), np.sin(poses[i,2])
            if kind in ('vo','wheel','loop'):
                body = rotate(poses[j,:2]-poses[i,:2], -poses[i,2])
                err = np.column_stack((body-z[:,:2], wrap(poses[j,2]-poses[i,2]-z[:,2])))
                if jac:
                    rj, rk = rows[:,1], rows[:,2]
                    J[ri,3*j], J[ri,3*j+1], J[rj,3*j], J[rj,3*j+1] = ca, sa, -sa, ca
                    J[ri,3*i], J[ri,3*i+1], J[rj,3*i], J[rj,3*i+1] = -ca, -sa, sa, -ca
                    J[ri,3*i+2], J[rj,3*i+2] = body[:,1], -body[:,0]
                    J[rk,3*i+2], J[rk,3*j+2] = -1., 1.
            elif kind == 'gyro':
                err = wrap(poses[j,2]-poses[i,2]+bias*dt-z[:,0])[:,None]
                if jac:
                    J[ri,3*i+2], J[ri,3*j+2], J[ri,-1] = -1., 1., dt
            else:
                dx, dy = (marks[j]-poses[i,:2]).T
                dist = np.hypot(dx,dy)
                if np.any(dist < 1e-8):
                    raise ValueError('undefined bearing at zero distance')
                err = np.column_stack((dist-z[:,0], wrap(np.arctan2(dy,dx)-poses[i,2]-z[:,1])))
                if jac:
                    col = 3*self.graph.poses+2*j
                    rb = rows[:,1]
                    J[ri,col], J[ri,col+1] = dx/dist, dy/dist
                    J[rb,col], J[rb,col+1] = -dy/dist**2, dx/dist**2
                    J[ri,3*i], J[ri,3*i+1] = -dx/dist, -dy/dist
                    J[rb,3*i], J[rb,3*i+1], J[rb,3*i+2] = dy/dist**2, -dx/dist**2, -1.
            residual[rows] = err/sigma
            if jac:
                J[rows] /= sigma[:,:,None]
        return residual, (J[:,3:] if jac else None)

    def loss(self, r, delta):
        # Huber is applied to the norm of a whole normalized sensor block.
        squared = np.add.reduceat(r*r, self.starts)
        self.loss_evaluations += 1
        if delta is None:
            return .5 * squared.sum(), np.ones_like(squared)
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError('Huber threshold must be positive')
        lengths = np.sqrt(squared)
        tails = self.robust_blocks & (lengths > delta)
        weights = np.ones_like(lengths)
        weights[tails] = delta / lengths[tails]
        cost = np.where(tails, delta*(lengths-.5*delta), .5*squared)
        return cost.sum(), weights

    def objective(self, x, delta):
        return self.loss(self.linearize(x, jac=False)[0], delta)[0]
