"""Synthetic observations, kept apart from the truth used for scoring."""
from dataclasses import dataclass
import numpy as np
from .graph import Factor, Graph, relative, rotate, wrap


@dataclass(frozen=True)
class Reference:
    poses: np.ndarray
    landmarks: np.ndarray
    bias: float


def episode(seed, condition='outliers', length=21):
    if condition not in ('clean','outliers','dropout','false_loop','transfer'):
        raise ValueError('unknown episode condition')
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, length)
    a, b = rng.uniform(1.7,2.4), rng.uniform(1.,1.5)
    if condition == 'transfer':
        b *= .55
    xy = np.column_stack((a*np.sin(t), b*(1-np.cos(t))))
    yaw = np.arctan2(b*np.sin(t), a*np.cos(t))
    poses = np.column_stack((xy,yaw))
    marks = np.array([[-3.4,.4],[-1.3,4.2],[1.8,4.5],[3.6,.7],[.3,-1.7]]) + rng.normal(0,.15,(5,2))
    bias = rng.uniform(.025,.065)
    factors = []
    def add(kind,i,j,z,sigma):
        factors.append(Factor(kind,i,j,tuple(z),tuple(sigma)))
    for i,p in enumerate(poses):
        if i < length-1:
            rel = relative(p, poses[i+1])
            wheel = rel + rng.normal(0,[.040,.018,.021]) + [.045 + .075*(i%9 < 3),0,.012]
            add('wheel',i,i+1,wheel,[.09,.04,.05])
            visual = rel + rng.normal(0,[.035,.035,.025])
            if rng.random() > .13 and not (condition == 'dropout' and length//4 <= i < 3*length//4):
                add('vo',i,i+1,visual,[.035,.035,.025])
            add('gyro',i,i+1,[wrap(rel[2]+bias+rng.normal(0,.012))],[.012])
        for j in (i % 5,(i+2)%5):
            d = marks[j]-p[:2]
            measure = [np.linalg.norm(d)+rng.normal(0,.065), wrap(np.arctan2(d[1],d[0])-p[2]+rng.normal(0,.028))]
            if condition != 'clean' and (3*i+j)%17 == 0:
                measure[0] += 1.5
                measure[1] = wrap(measure[1]-.43)
            add('beacon',i,j,measure,[.065,.028])
    loop = relative(poses[0],poses[-1]) + rng.normal(0,[.045,.045,.02])
    if condition == 'false_loop':
        loop += [1.6,-.9,.45]
    add('loop',0,length-1,loop,[.055,.055,.025])
    return Graph(length,len(marks),tuple(factors)), Reference(poses,marks,bias)


def score(poses, truth, landmarks=None):
    xy_error = poses[:,:2]-truth.poses[:,:2]
    est = rotate(np.diff(poses[:,:2],axis=0), -poses[:-1,2])
    ref = rotate(np.diff(truth.poses[:,:2],axis=0), -truth.poses[:-1,2])
    def rmse(vec):
        return float(np.sqrt(np.mean(np.sum(vec**2,axis=1))))
    return {'ate_m':rmse(xy_error), 'rpe_m':rmse(est-ref),
            'yaw_rad':float(np.sqrt(np.mean(wrap(poses[:,2]-truth.poses[:,2])**2))),
            'map_m':rmse(landmarks-truth.landmarks) if landmarks is not None else float('nan')}
