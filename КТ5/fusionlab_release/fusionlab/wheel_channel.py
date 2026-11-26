"""One calibrated forward-wheel correction, usable with the lab graph or the F9 stand."""
from dataclasses import replace
import numpy as np


def scale_from_pairs(prediction, difference):
    p, d = np.asarray(prediction,float), np.asarray(difference,float)
    if p.ndim != 1 or d.shape != p.shape or not len(p) or not np.isfinite(np.r_[p,d]).all():
        return dict(a=0.,n_pairs=0,before=np.nan,after=np.nan,status='no_valid_calibration')
    norm2 = p @ p
    alpha = float(np.clip((p @ d) / norm2, 0., 1.)) if norm2 > 0 else 0.
    return dict(a=alpha,n_pairs=len(p),before=float(np.mean(d**2)),after=float(np.mean((d-alpha*p)**2)),
                status='calibrated' if norm2>0 else 'zero_prediction')


def observable(data):
    for key in ('truth_poses','truth_landmarks','true_gyro_bias'):
        if getattr(data,key,None) is not None:
            raise ValueError('remove reference fields before inference')


def differences(data, visual_kind):
    """Match both ends of a unit interval. Each sensor contributes at most one observation."""
    observable(data)
    by_sensor={'wheel':{},visual_kind:{}}
    for factor in data.factors:
        if factor.kind not in by_sensor:
            continue
        if factor.j != factor.i + 1:
            raise ValueError('wheel and visual factors must span one interval')
        key=(factor.i,factor.j)
        readings=by_sensor[factor.kind]
        if key in readings:
            raise ValueError('duplicate sensor observation on one interval')
        readings[key]=float(factor.z[0])
    wheel,visual=by_sensor['wheel'],by_sensor[visual_kind]
    common=sorted(wheel.keys() & visual.keys())
    return np.array([i for i,_ in common],int),np.array([wheel[k]-visual[k] for k in common],float)


def corrected(data, nposes, predicted, *, a, sigma):
    observable(data)
    p=np.asarray(predicted,float)
    if p.shape!=(nposes-1,) or not np.isfinite(p).all():
        raise ValueError('one finite forward residual per interval is required')
    if not (np.isfinite(a) and 0<=a<=1 and np.isfinite(sigma) and sigma>0):
        raise ValueError('invalid calibration or wheel uncertainty')
    correction=a*np.clip(p,-.55,.55)
    factors=[]
    for f in data.factors:
        if f.kind == 'wheel':
            if not (0<=f.i<nposes-1 and f.j==f.i+1):
                raise ValueError('invalid wheel interval')
            z,s=np.array(f.z,dtype=float,copy=True),np.array(f.sigma,dtype=float,copy=True)
            z[0]-=correction[f.i]
            s[0]=sigma
            f=replace(f,z=tuple(z) if isinstance(f.z,tuple) else z,sigma=tuple(s) if isinstance(f.sigma,tuple) else s)
        factors.append(f)
    return replace(data,factors=tuple(factors) if isinstance(data.factors,tuple) else factors)


def correct_graph(graph, prediction, *, a=0., sigma=.15):
    """Public adapter for the KT3 slam_lab.Graph. Leave all other observations intact."""
    return corrected(graph,graph.poses,prediction,a=a,sigma=sigma)
