"""FusionLab notebook interface for the reusable, calibrated wheel channel."""
from dataclasses import replace
import numpy as np
from . import slam_core
from .wheel_channel import corrected, differences, observable, scale_from_pairs


def observations(dataset):
    return replace(dataset,truth_poses=None,truth_landmarks=None,true_gyro_bias=None)


def paired_diff(raw):
    return differences(raw,'visual')


def calibrate_scale(pred,y):
    return scale_from_pairs(pred,y)


def correct_factors(raw, correction, a=0., sigma=.15):
    return corrected(raw,raw.nposes,correction,a=a,sigma=sigma)


def estimate_slam(raw, correction, a=0., sigma=.15, max_iter=25, **kwargs):
    observable(raw)
    x0=slam_core.initial_state(raw)
    revised=correct_factors(raw,correction,a=a,sigma=sigma)
    fit=slam_core.solve(revised,x0=x0,max_iter=max_iter,robust=True,**kwargs)
    if np.any(np.diff(fit['history'])>1e-9):
        raise ArithmeticError('the accepted objective increased')
    return fit


def evaluate(sol, reference):
    return slam_core.metrics(sol,reference)
