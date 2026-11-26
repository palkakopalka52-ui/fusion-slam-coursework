"""Damped Gauss–Newton with one acceptance rule for all damping policies."""
from dataclasses import dataclass
import time
import numpy as np
from .graph import Problem

MIN_LAMBDA, MAX_LAMBDA = 1e-7, 1e5


@dataclass
class Iteration:
    r: np.ndarray
    H: np.ndarray
    g: np.ndarray
    diag: np.ndarray
    cost: float
    features: np.ndarray


def linear_step(problem, x, delta, previous_lambda, rho):
    r, J = problem.linearize(x)
    cost, weights = problem.loss(r, delta)
    weighted = J * np.repeat(weights, problem.blocks)[:,None]
    H, g = weighted.T @ J, weighted.T @ r
    diag = np.maximum(np.diag(H), 1e-8)
    features = np.array([np.log10(1+np.linalg.norm(g,np.inf)), np.log10(1+np.linalg.norm(r)),
                         np.log10(diag.max()/diag.min()), np.mean(weights < 1.),
                         np.log10(previous_lambda), np.clip(rho,-2.,2.)])
    return Iteration(r,H,g,diag,float(cost),features)


def trial(problem, x, state, damping, delta):
    A = state.H.copy()
    A.flat[::len(x)+1] += damping * state.diag
    try:
        dx = np.linalg.solve(A, -state.g)
        candidate = x + dx
        cost = problem.objective(candidate, delta)
        predicted = -state.g@dx - .5*dx@state.H@dx
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return x, np.inf, 0.
    ratio = (state.cost-cost)/predicted if predicted > 0 else 0.
    return candidate, float(cost), float(ratio)


@dataclass
class Estimate:
    x: np.ndarray
    poses: np.ndarray
    landmarks: np.ndarray
    bias: float
    history: list
    solves: int
    rejects: int
    status: str
    seconds: float
    advisor_calls: int = 0
    loss_evaluations: int = 0


def solve(graph, *, delta=2.5, policy='classical', initial_lambda=.003, iterations=24,
          budget=48, x0=None, robust_kinds=None):
    """policy is 'classical', 'fixed', or an object implementing reset() and propose(features)."""
    if not MIN_LAMBDA <= initial_lambda <= MAX_LAMBDA or budget < 1 or iterations < 1:
        raise ValueError('invalid damping or solve budget')
    if isinstance(policy,str) and policy not in ('classical','fixed'):
        raise ValueError('unknown damping policy')
    started = time.perf_counter()
    pb = Problem(graph, robust_kinds)
    x = pb.initial() if x0 is None else np.asarray(x0,dtype=float).copy()
    if x.shape != (pb.dim,) or not np.isfinite(x).all():
        raise ValueError('invalid initial state')
    if not isinstance(policy,str):
        policy.reset()
    previous = current = float(initial_lambda)
    rho = 1.
    history = [float(pb.objective(x, delta))]
    solves = rejects = advisor_calls = 0
    status = 'iteration_limit'
    for _ in range(iterations):
        state = linear_step(pb, x, delta, previous, rho)
        if np.linalg.norm(state.g, np.inf) < 1e-7:
            status = 'small_gradient'
            break
        if isinstance(policy,str):
            damping = initial_lambda if policy == 'fixed' else current
        else:
            advisor_calls += 1
            proposal = float(policy.propose(state.features.copy()))
            damping = np.clip(proposal, MIN_LAMBDA, MAX_LAMBDA) if np.isfinite(proposal) and proposal > 0 else current
        accepted = False
        for _attempt in range(8):
            candidate, cost, new_rho = trial(pb,x,state,damping,delta)
            solves += 1
            if np.isfinite(cost) and cost < state.cost:
                x = candidate
                history.append(cost)
                previous, rho = damping, new_rho
                factor = .35 if rho > .75 else (2. if rho < .25 else 1.)
                current = float(np.clip(damping*factor, MIN_LAMBDA, MAX_LAMBDA))
                accepted = True
                break
            rejects += 1
            if solves == budget or damping >= MAX_LAMBDA:
                break
            damping = min(MAX_LAMBDA, 10*damping)
        if not accepted:
            status = 'budget' if solves >= budget else 'no_descent'
            break
        if state.cost-cost <= 1e-8*(1+state.cost):
            status = 'small_decrease'
            break
        if solves >= budget:
            status = 'budget'
            break
    poses, marks, bias = pb.unpack(x)
    return Estimate(x,poses,marks,float(bias),history,solves,rejects,status,time.perf_counter()-started,
                    advisor_calls,pb.loss_evaluations)
