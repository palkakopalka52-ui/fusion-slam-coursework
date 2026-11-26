"""Paired stress test on new routes using the stored advisors and actual cost counters."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from study_neuro import BUDGET, TRAIN, VALID, TEST, DampingAdvisor, episode, solve, score
from slam_lab import COURSE_HUBER, Problem
from slam_lab.solver import linear_step, trial

STRESS_SEEDS = tuple(range(2601,2607))


def gradient_check(graph, damping):
    pb = Problem(graph, COURSE_HUBER)
    x = pb.initial()
    state = linear_step(pb,x,2.5,damping,1.)
    h = 1e-6
    numeric = np.array([(pb.objective(x+h*d,2.5)-pb.objective(x-h*d,2.5))/(2*h)
                        for d in np.eye(pb.dim)])
    candidate, cost, _ = trial(pb,x,state,damping,2.5)
    check = dict(max_abs_error=float(np.max(abs(numeric-state.g))),
                 relative_inf_error=float(np.linalg.norm(numeric-state.g,np.inf) /
                                          max(1.,np.linalg.norm(numeric,np.inf))),
                 g_dot_step=float(state.g @ (candidate-x)),
                 initial_cost=state.cost,trial_cost=cost)
    if check['relative_inf_error'] > 1e-6 or not (check['g_dot_step'] < 0 and cost < state.cost):
        raise AssertionError(check)
    return check


def run(folder='results'):
    folder = Path(folder)
    protocol = json.loads((folder/'protocol.json').read_text())
    lam = protocol['initial_lambda']
    if set(STRESS_SEEDS).intersection((*TRAIN,*VALID,*TEST)):
        raise ValueError('stress seed overlaps an earlier split')
    methods = {'classical':'classical', 'mlp':DampingAdvisor.load(folder/'mlp.npz'),
               'rnn':DampingAdvisor.load(folder/'rnn.npz')}
    records = []
    with threadpool_limits(limits=1):
        for seed in STRESS_SEEDS:
            for scene in ('clean','outliers'):
                graph,truth = episode(seed,scene)
                base = Problem(graph).initial()
                error = np.random.default_rng(seed+8200).normal(size=(graph.poses-1,3))*[.9,.9,.6]
                for start in ('wheel','perturbed'):
                    x = base.copy()
                    if start == 'perturbed':
                        x[:error.size] += error.ravel()
                    for name,policy in methods.items():
                        est = solve(graph,x0=x,policy=policy,initial_lambda=lam,
                                    budget=BUDGET,robust_kinds=COURSE_HUBER)
                        records.append(dict(seed=seed,scenario=scene,start=start,policy=name,
                                ate_m=score(est.poses,truth)['ate_m'],solves=est.solves,
                                advisor_calls=est.advisor_calls,loss_evaluations=est.loss_evaluations,
                                rejected_steps=est.rejects,stop=est.status,
                                failed=est.status not in ('small_decrease','small_gradient')))
        certificate = gradient_check(episode(STRESS_SEEDS[0],'outliers')[0],lam)
    (folder/'gradient_check.json').write_text(json.dumps(certificate,indent=2))
    df = pd.DataFrame(records)
    df.to_csv(folder/'stress_runs.csv',index=False)
    summary = df.groupby(['scenario','start','policy']).agg(
        ate_median=('ate_m','median'),ate_min=('ate_m','min'),ate_max=('ate_m','max'),
        solves_median=('solves','median'),solves_min=('solves','min'),solves_max=('solves','max'),
        calls_median=('advisor_calls','median'),loss_median=('loss_evaluations','median'),
        rejected_steps=('rejected_steps','sum'),failed_runs=('failed','sum'))
    summary.to_csv(folder/'stress_summary.csv')
    return summary


if __name__ == '__main__':
    run()
