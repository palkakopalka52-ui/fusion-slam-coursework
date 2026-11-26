"""Reproduce the benchmark tables and the figures from independently generated episodes."""
from dataclasses import asdict, replace
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

from slam_lab import Problem, solve, wheel_path
from slam_lab.bench import episode, score

SEEDS = [23,24,25,26,71,72,73,74]
CONDITIONS = ['clean','outliers','dropout','false_loop']


def run(folder='results'):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    records = []
    example = None
    with threadpool_limits(limits=1):
        for condition in CONDITIONS:
            for seed in SEEDS:
                graph, truth = episode(seed,condition)
                wheel = wheel_path(graph)
                coverage = sum(f.kind == 'vo' for f in graph.factors)/(graph.poses-1)
                records.append(dict(seed=seed, scenario=condition, method='wheel', vo_coverage=coverage,
                                    **score(wheel,truth)))
                for method, delta, omit in [('WLS',None,False),('Huber',2.5,False),('Huber без петли',2.5,True)]:
                    active = replace(graph,factors=tuple(f for f in graph.factors if f.kind != 'loop')) if omit else graph
                    sol = solve(active,delta=delta)
                    records.append(dict(seed=seed,scenario=condition,method=method,vo_coverage=coverage,
                                        **score(sol.poses,truth,sol.landmarks), bias_rad_s=sol.bias,
                                        bias_error_rad_s=sol.bias-truth.bias, final_cost=sol.history[-1],
                                        accepted=len(sol.history)-1, solves=sol.solves, status=sol.status))
                    if seed == 23 and condition == 'false_loop' and method == 'Huber':
                        example = graph,truth,sol

        graph, truth, sol = example
        pb = Problem(graph)
        start = pb.initial()
        other = least_squares(lambda x:pb.linearize(x,False)[0], start, jac='2-point', method='trf',
                              ftol=1e-12, xtol=1e-12, gtol=1e-10)
        ours = solve(graph,delta=None,iterations=60,budget=100)
        check = {'seed':23,'scenario':'false_loop','own_WLS':ours.history[-1],'scipy_WLS':other.cost,
                 'max_state_diff':float(np.max(np.abs(ours.x-other.x)))}
        assert abs(check['own_WLS']-check['scipy_WLS']) < 1e-4
        (folder/'scipy_check.json').write_text(json.dumps(check,indent=2))
    df = pd.DataFrame(records)
    df.to_csv(folder/'runs.csv',index=False)
    agg = df.groupby(['scenario','method']).agg(ate_median=('ate_m','median'),ate_min=('ate_m','min'),
              ate_max=('ate_m','max'),rpe_median=('rpe_m','median'),map_median=('map_m','median')).reset_index()
    agg.to_csv(folder/'summary.csv',index=False)
    (folder/'measurements.json').write_text(json.dumps(asdict(graph),indent=2))
    np.savez(folder/'reference.npz',poses=truth.poses,landmarks=truth.landmarks,bias=truth.bias)
    fig,ax = plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
    for path,name,color in [(truth.poses,'Эталон','#222222'),(wheel_path(graph),'Колёса','#bd6c30'),
                             (ours.poses,'WLS','#73879a'),(sol.poses,'Huber','#007c83')]:
        ax[0].plot(*path[:,:2].T,'o-',label=name,color=color,ms=2)
    ax[0].scatter(*truth.landmarks.T,color='black',marker='x',label='Ориентиры, GT')
    ax[0].scatter(*sol.landmarks.T,edgecolor='#007c83',facecolor='none',label='Ориентиры, Huber')
    ax[0].set(aspect='equal',title='Ложная петля · seed 23',xlabel='x, м',ylabel='y, м')
    ax[0].legend(fontsize=8,ncol=2)
    for name,color in [('wheel','#bd6c30'),('WLS','#73879a'),('Huber','#007c83')]:
        part = df[(df.method==name)&(df.scenario=='false_loop')]
        ax[1].plot(range(len(SEEDS)),part.ate_m,'o-',label=name,color=color)
    ax[1].set(xticks=range(len(SEEDS)),xticklabels=SEEDS,xlabel='Seed маршрута',ylabel='ATE, м',title='Разброс по маршрутам')
    ax[1].grid(axis='y',alpha=.25); ax[1].legend()
    fig.savefig(folder/'trajectories.png',dpi=150)
    plt.close(fig)
    return agg


if __name__ == '__main__':
    print(run().to_string(index=False))
