"""Run an F9 measurement correction through the KT3 graph, with both KT4 damping policies."""
from pathlib import Path
import csv
import json
import sys
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')

ROOT=Path(__file__).resolve().parent
RELEASE=ROOT/'fusionlab_release'
sys.path[:0]=[str(RELEASE),str(ROOT.parent/'КТ3'),str(ROOT.parent/'КТ4')]
import keras
import numpy as np
from threadpoolctl import threadpool_limits

from fusionlab import learned_sensor,synthetic
from fusionlab.fusion_adapter import observations
from fusionlab.wheel_channel import correct_graph
from slam_lab import Factor,Graph,Problem,solve
from slam_lab.bench import Reference,score
from damping_advisor import DampingAdvisor


def to_graph(raw):
    kinds={'wheel':'wheel','visual':'vo','gyro':'gyro','landmark':'beacon'}
    # The KT3 state estimates gyro bias from gyro + geometric observations without a bias prior.
    factors=tuple(Factor(kinds[f.kind],f.i,f.j,tuple(f.z),tuple(f.sigma),f.dt)
                  for f in raw.factors if f.kind!='bias_prior')
    return Graph(raw.nposes,raw.nlandmarks,factors)


def run():
    config=json.loads((RELEASE/'config.json').read_text())
    model=keras.models.load_model(RELEASE/'predictor.keras',compile=False)
    rows=[]
    with threadpool_limits(limits=1):
        for scenario,a in config['a_by_bench_scenario'].items():
            X,_,benchmark=synthetic.generate(6021,scenario)
            graph=to_graph(observations(benchmark))
            truth=Reference(benchmark.truth_poses,benchmark.truth_landmarks,benchmark.true_gyro_bias)
            initial=Problem(graph).initial()
            predicted=learned_sensor.predict_bias(model,X)
            for name in ['classical','mlp','rnn']:
                policy=name if name=='classical' else DampingAdvisor.load(ROOT.parent/'КТ4/results'/(name+'.npz'))
                fits={}
                for method,alpha in [('raw',0.),('gate',a)]:
                    revised=correct_graph(graph,predicted,a=alpha,sigma=config['sigma_raw'])
                    fits[method]=solve(revised,x0=initial,policy=policy,budget=24)
                    fit=fits[method]
                    rows.append(dict(seed=6021,regime=scenario,policy=name,method=method,a=alpha,
                                     **score(fit.poses,truth,fit.landmarks),status=fit.status,solves=fit.solves))
                if a==0:
                    np.testing.assert_array_equal(fits['raw'].x,fits['gate'].x)
    with (ROOT/'integration.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print('Проверено сочетаний:',len(rows))


if __name__=='__main__':
    run()
