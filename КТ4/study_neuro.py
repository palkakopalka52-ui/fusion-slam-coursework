"""Train by imitating offline LM trials and compare policies on held-out routes."""
from pathlib import Path
import json
import platform
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sibling=Path(__file__).resolve().parents[1]/'КТ3'
if sibling.is_dir():
    sys.path.insert(0,str(sibling))

from slam_lab import solve
from slam_lab.bench import episode, score
from damping_advisor import DampingAdvisor, GRID, teacher_sequence

TRAIN=tuple(range(310,330))
VALID=tuple(range(710,716))
TEST=tuple(range(1810,1818))
WIDTHS=(8,12,16)
RATES=(.006,.014,.025)
SCENARIOS=('clean','outliers','dropout','false_loop')
BUDGET=24


def sequences(seeds):
    return [teacher_sequence(episode(seed,SCENARIOS[k%len(SCENARIOS)])[0]) for k,seed in enumerate(seeds)]


def tune_classical():
    rows=[]
    for lam in (.0003,.003,.03):
        for k,seed in enumerate(VALID):
            graph,_=episode(seed,SCENARIOS[k%4])
            sol=solve(graph,budget=BUDGET,initial_lambda=lam)
            rows.append(dict(initial_lambda=lam,seed=seed,solves=sol.solves,cost=sol.history[-1],status=sol.status))
    val=pd.DataFrame(rows)
    med=val.groupby('initial_lambda').solves.mean()
    return float(med.idxmin()),val


def select_model(name, train, valid):
    """Select weights, epoch and architecture without running the held-out SLAM tests."""
    candidates=[]
    best_key=None
    for width in WIDTHS:
        for lr in RATES:
            model=DampingAdvisor(width=width,recurrent=name=='rnn',cap=.8,seed=19)
            trace=model.fit(train,valid,epochs=65,lr=lr)
            epoch=int(np.argmin(trace[:,2]))
            mse=float(trace[epoch,2])
            row=dict(model=name,width=width,lr=lr,best_epoch=epoch+1,train_mse=float(trace[epoch,1]),
                     validation_mse=mse,max_abs_u=float(np.max(abs(model.params['u']))))
            candidates.append(row)
            # If validation MSE ties, prefer a narrower layer, then a smaller learning rate.
            key=(mse,width,lr)
            if best_key is None or key<best_key:
                best_key=key
                chosen,chosen_trace,chosen_row=model,trace,row
    for row in candidates:
        row['selected']=row is chosen_row
    return chosen,chosen_trace,chosen_row,candidates


def run(folder='results'):
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    records=[];history=[];selection=[];chosen={}
    with threadpool_limits(limits=1):
        train,valid=sequences(TRAIN),sequences(VALID)
        lam,baselines=tune_classical()
        baselines.to_csv(folder/'classical_validation.csv',index=False)
        policies={'classical':'classical','fixed':'fixed'}
        for name in ('mlp','rnn'):
            model,h,choice,candidates=select_model(name,train,valid)
            chosen[name]=choice
            selection.extend(candidates)
            model.save(folder/(name+'.npz'))
            policies[name]=DampingAdvisor.load(folder/(name+'.npz'))
            history.extend(dict(model=name,epoch=int(e),train=a,validation=b) for e,a,b in h)
        pd.DataFrame(selection).to_csv(folder/'selection.csv',index=False)
        ablated=DampingAdvisor.load(folder/'rnn.npz')
        ablated.params['u'][:]=0.
        policies['rnn_u0']=ablated
        for scenario in (*SCENARIOS,'transfer'):
            for seed in TEST:
                graph,ref=episode(seed,scenario)
                for name,policy in policies.items():
                    sol=solve(graph,policy=policy,initial_lambda=lam,budget=BUDGET)
                    assert np.all(np.diff(sol.history)<0)
                    records.append(dict(seed=seed,scenario=scenario,policy=name,**score(sol.poses,ref,sol.landmarks),
                          cost=sol.history[-1],solves=sol.solves,rejects=sol.rejects,status=sol.status,
                          elapsed_ms=1000*sol.seconds,accepted=len(sol.history)-1))
    df=pd.DataFrame(records); hist=pd.DataFrame(history)
    df.to_csv(folder/'runs.csv',index=False);hist.to_csv(folder/'learning.csv',index=False)
    summary=df.groupby(['scenario','policy'])[['ate_m','rpe_m','map_m','cost','solves','rejects','elapsed_ms']].mean()
    summary.to_csv(folder/'summary.csv')
    protocol=dict(train=TRAIN,validation=VALID,test=TEST,training_steps=sum(len(y) for _,y in train),
                  validation_steps=sum(len(y) for _,y in valid),teacher_grid=GRID.tolist(),
                  width_grid=WIDTHS,learning_rate_grid=RATES,selection_metric='minimum validation log-damping MSE',
                  selected=chosen,cap=.8,epochs=65,budget=BUDGET,initial_lambda=lam,
                  model_seed=19,numpy=np.__version__,python=platform.python_version())
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for name,color in [('mlp','#b56931'),('rnn','#007b80')]:
        sub=hist[hist.model==name]
        ax[0].plot(sub.epoch,sub.train,ls=':',color=color,label=name.upper()+' · train')
        ax[0].plot(sub.epoch,sub.validation,color=color,label=name.upper()+' · val')
    ax[0].set(xlabel='Эпоха',ylabel='MSE по log₁₀ λ',title='Обучение советника');ax[0].legend(fontsize=8)
    means=df.groupby('policy',sort=False).solves.mean()
    for ix,(method,m) in enumerate(means.items()):
        ax[1].bar(ix,m,color=['#546e7a','#adb8be','#b56931','#007b80','#81b7b4'][ix])
    ax[1].set(xticks=range(len(means)),xticklabels=means.index,ylabel='Линейных решений на запуск',title='40 отложенных сценариев')
    fig.savefig(folder/'learning_and_cost.png',dpi=150);plt.close(fig)
    return summary


if __name__=='__main__':
    run()
