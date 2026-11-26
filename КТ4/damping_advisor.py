"""Small trainable damping policies for the guarded solver in slam_lab."""
from pathlib import Path
import numpy as np
from slam_lab.solver import linear_step, trial
from slam_lab import Problem

# Six observable diagnostics in the order returned by slam_lab.solver.linear_step.
FEATURES = ('log_gradient','log_residual','log_diagonal_ratio','downweighted','last_log_lambda','last_rho')
GRID = np.logspace(-5,1,8)


def teacher_sequence(graph, delta=2.5):
    """Labels use objective reduction only. Resolve near-ties towards the safer damping."""
    pb = Problem(graph)
    x = pb.initial()
    previous, rho = .003, 1.
    features, labels = [], []
    for _ in range(12):
        state=linear_step(pb,x,delta,previous,rho)
        attempts=[trial(pb,x,state,lam,delta) for lam in GRID]
        drops=np.array([state.cost-cost for _,cost,_ in attempts])
        if drops.max() <= 1e-8*(1+state.cost):
            break
        chosen=np.flatnonzero(drops >= .95*drops.max())[-1]
        previous=float(GRID[chosen])
        x,_,rho=attempts[chosen]
        features.append(state.features)
        labels.append(np.log10(previous))
    return np.array(features),np.array(labels)


class DampingAdvisor:
    """A tanh MLP or diagonal recurrent layer, followed by bounded log10(lambda)."""
    def __init__(self,width=12,recurrent=False,cap=.8,seed=19):
        if not (width>=1 and 0<=cap<1):
            raise ValueError('invalid width or memory cap')
        rng=np.random.default_rng(seed)
        self.recurrent,self.cap=recurrent,float(cap)
        self.params={'W':rng.normal(0,.27,(len(FEATURES),width)),'b':np.zeros(width),
                     'v':rng.normal(0,.13,width),'c':np.zeros(1),'u':np.full(width,min(.22, cap) if recurrent else 0.)}
        self.center,self.scale=np.zeros(len(FEATURES)),np.ones(len(FEATURES))
        self.reset()

    def reset(self):
        self.hidden=np.zeros_like(self.params['b'])

    def advance(self, x, previous):
        p=self.params
        return np.tanh(x@p['W']+p['b']+p['u']*previous)

    def forward(self, features):
        x=(np.asarray(features)-self.center)/self.scale
        h=np.zeros((len(x)+1,len(self.hidden)))
        for k,row in enumerate(x):
            h[k+1]=self.advance(row,h[k])
        a=np.tanh(h[1:]@self.params['v']+self.params['c'][0])
        return x,h,a

    def propose(self,features):
        x=(features-self.center)/self.scale
        self.hidden=self.advance(x,self.hidden)
        out=np.tanh(self.hidden@self.params['v']+self.params['c'][0])
        return float(10**(-2.+3.*out))

    def loss(self,sequences):
        err=[(-2+3*self.forward(X)[2]-y)**2 for X,y in sequences]
        return float(np.mean(np.concatenate(err)))

    def fit(self,training,validation,epochs=65,lr=.014):
        X=np.concatenate([x for x,_ in training])
        self.center=X.mean(0)
        self.scale=np.maximum(X.std(0),.05)
        p=self.params
        m={k:np.zeros_like(v) for k,v in p.items()}
        v={k:np.zeros_like(q) for k,q in p.items()}
        n=sum(len(y) for _,y in training)
        history=[]
        best=None
        for epoch in range(1,epochs+1):
            grad={k:np.zeros_like(w) for k,w in p.items()}
            for observed,target in training:
                x,h,a=self.forward(observed)
                # Backpropagate log-damping MSE through the bounded readout and through time.
                do=(2/n)*(-2+3*a-target)*3*(1-a*a)
                grad['v']+=h[1:].T@do
                grad['c'][0]+=do.sum()
                carry=np.zeros_like(self.hidden)
                for k in reversed(range(len(x))):
                    dz=(p['v']*do[k]+carry)*(1-h[k+1]**2)
                    grad['W']+=np.outer(x[k],dz)
                    grad['b']+=dz
                    if self.recurrent:
                        grad['u']+=h[k]*dz
                    carry=p['u']*dz
            for key in p:
                m[key]=.9*m[key]+.1*grad[key]
                v[key]=.999*v[key]+.001*grad[key]**2
                p[key]-=lr*(m[key]/(1-.9**epoch))/(np.sqrt(v[key]/(1-.999**epoch))+1e-8)
            p['u'][:]=np.clip(p['u'],-self.cap,self.cap) if self.recurrent else 0.
            val=self.loss(validation)
            history.append((epoch,self.loss(training),val))
            if best is None or val<best[0]:
                best=(val,{k:w.copy() for k,w in p.items()})
        self.params=best[1]
        self.reset()
        return np.array(history)

    def save(self,path):
        np.savez(Path(path),**self.params,center=self.center,scale=self.scale,cap=self.cap,
                 recurrent=self.recurrent,features=np.array(FEATURES))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if tuple(data['features'])!=FEATURES:
                raise ValueError('incompatible feature order')
            model=cls(len(data['b']),bool(data['recurrent']),float(data['cap']))
            for key,template in model.params.items():
                a=data[key]
                if a.shape!=template.shape or not np.isfinite(a).all():
                    raise ValueError(f'invalid parameter: {key}')
                model.params[key]=a.copy()
            model.center,model.scale=data['center'].copy(),data['scale'].copy()
            if (model.scale.shape!=(len(FEATURES),) or model.center.shape!=model.scale.shape
                    or not np.isfinite(np.r_[model.center,model.scale]).all() or np.any(model.scale<=0)
                    or np.max(np.abs(model.params['u']))>model.cap
                    or (not model.recurrent and np.any(model.params['u']))):
                raise ValueError('incompatible scaling or memory')
        model.reset()
        return model
