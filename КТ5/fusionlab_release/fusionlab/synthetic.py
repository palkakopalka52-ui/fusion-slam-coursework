"""Synthetic teaching generator adapted verbatim from the final-sprint pilot.
The input channels are dimensionless synthetic diagnostic proxies, not actual sensors.
Ground truth is returned separately for training labels and benchmarking.
"""
import numpy as np
from . import slam_core as sc

W=8; T=24

def generate(seed,regime='nominal'):
    rng=np.random.default_rng(seed)
    load=np.empty(T); vib=np.empty(T); load[0]=rng.uniform(-1,1); vib[0]=rng.uniform(-1,1)
    for k in range(1,T):
        load[k]=np.clip(.65*load[k-1]+rng.normal(0,.55),-1.5,1.5)
        vib[k]=np.clip(.72*vib[k-1]+rng.normal(0,.50),-1.5,1.5)
    features=np.column_stack([load,vib])+rng.normal(0,.12,(T,2))
    # Assumed persistent nonlinear slip (m per interval); diagnostic features are synthetic.
    b=np.zeros(T); prev=0.
    for k in range(T):
        target=.38*np.tanh(2.2*(load[k]+.45*vib[k]-.15))+.15*np.tanh(2.5*load[k]*vib[k])
        prev=.45*prev+.55*target; b[k]=prev
    if regime=='null': b[:]=0.
    if regime=='shift': b=-b
    poses=np.zeros((T+1,3)); factors=[]; labels=[]
    for k in range(T):
        dyaw=.20+.12*np.sin(k*.43+seed*.1)+rng.normal(0,.025)
        dx=.48+.09*np.cos(k*.65+seed*.2)+rng.normal(0,.025)
        d=np.array([dx,.02*np.sin(k*.4),dyaw]); poses[k+1]=sc.integrate_relative(poses[k],d)
        sig=np.array([.065,.08,.045]); w=d+rng.normal(0,sig); w[0]+=b[k]; w[2]=sc.wrap(w[2]); labels.append(w[0]-d[0])
        factors.append(sc.Factor('wheel',k,k+1,w,sig.copy()))
        if rng.random()<.45:
            s=np.array([.065,.065,.038]); z=d+rng.normal(0,s); z[2]=sc.wrap(z[2]); factors.append(sc.Factor('visual',k,k+1,z,s))
        factors.append(sc.Factor('gyro',k,k+1,np.array([sc.wrap(dyaw+.025+rng.normal(0,.025))]),np.array([.025])))
    idx=np.linspace(2,T-2,6,dtype=int); landmarks=poses[idx,:2]+rng.normal(0,1.,(6,2))
    for j,pt in enumerate(landmarks):
        closest=np.argsort(np.linalg.norm(poses[:,:2]-pt,axis=1))[:5]
        for k in closest:
            z=sc.range_bearing(poses[k],pt)+rng.normal(0,[.12,.055]); z[0]=max(.05,z[0]); z[1]=sc.wrap(z[1]); factors.append(sc.Factor('landmark',int(k),j,z,np.array([.12,.055])))
    factors.append(sc.Factor('bias_prior',0,0,np.array([0.]),np.array([.12])))
    data=sc.Dataset(factors,T+1,6,seed,.025,poses,landmarks)
    pad=np.vstack([np.zeros((W-1,2)),features]); windows=np.stack([pad[k:k+W] for k in range(T)])
    return windows,np.asarray(labels),data

def arrays(seeds):
    pairs=[generate(s) for s in seeds]; return np.concatenate([p[0] for p in pairs]),np.concatenate([p[1] for p in pairs])
