"""Run from the exported folder: python smoke_test.py."""
import os, importlib.util, json
from pathlib import Path
os.environ.setdefault('KERAS_BACKEND', 'tensorflow' if importlib.util.find_spec('tensorflow') else 'torch')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import keras
import joblib
from fusionlab import slam_core as sc, synthetic as sy, fusion_adapter as fa, learned_sensor as ls
from fusionlab.geometry2d import pose_matrix, inverse
from fusionlab.linear_solver import normal_system, fixed_point
root=Path(__file__).resolve().parent
config=json.loads((root/'config.json').read_text())
net=keras.models.load_model(root/'predictor.keras',compile=False)
boost=joblib.load(root/'boost.joblib')
G=pose_matrix([1.,2.,.3]); np.testing.assert_allclose(inverse(G)@G,np.eye(3),atol=1e-12)
A=np.array([[1.,0.],[-1.,1.],[0.,1.]]);z=np.array([1.,1.,2.1]);H,b,B,d=normal_system(A,z,np.ones(3))
np.testing.assert_allclose(fixed_point(H,b)['x'],np.linalg.lstsq(B,d,rcond=None)[0],atol=1e-7)
assert fa.calibrate_scale([1.,2.],[-1.,-2.])['a']==0
assert fa.calibrate_scale([],[])['a']==0
for reg in ['nominal','shift']:
    X,_,ref=sy.generate(6000,reg); raw=fa.observations(ref)
    pred=ls.predict_bias(net,X); assert pred.shape==(sy.T,)
    assert ls.predict_bias(boost,X).shape==(sy.T,)
    a=config['a_by_bench_scenario'][reg]  # bench scenario only
    sol=fa.estimate_slam(raw,pred,a=a,sigma=config['sigma_raw'])
    baseline=fa.estimate_slam(raw,np.zeros(sy.T),a=0,sigma=config['sigma_raw'])
    if reg == 'shift':
        by_default = fa.correct_factors(raw, pred)
        for original, revised in zip(raw.factors, by_default.factors, strict=True):
            np.testing.assert_array_equal(revised.z, original.z)
        uncalibrated = fa.estimate_slam(raw, pred, sigma=config['sigma_raw'])
        np.testing.assert_array_equal(uncalibrated['x'], baseline['x'])
    if a==0: np.testing.assert_allclose(sol['x'],baseline['x'],atol=1e-12)
    print(reg, 'ATE [m]', round(fa.evaluate(sol,ref)['ate_rmse_m'],4), 'status',sol['status'])
print('PASS: export loaded, geometry, linear solver, gate, new episode')
