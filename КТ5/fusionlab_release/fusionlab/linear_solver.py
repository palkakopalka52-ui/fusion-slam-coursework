"""Weighted linear least squares, anchored 1-D graph, fixed-point iteration."""
import numpy as np


def normal_system(A, z, sigma):
    A, z, sigma = map(lambda v: np.asarray(v, float), (A, z, sigma))
    if A.ndim != 2 or z.shape != (A.shape[0],) or sigma.shape != z.shape:
        raise ValueError('incompatible shapes')
    if not (np.isfinite(A).all() and np.isfinite(z).all() and
            np.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError('finite inputs and positive sigma required')
    B, d = A / sigma[:, None], z / sigma
    if np.linalg.matrix_rank(B) != A.shape[1]:
        raise ValueError('unobservable direction: fix gauge or add information')
    return B.T @ B, B.T @ d, B, d


def fixed_point(H, b, x0=None, alpha=None, tol=1e-8, max_iter=5000):
    H, b = np.asarray(H,float), np.asarray(b,float)
    if H.shape != (b.size,b.size) or not np.allclose(H,H.T):
        raise ValueError('symmetric square H required')
    eig = np.linalg.eigvalsh(H)
    mu, L = eig[0], eig[-1]
    if not np.isfinite(eig).all() or mu <= 0:
        raise ValueError('H must be positive definite')
    alpha = 2/(mu+L) if alpha is None else float(alpha)
    if not (0 < alpha < 2/L):
        raise ValueError('alpha outside proven contraction interval')
    c = max(abs(1-alpha*mu), abs(1-alpha*L))
    x = np.zeros_like(b) if x0 is None else np.asarray(x0,float).copy()
    history = [x.copy()]
    for _ in range(max_iter):
        if np.linalg.norm(H@x-b)/mu <= tol:
            break
        x -= alpha*(H@x-b)
        history.append(x.copy())
    bound = np.linalg.norm(H@x-b)/mu
    return dict(x=x, history=np.asarray(history), c=c, mu=mu, L=L,
                alpha=alpha, error_bound=bound,
                status='tolerance' if bound <= tol else 'max_iter')
