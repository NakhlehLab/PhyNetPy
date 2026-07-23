"""Check eigendecomposition-based expm vs scipy expm for biallelic Q."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.linalg import expm
from phynetpy.BiMarkers import BiMarkersTransition, state_dim

for max_n in [6, 8, 10, 12, 15]:
    q = BiMarkersTransition(max_n, 1.0, 1.0, 0.005)
    Q = q.getQ()
    d = Q.shape[0]
    # eigendecomposition
    w, V = np.linalg.eig(Q)
    Vinv = np.linalg.inv(V)
    cond = np.linalg.cond(V)

    maxerr = 0.0
    ts = [0.001, 0.01, 0.05, 0.2, 1.0, 5.0]
    for t in ts:
        P_true = expm(Q * t)
        P_eig = (V * np.exp(w * t)) @ Vinv
        P_eig = np.real(P_eig)
        maxerr = max(maxerr, np.max(np.abs(P_true - P_eig)))

    # timing
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        for t in ts:
            expm(Q * t)
    t_expm = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        for t in ts:
            np.real((V * np.exp(w * t)) @ Vinv)
    t_eig = time.perf_counter() - t0

    print(f"max_n={max_n:2d} d={d:3d} cond(V)={cond:.2e} maxerr={maxerr:.2e} "
          f"expm={1000*t_expm/(N*len(ts)):.3f}ms eig={1000*t_eig/(N*len(ts)):.3f}ms "
          f"speedup={t_expm/t_eig:.1f}x")
