"""
Benchmark jax-learn's Ward implement compared to SciPy's
"""

import time

import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.cluster import hierarchy

from xlearn.cluster import AgglomerativeClustering

ward = AgglomerativeClustering(n_clusters=3, linkage="ward")

n_samples = jnp.logspace(0.5, 3, 9)
n_features = jnp.logspace(1, 3.5, 7)
N_samples, N_features = jnp.meshgrid(n_samples, n_features)
jaxs_time = jnp.zeros(N_samples.shape)
scipy_time = jnp.zeros(N_samples.shape)

for i, n in enumerate(n_samples):
    for j, p in enumerate(n_features):
        X = np.random.normal(size=(n, p))
        t0 = time.time()
        ward.fit(X)
        jaxs_time[j, i] = time.time() - t0
        t0 = time.time()
        hierarchy.ward(X)
        scipy_time[j, i] = time.time() - t0

ratio = jaxs_time / scipy_time

plt.figure("jax-learn Ward's method benchmark results")
plt.imshow(jnp.log(ratio), aspect="auto", origin="lower")
plt.colorbar()
plt.contour(
    ratio,
    levels=[
        1,
    ],
    colors="k",
)
plt.yticks(range(len(n_features)), n_features.astype(int))
plt.ylabel("N features")
plt.xticks(range(len(n_samples)), n_samples.astype(int))
plt.xlabel("N samples")
plt.title("Jax's time, in units of scipy time (log)")
plt.show()
