import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, pmap
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import row_norms

# Number of samples per data chunk defined as a global constant.
CHUNK_SIZE = 256

# Euclidean distance between a dense and b dense
@jit
def _euclidean_dense_dense(a, b, squared):
    result = jnp.sum((a - b)**2)
    return result if squared else jnp.sqrt(result)

# Euclidean distance between a sparse and b dense
@jit
def _euclidean_sparse_dense(a_data, a_indices, b, b_squared_norm, squared):
    b = b[a_indices]
    result = jnp.sum((a_data - b)**2) - b_squared_norm
    result += b_squared_norm
    result = jnp.where(result < 0, 0.0, result)
    return result if squared else jnp.sqrt(result)

# Compute inertia for dense input data
@pmap
def _inertia_dense(X, sample_weight, centers, labels, n_threads, single_label=-1):
    n_samples, n_features = X.shape
    inertia = 0.0
    for i in range(n_samples):
        if single_label < 0 or single_label == labels[i]:
            sq_dist = _euclidean_dense_dense(X[i], centers[labels[i]], True)
            inertia += sq_dist * sample_weight[i]
    return inertia

# Compute inertia for sparse input data
@pmap
def _inertia_sparse(X, sample_weight, centers, labels, n_threads, single_label=-1):
    n_samples = X.shape[0]
    inertia = 0.0
    centers_squared_norms = row_norms(centers, squared=True)
    for i in range(n_samples):
        if single_label < 0 or single_label == labels[i]:
            sq_dist = _euclidean_sparse_dense(
                X.data[X.indptr[i]: X.indptr[i + 1]],
                X.indices[X.indptr[i]: X.indptr[i + 1]],
                centers[labels[i]], centers_squared_norms[labels[i]], True)
            inertia += sq_dist * sample_weight[i]
    return inertia

# Relocate centers which have no sample assigned to them
@jit
# Searching for the JAX equivalent of "numpy.argpartition"
search("numpy.argpartition jax equivalent")
