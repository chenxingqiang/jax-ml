# Translation of the cython/C++ functions to Python with JAX
import jax.numpy as jnp
from jax import jit, vmap
from scipy.sparse import csr_matrix, csc_matrix


@jax.jit
def csr_row_norms(X):
    """Squared L2 norm of each row in CSR matrix X."""
    if X.dtype not in [np.float32, np.float64]:
        X = csr_matrix(X, dtype=np.float64)
    return _sqeuclidean_row_norms_sparse(X.data, X.indptr)


@jax.jit
def _sqeuclidean_row_norms_sparse(X_data, X_indptr):
    n_samples = X_indptr.shape[0] - 1

    dtype = jnp.float32 if X_data.dtype == jnp.float32 else jnp.float64

    squared_row_norms = jnp.zeros(n_samples, dtype=dtype)

    for i in range(n_samples):
        for j in range(X_indptr[i], X_indptr[i + 1]):
            squared_row_norms = jax.ops.index_add(squared_row_norms, i, X_data[j] * X_data[j])

    return squared_row_norms


@jax.jit
def csr_mean_variance_axis0(X, weights=None, return_sum_weights=False):
    """Compute mean and variance along axis 0 on a CSR matrix
    Uses a jnp.float64 accumulator."""

    if X.dtype not in [np.float32, np.float64]:
        X = csr_matrix(X, dtype=np.float64)

    if weights is None:
        weights = jnp.ones(X.shape[0], dtype=X.dtype)

    means, variances, sum_weights = _csr_mean_variance_axis0(
        X.data, X.shape[0], X.shape[1], X.indices, X.indptr, weights)

    if return_sum_weights:
        return means, variances, sum_weights
    return means, variances


@jax.jit
def _csr_mean_variance_axis0(X_data, n_samples, n_features, X_indices, X_indptr, weights):
    # Initialize variables
    means = jnp.zeros(n_features, dtype=jnp.float64)
    variances = jnp.zeros(n_features, dtype=jnp.float64)
    sum_weights = jnp.full(n_features, jnp.sum(weights, dtype=jnp.float64))
    sum_weights_nz = jnp.zeros(n_features, dtype=jnp.float64)
    correction = jnp.zeros(n_features, dtype=jnp.float64)
    counts = jnp.full(n_features, weights.shape[0], dtype=jnp.uint64)
    counts_nz = jnp.zeros(n_features, dtype=jnp.uint64)

    # Main loops
    for row_ind in range(len(X_indptr) - 1):
        for i in range(X_indptr[row_ind], X_indptr[row_ind + 1]):
            col_ind = X_indices[i]
            if not isnan(X_data[i]):
                means = jax.ops.index_add(means, col_ind, X_data[i] * weights[row_ind])
                sum_weights_nz = jax.ops.index_add(sum_weights_nz, col_ind, weights[row_ind])
                counts_nz = jax.ops.index_add(counts_nz, col_ind, 1)
            else:
                sum_weights = jax.ops.index_add(sum_weights, col_ind, -weights[row_ind])
                counts = jax.ops.index_add(counts, col_ind, -1)

    # Normalize means
    means = means / sum_weights

    # Second loop for variances
    for row_ind in range(len(X_indptr) - 1):
        for i in range(X_indptr[row_ind], X_indptr[row_ind + 1]):
            col_ind = X_indices[i]
            if not isnan(X_data[i]):
                diff = X_data[i] - means[col_ind]
                correction = jax.ops.index_add(correction, col_ind, diff * weights[row_ind])
                variances = jax.ops.index_add(variances, col_ind, diff * diff * weights[row_ind])

    # Finalize variances
    for feature_idx in range(n_features):
        if counts[feature_idx] != counts_nz[feature_idx]:
            correction = jax.ops.index_update(correction, feature_idx, (sum_weights[feature_idx] - sum_weights_nz[feature_idx]) * means[feature_idx])
        correction = correction**2 / sum_weights
        if counts[feature_idx] != counts_nz[feature_idx]:
            variances = jax.ops.index_add(variances, feature_idx, (sum_weights[feature_idx] - sum_weights_nz[feature_idx]) * means[feature_idx]**2)
        variances = (variances - correction) / sum_weights

    # Return results depending on type
    if X_data.dtype == jnp.float32:
        return (jnp.array(means, dtype=jnp.float32),
                jnp.array(variances, dtype=jnp.float32),
                jnp.array(sum_weights, dtype=jnp.float32))
    else:
        return jnp.asarray(means), jnp.asarray(variances), jnp.asarray(sum_weights)