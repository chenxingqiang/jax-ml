import jax.numpy as jnp

# Y_DYTPE is the dtype to which the targets y are converted to. This is also
# dtype for leaf values, gains, and sums of gradients / hessians. The gradients
# and hessians arrays are stored as floats to avoid using too much memory.
Y_DTYPE = jnp.float64
X_DTYPE = jnp.float64
X_BINNED_DTYPE = jnp.uint8  # hence max_bins == 256
# dtype for gradients and hessians arrays
G_H_DTYPE = jnp.float32
X_BITSET_INNER_DTYPE = jnp.uint32

HISTOGRAM_DTYPE = jnp.dtype([
    ('sum_gradients', Y_DTYPE),  # sum of sample gradients in bin
    ('sum_hessians', Y_DTYPE),  # sum of sample hessians in bin
    ('count', jnp.uint32),  # number of samples in bin
])

PREDICTOR_RECORD_DTYPE = jnp.dtype([
    ('value', Y_DTYPE),
    ('count', jnp.uint32),
    ('feature_idx', jnp.intp),
    ('num_threshold', X_DTYPE),
    ('missing_go_to_left', jnp.uint8),
    ('left', jnp.uint32),
    ('right', jnp.uint32),
    ('gain', Y_DTYPE),
    ('depth', jnp.uint32),
    ('is_leaf', jnp.uint8),
    ('bin_threshold', X_BINNED_DTYPE),
    ('is_categorical', jnp.uint8),
    # The index of the corresponding bitsets in the Predictor's bitset arrays.
    # Only used if is_categorical is True
    ('bitset_idx', jnp.uint32)
])

ALMOST_INF = 1e300  # see LightGBM AvoidInf()
