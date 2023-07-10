# Author: Nelle Varoquaux, Andrew Tulloch, Antony Lee

# Uses the pool adjacent violators algorithm (PAVA), with the
# enhancement of searching for the longest decreasing subsequence to
# pool at each step.

import jax.numpy as jnp

def _inplace_contiguous_isotonic_regression(y: jnp.ndarray, w: jnp.ndarray) -> tuple:
    n = y.shape[0]
    target = jnp.arange(n)

    i = 0
    while i < n:
        k = target[i] + 1
        if k == n:
            break
        if y[i] < y[k]:
            i = k
            continue
        sum_wy = w[i] * y[i]
        sum_w = w[i]
        while True:
            prev_y = y[k]
            sum_wy += w[k] * y[k]
            sum_w += w[k]
            k = target[k] + 1
            if k == n or prev_y < y[k]:
                y = y.at[i].set(sum_wy / sum_w)
                w = w.at[i].set(sum_w)
                target = target.at[i].set(k - 1)
                target = target.at[k - 1].set(i)
                if i > 0:
                    i = target[i - 1]
                break
    i = 0
    while i < n:
        k = target[i] + 1
        y = y.at[i + 1 : k].set(y[i])
        i = k
    return y, w


def _make_unique(X: jnp.ndarray, y: jnp.ndarray, sample_weights: jnp.ndarray) -> tuple:
    """Average targets for duplicate X, drop duplicates.

    Aggregates duplicate X values into a single X value where
    the target y is a (sample_weighted) average of the individual
    targets.

    Assumes that X is ordered, so that all duplicates follow each other.
    """
    unique_values = len(jnp.unique(X))

    if X.dtype == jnp.float32:
        dtype = jnp.float32
    else:
        dtype = jnp.float64

    y_out = jnp.empty(unique_values, dtype=dtype)
    x_out = jnp.empty_like(y_out)
    weights_out = jnp.empty_like(y_out)

    current_x = X[0]
    current_y = 0
    current_weight = 0
    i = 0
    n_samples = len(X)
    eps = jnp.finfo(dtype).resolution

    for j in range(n_samples):
        x = X[j]
        if x - current_x >= eps:
            # next unique value
            x_out = x_out.at[i].set(current_x)
            weights_out = weights_out.at[i].set(current_weight)
            y_out = y_out.at[i].set(current_y / current_weight)
            i += 1
            current_x = x
            current_weight = sample_weights[j]
            current_y = y[j] * sample_weights[j]
        else:
            current_weight += sample_weights[j]
            current_y += y[j] * sample_weights[j]

    x_out = x_out.at[i].set(current_x)
    weights_out = weights_out.at[i].set(current_weight)
    y_out = y_out.at[i].set(current_y / current_weight)
    return x_out[:i+1], y_out[:i+1], weights_out[:i+1]
