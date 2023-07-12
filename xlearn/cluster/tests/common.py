"""
Common utilities for testing clustering.

"""

import jax.numpy as jnp

###############################################################################
# Generate sample data


def generate_clustered_data(
    seed=0, n_clusters=3, n_features=2, n_samples_per_cluster=20, std=0.4
):
    prng = jax.random.RandomState(seed)

    # the data is voluntary shifted away from zero to check clustering
    # algorithm robustness with regards to non centered data
    means = (
        jnp.array(
            [
                [1, 1, 1, 0],
                [-1, -1, 0, 1],
                [1, -1, 1, 1],
                [-1, 1, 1, 0],
            ]
        )
        + 10
    )

    X = jnp.empty((0, n_features))
    for i in range(n_clusters):
        X = jnp.r_[
            X,
            means[i][:n_features] + std * prng.randn(n_samples_per_cluster, n_features),
        ]
    return X
