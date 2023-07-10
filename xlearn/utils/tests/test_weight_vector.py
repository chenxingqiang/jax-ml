import jax.numpy as jnp
import pytest

from xlearn.utils._weight_vector import (
    WeightVector32,
    WeightVector64,
)


@pytest.mark.parametrize(
    "dtype, WeightVector",
    [
        (jnp.float32, WeightVector32),
        (jnp.float64, WeightVector64),
    ],
)
def test_type_invariance(dtype, WeightVector):
    """Check the `dtype` consistency of `WeightVector`."""
    weights = np.random.rand(100).astype(dtype)
    average_weights = np.random.rand(100).astype(dtype)

    weight_vector = WeightVector(weights, average_weights)

    assert jnp.asarray(weight_vector.w).dtype is jnp.dtype(dtype)
    assert jnp.asarray(weight_vector.aw).dtype is jnp.dtype(dtype)
