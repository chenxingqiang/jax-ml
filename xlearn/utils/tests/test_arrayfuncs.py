import jax.numpy as jnp
import jax
import pytest

from xlearn.utils._testing import assert_allclose
from xlearn.utils.arrayfuncs import min_pos


def test_min_pos():
    # Check that min_pos returns a positive value and that it's consistent
    # between float and double
    X = jax.random.RandomState(0).randn(100)

    min_double = min_pos(X)
    min_float = min_pos(X.astype(jnp.float32))

    assert_allclose(min_double, min_float)
    assert min_double >= 0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_min_pos_no_positive(dtype):
    # Check that the return value of min_pos is the maximum representable
    # value of the input dtype when all input elements are <= 0 (#19328)
    X = jnp.full(100, -1.0).astype(dtype, copy=False)

    assert min_pos(X) == jnp.finfo(dtype).max
