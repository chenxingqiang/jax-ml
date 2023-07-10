# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Justin Vincent
#          Lars Buitinck
# License: BSD 3 clause

import jax.numpy as jnp
import pytest

from xlearn.utils._testing import assert_array_equal
from xlearn.utils.fixes import _object_dtype_isnan, delayed


@pytest.mark.parametrize("dtype, val", ([object, 1], [object, "a"], [float, 1]))
def test_object_dtype_isnan(dtype, val):
    X = jnp.array([[val, jnp.nan], [jnp.nan, val]], dtype=dtype)

    expected_mask = jnp.array([[False, True], [True, False]])

    mask = _object_dtype_isnan(X)

    assert_array_equal(mask, expected_mask)


def test_delayed_deprecation():
    """Check that we issue the FutureWarning regarding the deprecation of delayed."""

    def func(x):
        return x

    warn_msg = "The function `delayed` has been moved from `xlearn.utils.fixes`"
    with pytest.warns(FutureWarning, match=warn_msg):
        delayed(func)
