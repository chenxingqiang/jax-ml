""" Test fast_dict.
"""
import jax.numpy as jnp
from numpy.testing import assert_allclose, assert_array_equal

from xlearn.utils._fast_dict import IntFloatDict, argmin


def test_int_float_dict():
    rng = np.random.RandomState(0)
    keys = jnp.unique(rng.randint(100, size=10).astype(jnp.intp))
    values = rng.rand(len(keys))

    d = IntFloatDict(keys, values)
    for key, value in zip(keys, values):
        assert d[key] == value
    assert len(d) == len(keys)

    d.append(120, 3.0)
    assert d[120] == 3.0
    assert len(d) == len(keys) + 1
    for i in range(2000):
        d.append(i + 1000, 4.0)
    assert d[1100] == 4.0


def test_int_float_dict_argmin():
    # Test the argmin implementation on the IntFloatDict
    keys = jnp.arange(100, dtype=jnp.intp)
    values = jnp.arange(100, dtype=jnp.float64)
    d = IntFloatDict(keys, values)
    assert argmin(d) == (0, 0)


def test_to_arrays():
    # Test that an IntFloatDict is converted into arrays
    # of keys and values correctly
    keys_in = jnp.array([1, 2, 3], dtype=jnp.intp)
    values_in = jnp.array([4, 5, 6], dtype=jnp.float64)

    d = IntFloatDict(keys_in, values_in)
    keys_out, values_out = d.to_arrays()

    assert keys_out.dtype == keys_in.dtype
    assert values_in.dtype == values_out.dtype
    assert_array_equal(keys_out, keys_in)
    assert_allclose(values_out, values_in)
