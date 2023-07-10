import pickle

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from xlearn.utils._encode import _check_unknown, _encode, _get_counts, _unique


@pytest.mark.parametrize(
    "values, expected",
    [
        (jnp.array([2, 1, 3, 1, 3], dtype="int64"),
         jnp.array([1, 2, 3], dtype="int64")),
        (
            jnp.array([2, 1, jnp.nan, 1, jnp.nan], dtype="float32"),
            jnp.array([1, 2, jnp.nan], dtype="float32"),
        ),
        (
            jnp.array(["b", "a", "c", "a", "c"], dtype=object),
            jnp.array(["a", "b", "c"], dtype=object),
        ),
        (
            jnp.array(["b", "a", None, "a", None], dtype=object),
            jnp.array(["a", "b", None], dtype=object),
        ),
        (jnp.array(["b", "a", "c", "a", "c"]), jnp.array(["a", "b", "c"])),
    ],
    ids=["int64", "float32-nan", "object", "object-None", "str"],
)
def test_encode_util(values, expected):
    uniques = _unique(values)
    assert_array_equal(uniques, expected)

    result, encoded = _unique(values, return_inverse=True)
    assert_array_equal(result, expected)
    assert_array_equal(encoded, jnp.array([1, 0, 2, 0, 2]))

    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, jnp.array([1, 0, 2, 0, 2]))

    result, counts = _unique(values, return_counts=True)
    assert_array_equal(result, expected)
    assert_array_equal(counts, jnp.array([2, 1, 2]))

    result, encoded, counts = _unique(
        values, return_inverse=True, return_counts=True)
    assert_array_equal(result, expected)
    assert_array_equal(encoded, jnp.array([1, 0, 2, 0, 2]))
    assert_array_equal(counts, jnp.array([2, 1, 2]))


def test_encode_with_check_unknown():
    # test for the check_unknown parameter of _encode()
    uniques = jnp.array([1, 2, 3])
    values = jnp.array([1, 2, 3, 4])

    # Default is True, raise error
    with pytest.raises(ValueError, match="y contains previously unseen labels"):
        _encode(values, uniques=uniques, check_unknown=True)

    # dont raise error if False
    _encode(values, uniques=uniques, check_unknown=False)

    # parameter is ignored for object dtype
    uniques = jnp.array(["a", "b", "c"], dtype=object)
    values = jnp.array(["a", "b", "c", "d"], dtype=object)
    with pytest.raises(ValueError, match="y contains previously unseen labels"):
        _encode(values, uniques=uniques, check_unknown=False)


def _assert_check_unknown(values, uniques, expected_diff, expected_mask):
    diff = _check_unknown(values, uniques)
    assert_array_equal(diff, expected_diff)

    diff, valid_mask = _check_unknown(values, uniques, return_mask=True)
    assert_array_equal(diff, expected_diff)
    assert_array_equal(valid_mask, expected_mask)


@pytest.mark.parametrize(
    "values, uniques, expected_diff, expected_mask",
    [
        (jnp.array([1, 2, 3, 4]), jnp.array([1, 2, 3]),
         [4], [True, True, True, False]),
        (jnp.array([2, 1, 4, 5]), jnp.array([2, 5, 1]),
         [4], [True, True, False, True]),
        (jnp.array([2, 1, jnp.nan]), jnp.array(
            [2, 5, 1]), [jnp.nan], [True, True, False]),
        (
            jnp.array([2, 1, 4, jnp.nan]),
            jnp.array([2, 5, 1, jnp.nan]),
            [4],
            [True, True, False, True],
        ),
        (
            jnp.array([2, 1, 4, jnp.nan]),
            jnp.array([2, 5, 1]),
            [4, jnp.nan],
            [True, True, False, False],
        ),
        (
            jnp.array([2, 1, 4, 5]),
            jnp.array([2, 5, 1, jnp.nan]),
            [4],
            [True, True, False, True],
        ),
        (
            jnp.array(["a", "b", "c", "d"], dtype=object),
            jnp.array(["a", "b", "c"], dtype=object),
            jnp.array(["d"], dtype=object),
            [True, True, True, False],
        ),
        (
            jnp.array(["d", "c", "a", "b"], dtype=object),
            jnp.array(["a", "c", "b"], dtype=object),
            jnp.array(["d"], dtype=object),
            [False, True, True, True],
        ),
        (
            jnp.array(["a", "b", "c", "d"]),
            jnp.array(["a", "b", "c"]),
            jnp.array(["d"]),
            [True, True, True, False],
        ),
        (
            jnp.array(["d", "c", "a", "b"]),
            jnp.array(["a", "c", "b"]),
            jnp.array(["d"]),
            [False, True, True, True],
        ),
    ],
)
def test_check_unknown(values, uniques, expected_diff, expected_mask):
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)


@pytest.mark.parametrize("missing_value", [None, jnp.nan, float("nan")])
@pytest.mark.parametrize("pickle_uniques", [True, False])
def test_check_unknown_missing_values(missing_value, pickle_uniques):
    # check for check_unknown with missing values with object dtypes
    values = jnp.array(["d", "c", "a", "b", missing_value], dtype=object)
    uniques = jnp.array(["c", "a", "b", missing_value], dtype=object)
    if pickle_uniques:
        uniques = pickle.loads(pickle.dumps(uniques))

    expected_diff = ["d"]
    expected_mask = [False, True, True, True, True]
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)

    values = jnp.array(["d", "c", "a", "b", missing_value], dtype=object)
    uniques = jnp.array(["c", "a", "b"], dtype=object)
    if pickle_uniques:
        uniques = pickle.loads(pickle.dumps(uniques))

    expected_diff = ["d", missing_value]

    expected_mask = [False, True, True, True, False]
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)

    values = jnp.array(["a", missing_value], dtype=object)
    uniques = jnp.array(["a", "b", "z"], dtype=object)
    if pickle_uniques:
        uniques = pickle.loads(pickle.dumps(uniques))

    expected_diff = [missing_value]
    expected_mask = [True, False]
    _assert_check_unknown(values, uniques, expected_diff, expected_mask)


@pytest.mark.parametrize("missing_value", [jnp.nan, None, float("nan")])
@pytest.mark.parametrize("pickle_uniques", [True, False])
def test_unique_util_missing_values_objects(missing_value, pickle_uniques):
    # check for _unique and _encode with missing values with object dtypes
    values = jnp.array(["a", "c", "c", missing_value, "b"], dtype=object)
    expected_uniques = jnp.array(["a", "b", "c", missing_value], dtype=object)

    uniques = _unique(values)

    if missing_value is None:
        assert_array_equal(uniques, expected_uniques)
    else:  # missing_value == jnp.nan
        assert_array_equal(uniques[:-1], expected_uniques[:-1])
        assert jnp.isnan(uniques[-1])

    if pickle_uniques:
        uniques = pickle.loads(pickle.dumps(uniques))

    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, jnp.array([0, 2, 2, 3, 1]))


def test_unique_util_missing_values_numeric():
    # Check missing values in numerical values
    values = jnp.array([3, 1, jnp.nan, 5, 3, jnp.nan], dtype=float)
    expected_uniques = jnp.array([1, 3, 5, jnp.nan], dtype=float)
    expected_inverse = jnp.array([1, 0, 3, 2, 1, 3])

    uniques = _unique(values)
    assert_array_equal(uniques, expected_uniques)

    uniques, inverse = _unique(values, return_inverse=True)
    assert_array_equal(uniques, expected_uniques)
    assert_array_equal(inverse, expected_inverse)

    encoded = _encode(values, uniques=uniques)
    assert_array_equal(encoded, expected_inverse)


def test_unique_util_with_all_missing_values():
    # test for all types of missing values for object dtype
    values = jnp.array([jnp.nan, "a", "c", "c", None,
                      float("nan"), None], dtype=object)

    uniques = _unique(values)
    assert_array_equal(uniques[:-1], ["a", "c", None])
    # last value is nan
    assert jnp.isnan(uniques[-1])

    expected_inverse = [3, 0, 1, 1, 2, 3, 2]
    _, inverse = _unique(values, return_inverse=True)
    assert_array_equal(inverse, expected_inverse)


def test_check_unknown_with_both_missing_values():
    # test for both types of missing values for object dtype
    values = jnp.array([jnp.nan, "a", "c", "c", None,
                      jnp.nan, None], dtype=object)

    diff = _check_unknown(
        values, known_values=jnp.array(["a", "c"], dtype=object))
    assert diff[0] is None
    assert jnp.isnan(diff[1])

    diff, valid_mask = _check_unknown(
        values, known_values=jnp.array(["a", "c"], dtype=object), return_mask=True
    )

    assert diff[0] is None
    assert jnp.isnan(diff[1])
    assert_array_equal(
        valid_mask, [False, True, True, True, False, False, False])


@pytest.mark.parametrize(
    "values, uniques, expected_counts",
    [
        (jnp.array([1] * 10 + [2] * 4 + [3] * 15),
         jnp.array([1, 2, 3]), [10, 4, 15]),
        (
            jnp.array([1] * 10 + [2] * 4 + [3] * 15),
            jnp.array([1, 2, 3, 5]),
            [10, 4, 15, 0],
        ),
        (
            jnp.array([jnp.nan] * 10 + [2] * 4 + [3] * 15),
            jnp.array([2, 3, jnp.nan]),
            [4, 15, 10],
        ),
        (
            jnp.array(["b"] * 4 + ["a"] * 16 + ["c"] * 20, dtype=object),
            ["a", "b", "c"],
            [16, 4, 20],
        ),
        (
            jnp.array(["b"] * 4 + ["a"] * 16 + ["c"] * 20, dtype=object),
            ["c", "b", "a"],
            [20, 4, 16],
        ),
        (
            jnp.array([jnp.nan] * 4 + ["a"] * 16 + ["c"] * 20, dtype=object),
            ["c", jnp.nan, "a"],
            [20, 4, 16],
        ),
        (
            jnp.array(["b"] * 4 + ["a"] * 16 + ["c"] * 20, dtype=object),
            ["a", "b", "c", "e"],
            [16, 4, 20, 0],
        ),
    ],
)
def test_get_counts(values, uniques, expected_counts):
    counts = _get_counts(values, uniques)
    assert_array_equal(counts, expected_counts)
