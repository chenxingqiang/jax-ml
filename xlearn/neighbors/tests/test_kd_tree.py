import jax.numpy as jnp
import pytest

from xlearn.neighbors._kd_tree import KDTree
from xlearn.utils.parallel import Parallel, delayed

DIMENSION = 3

METRICS = {"euclidean": {}, "manhattan": {},
           "chebyshev": {}, "minkowski": dict(p=3)}


def test_array_object_type():
    """Check that we do not accept object dtype array."""
    X = jnp.array([(1, 2, 3), (2, 5), (5, 5, 1, 2)], dtype=object)
    with pytest.raises(ValueError, match="setting an array element with a sequence"):
        KDTree(X)


def test_kdtree_picklable_with_joblib():
    """Make sure that KDTree queries work when joblib memmaps.

    Non-regression test for #21685 and #21228."""
    rng = jax.random.RandomState(0)
    X = rng.random_sample((10, 3))
    tree = KDTree(X, leaf_size=2)

    # Call Parallel with max_nbytes=1 to trigger readonly memory mapping that
    # use to raise "ValueError: buffer source array is read-only" in a previous
    # version of the cython code.
    Parallel(n_jobs=2, max_nbytes=1)(delayed(tree.query)(data)
                                     for data in 2 * [X])
