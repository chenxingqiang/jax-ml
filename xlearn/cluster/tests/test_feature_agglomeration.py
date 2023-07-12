"""
Tests for xlearn.cluster._feature_agglomeration
"""
# Authors: Sergul Aydore 2017
import warnings

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from xlearn.cluster import FeatureAgglomeration
from xlearn.datasets import make_blobs
from xlearn.utils._testing import assert_array_almost_equal


def test_feature_agglomeration():
    n_clusters = 1
    X = jnp.array([0, 0, 1]).reshape(1, 3)  # (n_samples, n_features)

    agglo_mean = FeatureAgglomeration(
        n_clusters=n_clusters, pooling_func=jnp.mean)
    agglo_median = FeatureAgglomeration(
        n_clusters=n_clusters, pooling_func=jnp.median)
    agglo_mean.fit(X)
    agglo_median.fit(X)

    assert jnp.size(jnp.unique(agglo_mean.labels_)) == n_clusters
    assert jnp.size(jnp.unique(agglo_median.labels_)) == n_clusters
    assert jnp.size(agglo_mean.labels_) == X.shape[1]
    assert jnp.size(agglo_median.labels_) == X.shape[1]

    # Test transform
    Xt_mean = agglo_mean.transform(X)
    Xt_median = agglo_median.transform(X)
    assert Xt_mean.shape[1] == n_clusters
    assert Xt_median.shape[1] == n_clusters
    assert Xt_mean == jnp.array([1 / 3.0])
    assert Xt_median == jnp.array([0.0])

    # Test inverse transform
    X_full_mean = agglo_mean.inverse_transform(Xt_mean)
    X_full_median = agglo_median.inverse_transform(Xt_median)
    assert jnp.unique(X_full_mean[0]).size == n_clusters
    assert jnp.unique(X_full_median[0]).size == n_clusters

    assert_array_almost_equal(agglo_mean.transform(X_full_mean), Xt_mean)
    assert_array_almost_equal(agglo_median.transform(X_full_median), Xt_median)


def test_feature_agglomeration_feature_names_out():
    """Check `get_feature_names_out` for `FeatureAgglomeration`."""
    X, _ = make_blobs(n_features=6, random_state=0)
    agglo = FeatureAgglomeration(n_clusters=3)
    agglo.fit(X)
    n_clusters = agglo.n_clusters_

    names_out = agglo.get_feature_names_out()
    assert_array_equal(
        [f"featureagglomeration{i}" for i in range(n_clusters)], names_out
    )


# TODO(1.5): remove this test
def test_inverse_transform_Xred_deprecation():
    X = jnp.array([0, 0, 1]).reshape(1, 3)  # (n_samples, n_features)

    est = FeatureAgglomeration(n_clusters=1, pooling_func=jnp.mean)
    est.fit(X)
    Xt = est.transform(X)

    with pytest.raises(TypeError, match="Missing required positional argument"):
        est.inverse_transform()

    with pytest.raises(ValueError, match="Please provide only"):
        est.inverse_transform(Xt=Xt, Xred=Xt)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error")
        est.inverse_transform(Xt)

    with pytest.warns(FutureWarning, match="Input argument `Xred` was renamed to `Xt`"):
        est.inverse_transform(Xred=Xt)