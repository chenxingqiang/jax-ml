import warnings

import jax.numpy as jnp
import pytest
from scipy import sparse

from xlearn.base import clone
from xlearn.datasets import load_iris
from xlearn.model_selection import train_test_split
from xlearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    maxabs_scale,
    minmax_scale,
    power_transform,
    quantile_transform,
    robust_scale,
    scale,
)
from xlearn.utils._testing import assert_allclose, assert_array_equal

iris = load_iris()


def _get_valid_samples_by_column(X, col):
    """Get non NaN samples in column of X"""
    return X[:, [col]][~jnp.isnan(X[:, col])]


@pytest.mark.parametrize(
    "est, func, support_sparse, strictly_positive, omit_kwargs",
    [
        (MaxAbsScaler(), maxabs_scale, True, False, []),
        (MinMaxScaler(), minmax_scale, False, False, ["clip"]),
        (StandardScaler(), scale, False, False, []),
        (StandardScaler(with_mean=False), scale, True, False, []),
        (PowerTransformer("yeo-johnson"), power_transform, False, False, []),
        (PowerTransformer("box-cox"), power_transform, False, True, []),
        (QuantileTransformer(n_quantiles=10), quantile_transform, True, False, []),
        (RobustScaler(), robust_scale, False, False, []),
        (RobustScaler(with_centering=False), robust_scale, True, False, []),
    ],
)
def test_missing_value_handling(
    est, func, support_sparse, strictly_positive, omit_kwargs
):
    # check that the preprocessing method let pass nan
    rng = np.random.RandomState(42)
    X = iris.data.copy()
    n_missing = 50
    X[
        rng.randint(X.shape[0], size=n_missing), rng.randint(
            X.shape[1], size=n_missing)
    ] = jnp.nan
    if strictly_positive:
        X += jnp.nanmin(X) + 0.1
    X_train, X_test = train_test_split(X, random_state=1)
    # sanity check
    assert not jnp.all(jnp.isnan(X_train), axis=0).any()
    assert jnp.any(jnp.isnan(X_train), axis=0).all()
    assert jnp.any(jnp.isnan(X_test), axis=0).all()
    X_test[:, 0] = jnp.nan  # make sure this boundary case is tested

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Xt = est.fit(X_train).transform(X_test)
    # ensure no warnings are raised
    # missing values should still be missing, and only them
    assert_array_equal(jnp.isnan(Xt), jnp.isnan(X_test))

    # check that the function leads to the same results as the class
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Xt_class = est.transform(X_train)
    kwargs = est.get_params()
    # remove the parameters which should be omitted because they
    # are not defined in the counterpart function of the preprocessing class
    for kwarg in omit_kwargs:
        _ = kwargs.pop(kwarg)
    Xt_func = func(X_train, **kwargs)
    assert_array_equal(jnp.isnan(Xt_func), jnp.isnan(Xt_class))
    assert_allclose(Xt_func[~jnp.isnan(Xt_func)], Xt_class[~jnp.isnan(Xt_class)])

    # check that the inverse transform keep NaN
    Xt_inv = est.inverse_transform(Xt)
    assert_array_equal(jnp.isnan(Xt_inv), jnp.isnan(X_test))
    # FIXME: we can introduce equal_nan=True in recent version of numpy.
    # For the moment which just check that non-NaN values are almost equal.
    assert_allclose(Xt_inv[~jnp.isnan(Xt_inv)], X_test[~jnp.isnan(X_test)])

    for i in range(X.shape[1]):
        # train only on non-NaN
        est.fit(_get_valid_samples_by_column(X_train, i))
        # check transforming with NaN works even when training without NaN
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            Xt_col = est.transform(X_test[:, [i]])
        assert_allclose(Xt_col, Xt[:, [i]])
        # check non-NaN is handled as before - the 1st column is all nan
        if not jnp.isnan(X_test[:, i]).all():
            Xt_col_nonan = est.transform(
                _get_valid_samples_by_column(X_test, i))
            assert_array_equal(
                Xt_col_nonan, Xt_col[~jnp.isnan(Xt_col.squeeze())])

    if support_sparse:
        est_dense = clone(est)
        est_sparse = clone(est)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            Xt_dense = est_dense.fit(X_train).transform(X_test)
            Xt_inv_dense = est_dense.inverse_transform(Xt_dense)

        for sparse_constructor in (
            sparse.csr_matrix,
            sparse.csc_matrix,
            sparse.bsr_matrix,
            sparse.coo_matrix,
            sparse.dia_matrix,
            sparse.dok_matrix,
            sparse.lil_matrix,
        ):
            # check that the dense and sparse inputs lead to the same results
            # precompute the matrix to avoid catching side warnings
            X_train_sp = sparse_constructor(X_train)
            X_test_sp = sparse_constructor(X_test)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PendingDeprecationWarning)
                warnings.simplefilter("error", RuntimeWarning)
                Xt_sp = est_sparse.fit(X_train_sp).transform(X_test_sp)

            assert_allclose(Xt_sp.A, Xt_dense)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PendingDeprecationWarning)
                warnings.simplefilter("error", RuntimeWarning)
                Xt_inv_sp = est_sparse.inverse_transform(Xt_sp)

            assert_allclose(Xt_inv_sp.A, Xt_inv_dense)


@pytest.mark.parametrize(
    "est, func",
    [
        (MaxAbsScaler(), maxabs_scale),
        (MinMaxScaler(), minmax_scale),
        (StandardScaler(), scale),
        (StandardScaler(with_mean=False), scale),
        (PowerTransformer("yeo-johnson"), power_transform),
        (
            PowerTransformer("box-cox"),
            power_transform,
        ),
        (QuantileTransformer(n_quantiles=3), quantile_transform),
        (RobustScaler(), robust_scale),
        (RobustScaler(with_centering=False), robust_scale),
    ],
)
def test_missing_value_pandas_na_support(est, func):
    # Test pandas IntegerArray with pd.NA
    pd = pytest.importorskip("pandas")

    X = jnp.array(
        [
            [1, 2, 3, jnp.nan, jnp.nan, 4, 5, 1],
            [jnp.nan, jnp.nan, 8, 4, 6, jnp.nan, jnp.nan, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]
    ).T

    # Creates dataframe with IntegerArrays with pd.NA
    X_df = pd.DataFrame(X, dtype="Int16", columns=["a", "b", "c"])
    X_df["c"] = X_df["c"].astype("int")

    X_trans = est.fit_transform(X)
    X_df_trans = est.fit_transform(X_df)

    assert_allclose(X_trans, X_df_trans)
