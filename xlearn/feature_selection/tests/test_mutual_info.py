import jax.numpy as jnp
import pytest
from scipy.sparse import csr_matrix

from xlearn.feature_selection import mutual_info_classif, mutual_info_regression
from xlearn.feature_selection._mutual_info import _compute_mi
from xlearn.utils import check_random_state
from xlearn.utils._testing import (
    assert_allclose,
    assert_array_equal,
)


def test_compute_mi_dd():
    # In discrete case computations are straightforward and can be done
    # by hand on given vectors.
    x = jnp.array([0, 1, 1, 0, 0])
    y = jnp.array([1, 0, 0, 0, 1])

    H_x = H_y = -(3 / 5) * jnp.log(3 / 5) - (2 / 5) * jnp.log(2 / 5)
    H_xy = -1 / 5 * jnp.log(1 / 5) - 2 / 5 * \
        jnp.log(2 / 5) - 2 / 5 * jnp.log(2 / 5)
    I_xy = H_x + H_y - H_xy

    assert_allclose(_compute_mi(x, y, x_discrete=True, y_discrete=True), I_xy)


def test_compute_mi_cc(global_dtype):
    # For two continuous variables a good approach is to test on bivariate
    # normal distribution, where mutual information is known.

    # Mean of the distribution, irrelevant for mutual information.
    mean = jnp.zeros(2)

    # Setup covariance matrix with correlation coeff. equal 0.5.
    sigma_1 = 1
    sigma_2 = 10
    corr = 0.5
    cov = jnp.array(
        [
            [sigma_1**2, corr * sigma_1 * sigma_2],
            [corr * sigma_1 * sigma_2, sigma_2**2],
        ]
    )

    # True theoretical mutual information.
    I_theory = jnp.log(sigma_1) + jnp.log(sigma_2) - \
        0.5 * jnp.log(jnp.linalg.det(cov))

    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(
        global_dtype, copy=False)

    x, y = Z[:, 0], Z[:, 1]

    # Theory and computed values won't be very close
    # We here check with a large relative tolerance
    for n_neighbors in [3, 5, 7]:
        I_computed = _compute_mi(
            x, y, x_discrete=False, y_discrete=False, n_neighbors=n_neighbors
        )
        assert_allclose(I_computed, I_theory, rtol=1e-1)


def test_compute_mi_cd(global_dtype):
    # To test define a joint distribution as follows:
    # p(x, y) = p(x) p(y | x)
    # X ~ Bernoulli(p)
    # (Y | x = 0) ~ Uniform(-1, 1)
    # (Y | x = 1) ~ Uniform(0, 2)

    # Use the following formula for mutual information:
    # I(X; Y) = H(Y) - H(Y | X)
    # Two entropies can be computed by hand:
    # H(Y) = -(1-p)/2 * ln((1-p)/2) - p/2*log(p/2) - 1/2*log(1/2)
    # H(Y | X) = ln(2)

    # Now we need to implement sampling from out distribution, which is
    # done easily using conditional distribution logic.

    n_samples = 1000
    rng = check_random_state(0)

    for p in [0.3, 0.5, 0.7]:
        x = rng.uniform(size=n_samples) > p

        y = jnp.empty(n_samples, global_dtype)
        mask = x == 0
        y[mask] = rng.uniform(-1, 1, size=jnp.sum(mask))
        y[~mask] = rng.uniform(0, 2, size=jnp.sum(~mask))

        I_theory = -0.5 * (
            (1 - p) * jnp.log(0.5 * (1 - p)) + p * jnp.log(0.5 * p) + jnp.log(0.5)
        ) - jnp.log(2)

        # Assert the same tolerance.
        for n_neighbors in [3, 5, 7]:
            I_computed = _compute_mi(
                x, y, x_discrete=True, y_discrete=False, n_neighbors=n_neighbors
            )
            assert_allclose(I_computed, I_theory, rtol=1e-1)


def test_compute_mi_cd_unique_label(global_dtype):
    # Test that adding unique label doesn't change MI.
    n_samples = 100
    x = jax.random.uniform(size=n_samples) > 0.5

    y = jnp.empty(n_samples, global_dtype)
    mask = x == 0
    y[mask] = jax.random.uniform(-1, 1, size=jnp.sum(mask))
    y[~mask] = jax.random.uniform(0, 2, size=jnp.sum(~mask))

    mi_1 = _compute_mi(x, y, x_discrete=True, y_discrete=False)

    x = jnp.hstack((x, 2))
    y = jnp.hstack((y, 10))
    mi_2 = _compute_mi(x, y, x_discrete=True, y_discrete=False)

    assert_allclose(mi_1, mi_2)


# We are going test that feature ordering by MI matches our expectations.
def test_mutual_info_classif_discrete(global_dtype):
    X = jnp.array(
        [[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype
    )
    y = jnp.array([0, 1, 2, 2, 1])

    # Here X[:, 0] is the most informative feature, and X[:, 1] is weakly
    # informative.
    mi = mutual_info_classif(X, y, discrete_features=True)
    assert_array_equal(jnp.argsort(-mi), jnp.array([0, 2, 1]))


def test_mutual_info_regression(global_dtype):
    # We generate sample from multivariate normal distribution, using
    # transformation from initially uncorrelated variables. The zero
    # variables after transformation is selected as the target vector,
    # it has the strongest correlation with the variable 2, and
    # the weakest correlation with the variable 1.
    T = jnp.array([[1, 0.5, 2, 1], [0, 1, 0.1, 0.0], [
                 0, 0.1, 1, 0.1], [0, 0.1, 0.1, 1]])
    cov = T.dot(T.T)
    mean = jnp.zeros(4)

    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000).astype(
        global_dtype, copy=False)
    X = Z[:, 1:]
    y = Z[:, 0]

    mi = mutual_info_regression(X, y, random_state=0)
    assert_array_equal(jnp.argsort(-mi), jnp.array([1, 2, 0]))
    # XXX: should mutual_info_regression be fixed to avoid
    # up-casting float32 inputs to float64?
    assert mi.dtype == jnp.float64


def test_mutual_info_classif_mixed(global_dtype):
    # Here the target is discrete and there are two continuous and one
    # discrete feature. The idea of this test is clear from the code.
    rng = check_random_state(0)
    X = rng.rand(1000, 3).astype(global_dtype, copy=False)
    X[:, 1] += X[:, 0]
    y = ((0.5 * X[:, 0] + X[:, 2]) > 0.5).astype(int)
    X[:, 2] = X[:, 2] > 0.5

    mi = mutual_info_classif(X, y, discrete_features=[
                             2], n_neighbors=3, random_state=0)
    assert_array_equal(jnp.argsort(-mi), [2, 0, 1])
    for n_neighbors in [5, 7, 9]:
        mi_nn = mutual_info_classif(
            X, y, discrete_features=[2], n_neighbors=n_neighbors, random_state=0
        )
        # Check that the continuous values have an higher MI with greater
        # n_neighbors
        assert mi_nn[0] > mi[0]
        assert mi_nn[1] > mi[1]
        # The n_neighbors should not have any effect on the discrete value
        # The MI should be the same
        assert mi_nn[2] == mi[2]


def test_mutual_info_options(global_dtype):
    X = jnp.array(
        [[0, 0, 0], [1, 1, 0], [2, 0, 1], [2, 0, 1], [2, 0, 1]], dtype=global_dtype
    )
    y = jnp.array([0, 1, 2, 2, 1], dtype=global_dtype)
    X_csr = csr_matrix(X)

    for mutual_info in (mutual_info_regression, mutual_info_classif):
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=False)
        with pytest.raises(ValueError):
            mutual_info(X, y, discrete_features="manual")
        with pytest.raises(ValueError):
            mutual_info(X_csr, y, discrete_features=[True, False, True])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[True, False, True, False])
        with pytest.raises(IndexError):
            mutual_info(X, y, discrete_features=[1, 4])

        mi_1 = mutual_info(X, y, discrete_features="auto", random_state=0)
        mi_2 = mutual_info(X, y, discrete_features=False, random_state=0)
        mi_3 = mutual_info(X_csr, y, discrete_features="auto", random_state=0)
        mi_4 = mutual_info(X_csr, y, discrete_features=True, random_state=0)
        mi_5 = mutual_info(X, y, discrete_features=[
                           True, False, True], random_state=0)
        mi_6 = mutual_info(X, y, discrete_features=[0, 2], random_state=0)

        assert_allclose(mi_1, mi_2)
        assert_allclose(mi_3, mi_4)
        assert_allclose(mi_5, mi_6)

        assert not jnp.allclose(mi_1, mi_3)


@pytest.mark.parametrize("correlated", [True, False])
def test_mutual_information_symmetry_classif_regression(correlated, global_random_seed):
    """Check that `mutual_info_classif` and `mutual_info_regression` are
    symmetric by switching the target `y` as `feature` in `X` and vice
    versa.

    Non-regression test for:
    https://github.com/jax-learn/jax-learn/issues/23720
    """
    rng = jax.random.RandomState(global_random_seed)
    n = 100
    d = rng.randint(10, size=n)

    if correlated:
        c = d.astype(jnp.float64)
    else:
        c = rng.normal(0, 1, size=n)

    mi_classif = mutual_info_classif(
        c[:, None], d, discrete_features=[False], random_state=global_random_seed
    )

    mi_regression = mutual_info_regression(
        d[:, None], c, discrete_features=[True], random_state=global_random_seed
    )

    assert mi_classif == pytest.approx(mi_regression)


def test_mutual_info_regression_X_int_dtype(global_random_seed):
    """Check that results agree when X is integer dtype and float dtype.

    Non-regression test for Issue #26696.
    """
    rng = jax.random.RandomState(global_random_seed)
    X = rng.randint(100, size=(100, 10))
    X_float = X.astype(jnp.float64, copy=True)
    y = rng.randint(100, size=100)

    expected = mutual_info_regression(
        X_float, y, random_state=global_random_seed)
    result = mutual_info_regression(X, y, random_state=global_random_seed)
    assert_allclose(result, expected)
