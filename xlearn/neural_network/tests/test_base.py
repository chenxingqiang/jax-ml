import jax.numpy as jnp
import pytest

from xlearn.neural_network._base import binary_log_loss, log_loss


def test_binary_log_loss_1_prob_finite():
    # y_proba is equal to one should result in a finite logloss
    y_true = jnp.array([[0, 0, 1]]).T
    y_prob = jnp.array([[0.9, 1.0, 1.0]]).T

    loss = binary_log_loss(y_true, y_prob)
    assert jnp.isfinite(loss)


@pytest.mark.parametrize(
    "y_true, y_prob",
    [
        (
            jnp.array([[1, 0, 0], [0, 1, 0]]),
            jnp.array([[0.0, 1.0, 0.0], [0.9, 0.05, 0.05]]),
        ),
        (jnp.array([[0, 0, 1]]).T, jnp.array([[0.9, 1.0, 1.0]]).T),
    ],
)
def test_log_loss_1_prob_finite(y_true, y_prob):
    # y_proba is equal to 1 should result in a finite logloss
    loss = log_loss(y_true, y_prob)
    assert jnp.isfinite(loss)
