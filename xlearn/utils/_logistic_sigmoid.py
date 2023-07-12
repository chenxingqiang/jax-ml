import jax.numpy as jnp

def log_logistic_sigmoid(X):
    """Compute the log of the logistic sigmoid function for each element in X.

    Parameters
    ----------
    X : jnp.ndarray, shape (n_samples, n_features)
        Input array.

    Returns
    -------
    jnp.ndarray
        Output array, where each element is the log of the logistic sigmoid
        function applied to the corresponding element in X.
    """
    return -jnp.logaddexp(0, -X)
