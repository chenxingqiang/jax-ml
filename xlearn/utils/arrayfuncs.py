
# Creating aliases for classes/functions

# Define min_pos
def min_pos(X):
    """Find the minimum value of an array over positive values

    Returns the maximum representable value of the input dtype if none of the
    values are positive.
    """
    min_val = min([x for x in X if x > 0], default=float('inf'))
    return min_val

# Define cholesky_delete
import numpy as np
def cholesky_delete(L, go_out):
    """
    Remove an element from the cholesky factorization

    Parameters
    ----------
    L : 2D array
    go_out : integer
        index of the row to be removed

    Returns
    -------
    The updated lower triangular matrix after row deletion
    """
    # delete row go_out
    L = np.delete(L, go_out, axis=0)

    # Apply Givens rotations
    n = L.shape[0]
    for i in range(go_out, n):
        G, _ = np.linalg.qr(np.eye(n-1, n, k=-i) - np.tril(L))
        L = G @ L

    return L
