import jax
import jax.numpy as jnp

def add(x, w, c):
    """Scales sample x by constant c and adds it to the weight vector."""
    return w + x * c

def add_average(x, w, c, num_iter):
    """Updates the average weight vector."""
    average_a = 0.0
    average_b = 1.0
    mu = 1.0 / num_iter
    for i in range(len(x)):
        w[i] += average_a * x[i] * (-c / w[i])
    average_b /= (1.0 - mu)
    average_a += mu * average_b * w[i]
    return w

def dot(x, w):
    """Computes the dot product of a sample x and the weight vector."""
    return jnp.dot(x, w)

def scale(w, c):
    """Scales the weight vector by a constant c."""
    return w * c

def reset_scale(w):
    """Scales each coef of w by wscale and resets it to 1."""
    w = w * wscale
    wscale = 1.0
    return w

def norm(w):
    """The L2 norm of the weight vector."""
    return jnp.sqrt(jnp.sum(w**2))
