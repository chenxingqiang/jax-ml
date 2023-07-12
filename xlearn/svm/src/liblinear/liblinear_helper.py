import sys
from typing import List
from linear_h import set_seed, set_print_string_function
import jax
import jax.numpy as jnp
from typing import List

def dense_to_sparse(x: jnp.ndarray, double_precision: bool, bias: float) -> List[List[float]]:
    n_samples, n_features = x.shape
    n_nonzero = jnp.count_nonzero(x)
    have_bias = bias > 0

    sparse = []
    T = []
    for i in range(n_samples):
        sparse.append(T)

        for j in range(n_features):
            if double_precision:
                if x[i, j] != 0:
                    T.append([j + 1, x[i, j]])
            else:
                if x[i, j] != 0:
                    T.append([j + 1, x[i, j]])

        if have_bias:
            T.append([n_features + 1, bias])

        T.append([-1, 0])

    return sparse

def csr_to_sparse(x: jnp.ndarray, indices: jnp.ndarray, indptr: jnp.ndarray, double_precision: bool, bias: float) -> List[List[float]]:
    n_samples, n_features = x.shape
    n_nonzero = indices.shape[0]
    have_bias = bias > 0

    sparse = []
    T = []
    k = 0
    for i in range(n_samples):
        sparse.append(T)
        n = indptr[i + 1] - indptr[i]

        for j in range(n):
            T.append([indices[k] + 1, x[k]])
            k += 1

        if have_bias:
            T.append([n_features + 1, bias])

        T.append([-1, 0])

    return sparse

def set_problem(X: jnp.ndarray, double_precision_X: bool, bias: float, sample_weight: jnp.ndarray, Y: jnp.ndarray):
    n_samples, n_features = X.shape
    n_nonzero = jnp.count_nonzero(X)

    problem = {
        'l': n_samples,
        'n': n_features + (bias > 0),
        'y': Y,
        'W': sample_weight,
        'x': dense_to_sparse(X, double_precision_X, bias),
        'bias': bias
    }

    return problem

def csr_set_problem(X: jnp.ndarray, double_precision_X: bool, indices: jnp.ndarray, indptr: jnp.ndarray, bias: float, sample_weight: jnp.ndarray, Y: jnp.ndarray):
    n_samples, n_features = X.shape
    n_nonzero = indices.shape[0]

    problem = {
        'l': n_samples,
        'n': n_features + (bias > 0),
        'y': Y,
        'W': sample_weight,
        'x': csr_to_sparse(X, indices, indptr, double_precision_X, bias),
        'bias': bias
    }

    return problem

def set_parameter(solver_type: int, eps: float, C: float, nr_weight: int, weight_label: jnp.ndarray, weight: jnp.ndarray, max_iter: int, seed: int, epsilon: float):
    param = {
        'solver_type': solver_type,
        'eps': eps,
        'C': C,
        'p': epsilon,
        'nr_weight': nr_weight,
        'weight_label': weight_label,
        'weight': weight,
        'max_iter': max_iter
    }

    jax.random.seed(seed)

    return param

def copy_w(data: jnp.ndarray, model: dict, length: int):
    data[:length] = model['w'][:length]

def get_bias(model: dict):
    return model['bias']

def free_problem(problem: dict):
    del problem['x']
    del problem

def free_parameter(param: dict):
    del param


def print_null(s: str):
    pass

def print_string_stdout(s: str):
    print(s, end='')
    sys.stdout.flush()

def set_verbosity(verbosity_flag: bool):
    if verbosity_flag:
        set_print_string_function(print_string_stdout)
    else:
        set_print_string_function(print_null)
