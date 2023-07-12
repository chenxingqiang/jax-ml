from typing import Optional, Union
import jax.numpy as jnp
import numpy as np
from numpy.random import RandomState
from enum import Enum

class Method(Enum):
    AUTO = "auto"
    TRACKING_SELECTION = "tracking_selection"
    RESERVOIR_SAMPLING = "reservoir_sampling"
    POOL = "pool"


class UINT32:
    def __init__(self, value):
        self.value = np.uint32(value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if isinstance(value, (np.uint32, np.int32)):
            self._value = np.uint32(value)
        else:
            raise ValueError('Expected a uint32 value, got %s.' % type(value))


DEFAULT_SEED = UINT32(1)


def check_random_state(seed):
    """Turn seed into a jax.random.RandomState instance

    If seed is None (or jax.random), return the RandomState singleton used
    by jax.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is jax.random:
        return jax.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return jax.random.RandomState(seed)
    if isinstance(seed, jax.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _sample_without_replacement_check_input(n_population: np.int32, n_samples: np.int32):
    """ Check that input are consistent for sample_without_replacement"""
    if n_population < 0:
        raise ValueError('n_population should be greater than 0, got %s.'
                         % n_population)

    if n_samples > n_population:
        raise ValueError('n_population should be greater or equal than '
                         'n_samples, got n_samples > n_population (%s > %s)'
                         % (n_samples, n_population))


def _sample_without_replacement_with_tracking_selection(n_population: np.int32,
                                                       n_samples: np.int32,
                                                       random_state=None):
    _sample_without_replacement_check_input(n_population, n_samples)

    rng = check_random_state(random_state)
    rng_randint = rng.randint

    selected = set()

    for i in range(n_samples):
        j = rng_randint(n_population)
        while j in selected:
            j = rng_randint(n_population)
        selected.add(j)

    return jnp.asarray(list(selected))


def _sample_without_replacement_with_pool(n_population: np.int32,
                                          n_samples: np.int32,
                                          random_state=None):
    _sample_without_replacement_check_input(n_population, n_samples)

    rng = check_random_state(random_state)
    rng_randint = rng.randint

    pool = jnp.arange(n_population, dtype=int)

    for i in range(n_samples):
        j = rng_randint(n_population - i)
        pool[j] = pool[n_population - i - 1]

    return pool[:n_samples]


def _sample_without_replacement_with_reservoir_sampling(n_population: np.int32,
                                                        n_samples: np.int32,
                                                        random_state=None):
    _sample_without_replacement_check_input(n_population, n_samples)

    rng = check_random_state(random_state)
    rng_randint = rng.randint

    reservoir = jnp.arange(n_samples, dtype=int)

    for i in range(n_samples, n_population):
        j = rng_randint(0, i + 1)
        if j < n_samples:
            reservoir[j] = i

    return reservoir


def sample_without_replacement(n_population: np.int32,
                               n_samples: np.int32,
                               method: Union[str, Method] = "auto",
                               random_state=None):
    _sample_without_replacement_check_input(n_population, n_samples)

    all_methods = [method.value for method in Method]
    if isinstance(method, str):
        method = Method(method)

    ratio = n_samples / n_population if n_population != 0 else 1.0

    if method == Method.AUTO and ratio > 0.01 and ratio < 0.99:
        rng = check_random_state(random_state)
        return rng.permutation(n_population)[:n_samples]

    elif method in [Method.AUTO, Method.TRACKING_SELECTION]:
        if ratio < 0.2:
            return _sample_without_replacement_with_tracking_selection(
                n_population, n_samples, random_state)
        else:
            return _sample_without_replacement_with_reservoir_sampling(
                n_population, n_samples, random_state)

    elif method == Method.RESERVOIR_SAMPLING:
        return _sample_without_replacement_with_reservoir_sampling(
            n_population, n_samples, random_state)

    elif method == Method.POOL:
        return _sample_without_replacement_with_pool(n_population, n_samples,
                                                     random_state)
    else:
        raise ValueError('Expected a method name in %s, got %s. '
                         % (all_methods, method))


def our_rand_r(seed: UINT32) -> UINT32:
    if seed.value == 0:
        seed.value = DEFAULT_SEED.value

    seed.value ^= seed.value << 13
    seed.value ^= seed.value >> 17
    seed.value ^= seed.value << 5

    return seed


def _our_rand_r_py(seed: UINT32) -> UINT32:
    return our_rand_r(seed)
