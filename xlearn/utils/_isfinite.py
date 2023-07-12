import jax.numpy as jnp
from enum import Enum

class FiniteStatus(Enum):
    all_finite = 0
    has_nan = 1
    has_infinite = 2

def isfinite(a, allow_nan=False):
    if allow_nan:
        return isfinite_allow_nan(a)
    else:
        return isfinite_disable_nan(a)

def isfinite_allow_nan(a):
    if jnp.any(jnp.isinf(a)):
        return FiniteStatus.has_infinite
    return FiniteStatus.all_finite

def isfinite_disable_nan(a):
    if jnp.any(jnp.isnan(a)):
        return FiniteStatus.has_nan
    elif jnp.any(jnp.isinf(a)):
        return FiniteStatus.has_infinite
    return FiniteStatus.all_finite
import jax.numpy as jnp
from enum import Enum

class FiniteStatus(Enum):
    all_finite = 0
    has_nan = 1
    has_infinite = 2

def isfinite(a, allow_nan=False):
    if allow_nan:
        return isfinite_allow_nan(a)
    else:
        return isfinite_disable_nan(a)

def isfinite_allow_nan(a):
    if jnp.any(jnp.isinf(a)):
        return FiniteStatus.has_infinite
    return FiniteStatus.all_finite

def isfinite_disable_nan(a):
    if jnp.any(jnp.isnan(a)):
        return FiniteStatus.has_nan
    elif jnp.any(jnp.isinf(a)):
        return FiniteStatus.has_infinite
    return FiniteStatus.all_finite
