import os
import jax
from jax import jit
from jax.interpreters import xla

# Module level cache for cpu_count as we do not expect this to change during
# the lifecle of a Python program. This dictionary is keyed by
# only_physical_cores.
_CPU_COUNTS = {}

def _openmp_parallelism_enabled():
    """Determines whether jax-learn has been built with OpenMP
    It allows to retrieve at runtime the information gathered at compile time.
    """
    # JAX doesn't support OpenMP, but parallelism can be managed at a higher level using
    # the `pmap` function for example. So, this function will always return False.
    return False

def _openmp_effective_n_threads(n_threads=None, only_physical_cores=True):
    """Determine the effective number of threads to be used for OpenMP calls
    For JAX, this function will return the number of available devices (CPUs, GPUs, TPUs).
    """
    if n_threads == 0:
        raise ValueError("n_threads = 0 is invalid")

    if not _openmp_parallelism_enabled():
        # OpenMP disabled at build-time => sequential mode
        return 1

    if os.getenv("OMP_NUM_THREADS"):
        # Fall back to user provided number of threads making it possible
        # to exceed the number of cpus.
        max_n_threads = len(jax.devices())
    else:
        try:
            n_cpus = _CPU_COUNTS[only_physical_cores]
        except KeyError:
            n_cpus = len(jax.devices())
            _CPU_COUNTS[only_physical_cores] = n_cpus
        max_n_threads = min(len(jax.devices()), n_cpus)

    if n_threads is None:
        return max_n_threads
    elif n_threads < 0:
        return max(1, max_n_threads + n_threads + 1)

    return n_threads
