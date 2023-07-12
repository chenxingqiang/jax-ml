import mmh3
import numpy as np

def murmurhash3_32(key, seed=0, positive=False):
    """Compute the 32bit murmurhash3 of key at seed.
    Parameters
    ----------
    key : int, bytes, str or ndarray of dtype=int
        The physical object to hash.
    seed : int, default=0
        Integer seed for the hashing algorithm.
    positive : bool, default=False
        True: the results is casted to an unsigned int
          from 0 to 2 ** 32 - 1
        False: the results is casted to a signed int
          from -(2 ** 31) to 2 ** 31 - 1
    """
    if isinstance(key, bytes) or isinstance(key, str):
        return mmh3.hash(key, seed, signed=not positive)
    elif isinstance(key, int):
        return mmh3.hash(str(key), seed, signed=not positive)
    elif isinstance(key, np.ndarray):
        if key.dtype != np.int32:
            raise TypeError("key.dtype should be int32, got %s" % key.dtype)
        hash_vectorized = np.vectorize(lambda x: mmh3.hash(str(x), seed, signed=not positive))
        return hash_vectorized(key)
    else:
        raise TypeError("key %r with type %s is not supported. "
                        "Explicit conversion to bytes is required" % (key, type(key)))
