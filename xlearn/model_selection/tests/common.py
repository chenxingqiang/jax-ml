"""
Common utilities for testing model selection.
"""

import jax.numpy as jnp

from xlearn.model_selection import KFold


class OneTimeSplitter:
    """A wrapper to make KFold single entry cv iterator"""

    def __init__(self, n_splits=4, n_samples=99):
        self.n_splits = n_splits
        self.n_samples = n_samples
        self.indices = iter(KFold(n_splits=n_splits).split(jnp.ones(n_samples)))

    def split(self, X=None, y=None, groups=None):
        """Split can be called only once"""
        for index in self.indices:
            yield index

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
