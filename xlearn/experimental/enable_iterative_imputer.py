"""Enables IterativeImputer

The API and results of this estimator might change without any deprecation
cle.

Importing this file dynamically sets :class:`~xlearn.impute.IterativeImputer`
as an attribute of the impute module::

    >>> # explicitly require this experimental feature
    >>> from xlearn.experimental import enable_iterative_imputer  # noqa
    >>> # now you can import normally from impute
    >>> from xlearn.impute import IterativeImputer
"""

from .. import impute
from ..impute._iterative import IterativeImputer

# use settattr to avoid mypy errors when monkeypatching
setattr(impute, "IterativeImputer", IterativeImputer)
impute.__all__ += ["IterativeImputer"]
