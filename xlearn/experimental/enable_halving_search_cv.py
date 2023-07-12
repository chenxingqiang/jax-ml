"""Enables Successive Halving search-estimators

The API and results of these estimators might change without any deprecation
cle.

Importing this file dynamically sets the
:class:`~xlearn.model_selection.HalvingRandomSearchCV` and
:class:`~xlearn.model_selection.HalvingGridSearchCV` as attributes of the
`model_selection` module::

    >>> # explicitly require this experimental feature
    >>> from xlearn.experimental import enable_halving_search_cv # noqa
    >>> # now you can import normally from model_selection
    >>> from xlearn.model_selection import HalvingRandomSearchCV
    >>> from xlearn.model_selection import HalvingGridSearchCV


The ``# noqa`` comment comment can be removed: it just tells linters like
flake8 to ignore the import, which appears as unused.
"""

from .. import model_selection
from ..model_selection._search_successive_halving import (
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)

# use settattr to avoid mypy errors when monkeypatching
setattr(model_selection, "HalvingRandomSearchCV", HalvingRandomSearchCV)
setattr(model_selection, "HalvingGridSearchCV", HalvingGridSearchCV)

model_selection.__all__ += ["HalvingRandomSearchCV", "HalvingGridSearchCV"]
