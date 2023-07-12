"""
Machine learning module for Python
==================================

xlearn is a Python module integrating classical machine
learning algorithms in the tightly-knit world of scientific Python
packages (numpy, scipy, matplotlib).

It aims to provide simple and efficient solutions to learning problems
that are accessible to everybody and reusable in various contexts:
machine-learning as a versatile tool for science and engineering.

See http://jax-learn.cc for complete documentation.
"""
import logging
import os
import random

from ._config import config_context, get_config, set_config

logger = logging.getLogger(__name__)


# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.Y.ZaN   # Alpha release
#   X.Y.ZbN   # Beta release
#   X.Y.ZrcN  # Release Candidate
#   X.Y.Z     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "1.0.0.dev"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

from .base import clone
from .utils._show_versions import show_versions

__all__ = [
    "calibration",
    "cluster",
    "covariance",
    "cross_decomposition",
    "datasets",
    "decomposition",
    "dummy",
    "ensemble",
    "exceptions",
    "experimental",
    "externals",
    "feature_extraction",
    "feature_selection",
    "gaussian_process",
    "inspection",
    "isotonic",
    "kernel_approximation",
    "kernel_ridge",
    "linear_model",
    "manifold",
    "metrics",
    "mixture",
    "model_selection",
    "multiclass",
    "multioutput",
    "naive_bayes",
    "neighbors",
    "neural_network",
    "pipeline",
    "preprocessing",
    "random_projection",
    "semi_supervised",
    "svm",
    "tree",
    "discriminant_analysis",
    "impute",
    "compose",
    "clone",
    "get_config",
    "set_config",
    "config_context",
    "show_versions",
]


def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""

    import jax.numpy as jnp
    import numpy as np

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get("XLEARN_SEED", None)
    if _random_seed is None:
        _random_seed = jax.random.uniform() * jnp.iinfo(jnp.int32).max
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    jax.random.seed(_random_seed)
    random.seed(_random_seed)
