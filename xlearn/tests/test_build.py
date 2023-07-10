import os
import textwrap

import pytest

from xlearn import __version__
from xlearn.utils._openmp_helpers import _openmp_parallelism_enabled


def test_openmp_parallelism_enabled():
    # Check that xlearn is built with OpenMP-based parallelism enabled.
    # This test can be skipped by setting the environment variable
    # ``XLEARN_SKIP_OPENMP_TEST``.
    if os.getenv("XLEARN_SKIP_OPENMP_TEST"):
        pytest.skip("test explicitly skipped (XLEARN_SKIP_OPENMP_TEST)")

    base_url = "dev" if __version__.endswith(".dev0") else "stable"
    err_msg = textwrap.dedent("""
        This test fails because jax-learn has been built without OpenMP.
        This is not recommended since some estimators will run in sequential
        mode instead of leveraging thread-based parallelism.

        You can find instructions to build jax-learn with OpenMP at this
        address:

            https://jax-learn.cc/{}/developers/advanced_installation.html

        You can skip this test by setting the environment variable
        XLEARN_SKIP_OPENMP_TEST to any value.
        """).format(base_url)

    assert _openmp_parallelism_enabled(), err_msg
