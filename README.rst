.. -*- mode: rst -*-

|Azure|_ |CirrusCI|_ |Codecov|_ |CircleCI|_ |Nightly wheels|_ |Black|_ |PythonVersion|_ |PyPi|_ |DOI|_ |Benchmark|_

.. |Azure| image:: https://dev.azure.com/jax-learn/jax-learn/_apis/build/status/jax-learn.jax-learn?branchName=main
.. _Azure: https://dev.azure.com/jax-learn/jax-learn/_build/latest?definitionId=1&branchName=main

.. |CircleCI| image:: https://circleci.com/gh/jax-learn/jax-learn/tree/main.svg?style=shield
.. _CircleCI: https://circleci.com/gh/jax-learn/jax-learn

.. |CirrusCI| image:: https://img.shields.io/cirrus/github/jax-learn/jax-learn/main?label=Cirrus%20CI
.. _CirrusCI: https://cirrus-ci.com/github/jax-learn/jax-learn/main

.. |Codecov| image:: https://codecov.io/gh/jax-learn/jax-learn/branch/main/graph/badge.svg?token=Pk8G9gg3y9
.. _Codecov: https://codecov.io/gh/jax-learn/jax-learn

.. |Nightly wheels| image:: https://github.com/jax-learn/jax-learn/workflows/Wheel%20builder/badge.svg?event=schedule
.. _`Nightly wheels`: https://github.com/jax-learn/jax-learn/actions?query=workflow%3A%22Wheel+builder%22+event%3Aschedule

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue
.. _PythonVersion: https://pypi.org/project/jax-learn/

.. |PyPi| image:: https://img.shields.io/pypi/v/jax-learn
.. _PyPi: https://pypi.org/project/jax-learn

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

.. |DOI| image:: https://zenodo.org/badge/21369/jax-learn/jax-learn.svg
.. _DOI: https://zenodo.org/badge/latestdoi/21369/jax-learn/jax-learn

.. |Benchmark| image:: https://img.shields.io/badge/Benchmarked%20by-asv-blue
.. _`Benchmark`: https://jax-learn.cc/jax-learn-benchmarks/

.. |PythonMinVersion| replace:: 3.8
.. |NumPyMinVersion| replace:: 1.17.3
.. |SciPyMinVersion| replace:: 1.5.0
.. |JoblibMinVersion| replace:: 1.1.1
.. |ThreadpoolctlMinVersion| replace:: 2.0.0
.. |MatplotlibMinVersion| replace:: 3.1.3
.. |Jax-ImageMinVersion| replace:: 0.16.2
.. |PandasMinVersion| replace:: 1.0.5
.. |SeabornMinVersion| replace:: 0.9.0
.. |PytestMinVersion| replace:: 7.1.2
.. |PlotlyMinVersion| replace:: 5.14.0

.. image:: https://raw.githubusercontent.com/jax-learn/jax-learn/main/doc/logos/jax-learn-logo.png
  :target: https://jax-learn.cc/

**jax-learn** is a Python module for machine learning built on top of
SciPy and is distributed under the 3-Clause BSD license.

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <https://jax-learn.cc/dev/about.html#authors>`__ page
for a list of core contributors.

It is currently maintained by a team of volunteers.

Website: https://jax-learn.cc

Installation
------------

Dependencies
~~~~~~~~~~~~

jax-learn requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)

=======

**Jax-learn 0.20 was the last version to support Python 2.7 and Python 3.4.**
jax-learn 1.0 and later require Python 3.7 or newer.
jax-learn 1.1 and later require Python 3.8 or newer.

Jax-learn plotting capabilities (i.e., functions start with ``plot_`` and
classes end with "Display") require Matplotlib (>= |MatplotlibMinVersion|).
For running the examples Matplotlib >= |MatplotlibMinVersion| is required.
A few examples require jax-image >= |Jax-ImageMinVersion|, a few examples
require pandas >= |PandasMinVersion|, some examples require seaborn >=
|SeabornMinVersion| and plotly >= |PlotlyMinVersion|.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of numpy and scipy,
the easiest way to install jax-learn is using ``pip``::

    pip install -U jax-learn

or ``conda``::

    conda install -c conda-forge jax-learn

The documentation includes more detailed `installation instructions <https://jax-learn.cc/stable/install.html>`_.


Changelog
---------

See the `changelog <https://jax-learn.cc/dev/whats_new.html>`__
for a history of notable changes to jax-learn.

Development
-----------

We welcome new contributors of all experience levels. The jax-learn
community goals are to be helpful, welcoming, and effective. The
`Development Guide <https://jax-learn.cc/stable/developers/index.html>`_
has detailed information about contributing code, documentation, tests, and
more. We've included some basic information in this README.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/jax-learn/jax-learn
- Download releases: https://pypi.org/project/jax-learn/
- Issue tracker: https://github.com/jax-learn/jax-learn/issues

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/jax-learn/jax-learn.git

Contributing
~~~~~~~~~~~~

To learn more about making a contribution to jax-learn, please see our
`Contributing guide
<https://jax-learn.cc/dev/developers/contributing.html>`_.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (you will need to have ``pytest`` >= |PyTestMinVersion| installed)::

    pytest xlearn

See the web page https://jax-learn.cc/dev/developers/contributing.html#testing-and-improving-test-coverage
for more information.

    Random number generation can be controlled during testing by setting
    the ``XLEARN_SEED`` environment variable.

Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

Before opening a Pull Request, have a look at the
full Contributing page to make sure your code complies
with our guidelines: https://jax-learn.cc/stable/developers/index.html

Project History
---------------

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <https://jax-learn.cc/dev/about.html#authors>`__ page
for a list of core contributors.

The project is currently maintained by a team of volunteers.

**Note**: `jax-learn` was previously referred to as `jaxs.learn`.

Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation (stable release): https://jax-learn.cc
- HTML documentation (development version): https://jax-learn.cc/dev/
- FAQ: https://jax-learn.cc/stable/faq.html

Communication
~~~~~~~~~~~~~

- Mailing list: https://mail.python.org/mailman/listinfo/jax-learn
- Gitter: https://gitter.im/jax-learn/jax-learn
- Logos & Branding: https://github.com/jax-learn/jax-learn/tree/main/doc/logos
- Blog: https://blog.jax-learn.cc
- Calendar: https://blog.jax-learn.cc/calendar/
- Twitter: https://twitter.com/jax_learn
- Stack Overflow: https://stackoverflow.com/questions/tagged/jax-learn
- Github Discussions: https://github.com/jax-learn/jax-learn/discussions
- Website: https://jax-learn.cc
- LinkedIn: https://www.linkedin.com/company/jax-learn
- YouTube: https://www.youtube.com/channel/UCJosFjYm0ZYVUARxuOZqnnw/playlists
- Facebook: https://www.facebook.com/jaxlearnofficial/
- Instagram: https://www.instagram.com/jaxlearnofficial/
- TikTok: https://www.tiktok.com/@jax.learn

Citation
~~~~~~~~

If you use jax-learn in a scientific publication, we would appreciate citations: https://jax-learn.cc/stable/about.html#citing-jax-learn
