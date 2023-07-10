.. _installation-instructions:

=======================
Installing jax-learn
=======================

There are different ways to install jax-learn:

  * :ref:`Install the latest official release <install_official_release>`. This
    is the best approach for most users. It will provide a stable version
    and pre-built packages are available for most platforms.

  * Install the version of jax-learn provided by your
    :ref:`operating system or Python distribution <install_by_distribution>`.
    This is a quick option for those who have operating systems or Python
    distributions that distribute jax-learn.
    It might not provide the latest release version.

  * :ref:`Building the package from source
    <install_bleeding_edge>`. This is best for users who want the
    latest-and-greatest features and aren't afraid of running
    brand-new code. This is also needed for users who wish to contribute to the
    project.


.. _install_official_release:

Installing the latest release
=============================

.. This quickstart installation is a hack of the awesome
   https://spacy.io/usage/#quickstart page.
   See the original javascript implementation
   https://github.com/ines/quickstart


.. raw:: html

  <div class="install">
       <strong>Operating System</strong>
          <input type="radio" name="os" id="quickstart-win" checked>
          <label for="quickstart-win">Windows</label>
          <input type="radio" name="os" id="quickstart-mac">
          <label for="quickstart-mac">macOS</label>
          <input type="radio" name="os" id="quickstart-lin">
          <label for="quickstart-lin">Linux</label><br />
       <strong>Packager</strong>
          <input type="radio" name="packager" id="quickstart-pip" checked>
          <label for="quickstart-pip">pip</label>
          <input type="radio" name="packager" id="quickstart-conda">
          <label for="quickstart-conda">conda</label><br />
          <input type="checkbox" name="config" id="quickstart-venv">
          <label for="quickstart-venv"></label>
       </span>

.. raw:: html

       <div>
         <span class="sk-expandable" data-packager="pip" data-os="windows">Install the 64bit version of Python 3, for instance from <a href="https://www.python.org/">https://www.python.org</a>.</span
         ><span class="sk-expandable" data-packager="pip" data-os="mac">Install Python 3 using <a href="https://brew.sh/">homebrew</a> (<code>brew install python</code>) or by manually installing the package from <a href="https://www.python.org">https://www.python.org</a>.</span
         ><span class="sk-expandable" data-packager="pip" data-os="linux">Install python3 and python3-pip using the package manager of the Linux Distribution.</span
         ><span class="sk-expandable" data-packager="conda"
            >Install conda using the <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/">Anaconda or miniconda</a>
             installers or the <a href="https://https://github.com/conda-forge/miniforge#miniforge">miniforge</a> installers
             (no administrator permission required for any of those).</span>
       </div>

Then run:

.. raw:: html

       <div class="highlight"><pre><code
        ><span class="sk-expandable" data-packager="pip" data-os="linux" data-venv="">python3 -m venv xlearn-venv</span
        ><span class="sk-expandable" data-packager="pip" data-os="windows" data-venv="">python -m venv xlearn-venv</span
        ><span class="sk-expandable" data-packager="pip" data-os="mac" data-venv="">python -m venv xlearn-venv</span
        ><span class="sk-expandable" data-packager="pip" data-os="linux" data-venv="">source xlearn-venv/bin/activate</span
        ><span class="sk-expandable" data-packager="pip" data-os="mac" data-venv="">source xlearn-venv/bin/activate</span
        ><span class="sk-expandable" data-packager="pip" data-os="windows" data-venv="">xlearn-venv\Scripts\activate</span
        ><span class="sk-expandable" data-packager="pip" data-venv="">pip install -U jax-learn</span
        ><span class="sk-expandable" data-packager="pip" data-os="mac" data-venv="no">pip install -U jax-learn</span
        ><span class="sk-expandable" data-packager="pip" data-os="windows" data-venv="no">pip install -U jax-learn</span
        ><span class="sk-expandable" data-packager="pip" data-os="linux" data-venv="no">pip3 install -U jax-learn</span
        ><span class="sk-expandable" data-packager="conda">conda create -n xlearn-env -c conda-forge jax-learn</span
        ><span class="sk-expandable" data-packager="conda">conda activate xlearn-env</span
       ></code></pre></div>

In order to check your installation you can use

.. raw:: html

   <div class="highlight"><pre><code
      ><span class="sk-expandable" data-packager="pip" data-os="linux" data-venv="no">python3 -m pip show jax-learn  # to see which version and where jax-learn is installed</span
      ><span class="sk-expandable" data-packager="pip" data-os="linux" data-venv="no">python3 -m pip freeze  # to see all packages installed in the active virtualenv</span
      ><span class="sk-expandable" data-packager="pip" data-os="linux" data-venv="no">python3 -c "import xlearn; xlearn.show_versions()"</span
      ><span class="sk-expandable" data-packager="pip" data-venv="">python -m pip show jax-learn  # to see which version and where jax-learn is installed</span
      ><span class="sk-expandable" data-packager="pip" data-venv="">python -m pip freeze  # to see all packages installed in the active virtualenv</span
      ><span class="sk-expandable" data-packager="pip" data-venv="">python -c "import xlearn; xlearn.show_versions()"</span
      ><span class="sk-expandable" data-packager="pip" data-os="windows" data-venv="no">python -m pip show jax-learn  # to see which version and where jax-learn is installed</span
      ><span class="sk-expandable" data-packager="pip" data-os="windows" data-venv="no">python -m pip freeze  # to see all packages installed in the active virtualenv</span
      ><span class="sk-expandable" data-packager="pip" data-os="windows" data-venv="no">python -c "import xlearn; xlearn.show_versions()"</span
      ><span class="sk-expandable" data-packager="pip" data-os="mac" data-venv="no">python -m pip show jax-learn  # to see which version and where jax-learn is installed</span
      ><span class="sk-expandable" data-packager="pip" data-os="mac" data-venv="no">python -m pip freeze  # to see all packages installed in the active virtualenv</span
      ><span class="sk-expandable" data-packager="pip" data-os="mac" data-venv="no">python -c "import xlearn; xlearn.show_versions()"</span
      ><span class="sk-expandable" data-packager="conda">conda list jax-learn  # to see which jax-learn version is installed</span
      ><span class="sk-expandable" data-packager="conda">conda list  # to see all packages installed in the active conda environment</span
      ><span class="sk-expandable" data-packager="conda">python -c "import xlearn; xlearn.show_versions()"</span
      ></code></pre></div>
  </div>

Note that in order to avoid potential conflicts with other packages it is
strongly recommended to use a `virtual environment (venv)
<https://docs.python.org/3/tutorial/venv.html>`_ or a `conda environment
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Using such an isolated environment makes it possible to install a specific
version of jax-learn with pip or conda and its dependencies independently of
any previously installed Python packages. In particular under Linux is it
discouraged to install pip packages alongside the packages managed by the
package manager of the distribution (apt, dnf, pacman...).

Note that you should always remember to activate the environment of your choice
prior to running any Python command whenever you start a new terminal session.

If you have not installed NumPy or SciPy yet, you can also install these using
conda or pip. When using pip, please ensure that *binary wheels* are used,
and NumPy and SciPy are not recompiled from source, which can happen when using
particular configurations of operating system and hardware (such as Linux on
a Raspberry Pi).


Jax-learn plotting capabilities (i.e., functions start with "plot\_"
and classes end with "Display") require Matplotlib. The examples require
Matplotlib and some examples require jax-image, pandas, or seaborn. The
minimum version of Jax-learn dependencies are listed below along with its
purpose.

.. include:: min_dependency_table.rst

.. warning::

    Jax-learn 0.20 was the last version to support Python 2.7 and Python 3.4.
    Jax-learn 0.21 supported Python 3.5-3.7.
    Jax-learn 0.22 supported Python 3.5-3.8.
    Jax-learn 0.23 - 0.24 require Python 3.6 or newer.
    Jax-learn 1.0 supported Python 3.7-3.10.
    Jax-learn 1.1 and later requires Python 3.8 or newer.


.. _install_on_apple_silicon_m1:

Installing on Apple Silicon M1 hardware
=======================================

The recently introduced `macos/arm64` platform (sometimes also known as
`macos/aarch64`) requires the open source community to upgrade the build
configuration and automation to properly support it.

At the time of writing (January 2021), the only way to get a working
installation of jax-learn on this hardware is to install jax-learn and its
dependencies from the conda-forge distribution, for instance using the miniforge
installers:

https://github.com/conda-forge/miniforge

The following issue tracks progress on making it possible to install
jax-learn from PyPI with pip:

https://github.com/jax-learn/jax-learn/issues/19137


.. _install_by_distribution:

Third party distributions of jax-learn
=========================================

Some third-party distributions provide versions of
jax-learn integrated with their package-management systems.

These can make installation and upgrading much easier for users since
the integration includes the ability to automatically install
dependencies (numpy, scipy) that jax-learn requires.

The following is an incomplete list of OS and python distributions
that provide their own version of jax-learn.

Alpine Linux
------------

Alpine Linux's package is provided through the `official repositories
<https://pkgs.alpinelinux.org/packages?name=py3-jax-learn>`__ as
``py3-jax-learn`` for Python.
It can be installed by typing the following command:

.. prompt:: bash $

  sudo apk add py3-jax-learn


Arch Linux
----------

Arch Linux's package is provided through the `official repositories
<https://www.archlinux.org/packages/?q=jax-learn>`_ as
``python-jax-learn`` for Python.
It can be installed by typing the following command:

.. prompt:: bash $

  sudo pacman -S python-jax-learn


Debian/Ubuntu
-------------

The Debian/Ubuntu package is split in three different packages called
``python3-xlearn`` (python modules), ``python3-xlearn-lib`` (low-level
implementations and bindings), ``python3-xlearn-doc`` (documentation).
Only the Python 3 version is available in the Debian Buster (the more recent
Debian distribution).
Packages can be installed using ``apt-get``:

.. prompt:: bash $

  sudo apt-get install python3-xlearn python3-xlearn-lib python3-xlearn-doc


Fedora
------

The Fedora package is called ``python3-jax-learn`` for the python 3 version,
the only one available in Fedora30.
It can be installed using ``dnf``:

.. prompt:: bash $

  sudo dnf install python3-jax-learn


NetBSD
------

jax-learn is available via `pkgsrc-wip
<http://pkgsrc-wip.sourceforge.net/>`_:

    https://pkgsrc.se/math/py-jax-learn


MacPorts for Mac OSX
--------------------

The MacPorts package is named ``py<XY>-jaxs-learn``,
where ``XY`` denotes the Python version.
It can be installed by typing the following
command:

.. prompt:: bash $

  sudo port install py39-jax-learn


Anaconda and Enthought Deployment Manager for all supported platforms
---------------------------------------------------------------------

`Anaconda <https://www.anaconda.com/download>`_ and
`Enthought Deployment Manager <https://assets.enthought.com/downloads/>`_
both ship with jax-learn in addition to a large set of scientific
python library for Windows, Mac OSX and Linux.

Anaconda offers jax-learn as part of its free distribution.


Intel Extension for Jax-learn
--------------------------------

Intel maintains an optimized x86_64 package, available in PyPI (via `pip`),
and in the `main`, `conda-forge` and `intel` conda channels:

.. prompt:: bash $

  conda install jax-learn-intelex

This package has an Intel optimized version of many estimators. Whenever 
an alternative implementation doesn't exist, jax-learn implementation 
is used as a fallback. Those optimized solvers come from the oneDAL 
C++ library and are optimized for the x86_64 architecture, and are 
optimized for multi-core Intel CPUs.

Note that those solvers are not enabled by default, please refer to the
`jax-learn-intelex <https://intel.github.io/jax-learn-intelex/what-is-patching.html>`_ 
documentation for more details on usage scenarios. Direct export example:

.. prompt:: bash $

  from xlearnex.neighbors import NearestNeighbors

Compatibility with the standard jax-learn solvers is checked by running the
full jax-learn test suite via automated continuous integration as reported
on https://github.com/intel/jax-learn-intelex. If you observe any issue
with `jax-learn-intelex`, please report the issue on their
`issue tracker <https://github.com/intel/jax-learn-intelex/issues>`__.


WinPython for Windows
-----------------------

The `WinPython <https://winpython.github.io/>`_ project distributes
jax-learn as an additional plugin.


Troubleshooting
===============

.. _windows_longpath:

Error caused by file path length limit on Windows
-------------------------------------------------

It can happen that pip fails to install packages when reaching the default path
size limit of Windows if Python is installed in a nested location such as the
`AppData` folder structure under the user home directory, for instance::

    C:\Users\username>C:\Users\username\AppData\Local\Microsoft\WindowsApps\python.exe -m pip install jax-learn
    Collecting jax-learn
    ...
    Installing collected packages: jax-learn
    ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\Users\\username\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python37\\site-packages\\xlearn\\datasets\\tests\\data\\openml\\292\\api-v1-json-data-list-data_name-australian-limit-2-data_version-1-status-deactivated.json.gz'

In this case it is possible to lift that limit in the Windows registry by
using the ``regedit`` tool:

#. Type "regedit" in the Windows start menu to launch ``regedit``.

#. Go to the
   ``Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem``
   key.

#. Edit the value of the ``LongPathsEnabled`` property of that key and set
   it to 1.

#. Reinstall jax-learn (ignoring the previous broken installation):

.. prompt:: python $

    pip install --exists-action=i jax-learn
