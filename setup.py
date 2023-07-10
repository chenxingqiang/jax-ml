#! /usr/bin/env python

import importlib
import os
import platform
import shutil
import sys
import traceback

from setuptools import Command, setup



DISTNAME = "jax-learn"
DESCRIPTION = "A set of python modules for machine learning and data mining with jax acceleration"
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = "Andreas Mueller"
MAINTAINER_EMAIL = "xingqiang.chen@openmodels.team"
URL = "http://jax-learn.cc"
DOWNLOAD_URL = "https://pypi.org/project/jax-learn/#files"
LICENSE = "new BSD"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/jax-learn/jax-learn/issues",
    "Documentation": "https://jax-learn.cc/stable/documentation.html",
    "Source Code": "https://github.com/jax-learn/jax-learn",
}

# We can actually import a restricted version of xlearn that
# does not need the compiled code
import xlearn  # noqa
import xlearn._min_dependencies as min_deps  # noqa
from xlearn.externals._packaging.version import parse as parse_version  # noqa


VERSION = xlearn.__version__

# Custom clean command to remove build artifacts

class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk("xlearn"):
            for filename in filenames:
                root, extension = os.path.splitext(filename)

                if extension in [".so", ".pyd", ".dll", ".pyc"]:
                    os.unlink(os.path.join(dirpath, filename))

                if remove_c_files and extension in [".c", ".cpp"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))

                if remove_c_files and extension == ".tp":
                    if os.path.exists(os.path.join(dirpath, root)):
                        os.unlink(os.path.join(dirpath, root))

            for dirname in dirnames:
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))


def check_package_status(package, min_version):
    """
    Returns a dictionary containing a boolean specifying whether given package
    is up-to-date, along with the version string (empty string if
    not installed).
    """
    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status["up_to_date"] = parse_version(package_version) >= parse_version(
            min_version
        )
        package_status["version"] = package_version
    except ImportError:
        traceback.print_exc()
        package_status["up_to_date"] = False
        package_status["version"] = ""

    req_str = "jax-learn requires {} >= {}.\n".format(package, min_version)

    instructions = (
        "Installation instructions are available on the "
        "jax-learn website: "
        "http://jax-learn.cc/stable/install.html\n"
    )

    if package_status["up_to_date"] is False:
        if package_status["version"]:
            raise ImportError(
                "Your installation of {} {} is out-of-date.\n{}{}".format(
                    package, package_status["version"], req_str, instructions
                )
            )
        else:
            raise ImportError(
                "{} is not installed.\n{}{}".format(
                    package, req_str, instructions)
            )


def setup_package():
    python_requires = ">=3.8"
    required_python_version = (3, 8)

    metadata = dict(
        name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls=PROJECT_URLS,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: C",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Development Status :: 5 - Production/Stable",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: Implementation :: CPython",
            "Programming Language :: Python :: Implementation :: PyPy",
        ],
        python_requires=python_requires,
        install_requires=min_deps.tag_to_packages["install"],
        package_data={"": ["*.csv", "*.gz",
                           "*.txt", "*.rst", "*.jpg"]},
        zip_safe=False,  # the package can run out of an .egg file
        extras_require={
            key: min_deps.tag_to_packages[key]
            for key in ["examples", "docs", "tests", "benchmark"]
        },
    )

    commands = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if not all(
        command in ("egg_info", "dist_info", "clean", "check") for command in commands
    ):
        if sys.version_info < required_python_version:
            required_version = "%d.%d" % required_python_version
            raise RuntimeError(
                "Jax-learn requires Python %s or later. The current"
                " Python version is %s installed in %s."
                % (required_version, platform.python_version(), sys.executable)
            )

        check_package_status("numpy", min_deps.NUMPY_MIN_VERSION)
        check_package_status("scipy", min_deps.SCIPY_MIN_VERSION)

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
