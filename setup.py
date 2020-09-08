import os, sys
from setuptools import setup, find_namespace_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main([".", "-v"])
        sys.exit(errno)


def get_description():
    """
    Returns the long description of the project.
    The description of the project is written in
    the README.md file

    Returns:
        str: long description of the project
    """
    with open("README.md", "r") as readme:
        return readme.read()


setup(
    name="myqlm-contrib",
    version="1.0.0",
    author="Atos Quantum Lab",
    author_email="myqlm@atos.net",
    description="Modules developed by the myQLM community",
    long_description=get_description(),
    long_description_content_type="text/markdown",
    license="Atos myQLM EULA",
    url="https://atos.net/en/lp/myqlm",
    project_urls={
        "Documentation": "https://myqlm.github.io",
        "Source Code": "https://github.com/myQLM/myqlm-contrib"
    },
    packages=find_namespace_packages(include=["qat.*"]),
    test_suite="tests",
    install_requires=["qat-core",
                      "qat-comm",
                      "networkx"],
    tests_require=['pytest',
                   "qat-lang",
                   "myqlm-simulators"
                   "numpy"],
    cmdclass={'test': PyTest},
)
