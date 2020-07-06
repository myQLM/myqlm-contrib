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


setup(
    name="qat-plugins",
    version="0.0.6",
    author="Atos Quantum Lab",
    license="Atos myQLM EULA",
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
