from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("calculate_dW_hat.pyx")
)