from distutils.core import setup
from setuptools import find_packages

setup(
    name='drl_algos',
    version='0.0.1dev',
    packages=find_packages(include=[
        "drl_algos",
        "drl_algos.*"
    ]),
    license='MIT License',
    long_description=open('README.md').read(),
)
