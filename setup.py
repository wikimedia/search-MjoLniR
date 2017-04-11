import os
from setuptools import find_packages, setup


requirements = [
    # mjolnir requirements
    'clickmodels',
    # pyspark requirements
    'py4j',
    'numpy',
]

test_requirements = [
    'pytest',
    'findspark',
    'flake8',
    'tox',
]

setup(
    name='MjoLniR',
    version='0.0.1',
    author='Wikimedia Search Team',
    author_email='discovery@lists.wikimedia.org',
    description='A plumbing library for Machine Learned Ranking at Wikimedia',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    data_files=['README.rst'],
    install_requires=requirements,
    test_requires=test_requirements,
    extras_require={
        "test": test_requirements
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
