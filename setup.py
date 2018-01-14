import os
from setuptools import find_packages, setup


requirements = [
    # mjolnir requirements
    'requests',
    'kafka',
    'pyyaml',
    'hyperopt',
    # python xgboost is only used for building
    # binary datasets. Primary usage is from jvm.
    'xgboost',
    # hyperopt requires networkx < 2.0, but doesn't say so
    'networkx<2.0',
    # pyspark requirements
    'py4j',
    'numpy',
]

test_requirements = [
    'findspark',
    'flake8',
    'pytest',
    'tox',
]

setup(
    name='mjolnir',
    version='0.0.1',
    author='Wikimedia Search Team',
    author_email='discovery@lists.wikimedia.org',
    description='A plumbing library for Machine Learned Ranking at Wikimedia',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mjolnir-utilities.py = mjolnir.__main__:main',
        ],
    },
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
