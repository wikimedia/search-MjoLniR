import os
from setuptools import find_packages, setup


requirements = [
    # mjolnir requirements
    'requests',
    'kafka-python<2.0',
    'pyyaml',
    'hyperopt<0.2.0',
    'elasticsearch>=5.0.0,<6.0.0',
    'jsonschema',
    'prometheus_client',
    'xgboost==0.90',
    # hyperopt requires networkx < 2.0, but doesn't say so
    'networkx<2.0',
    # pyspark requirements
    'py4j',
    'numpy',
    # for wmf logging integration
    'python-json-logger',
]

test_requirements = [
    'findspark',
    'flake8',
    'pytest',
    'pytest-mock',
    'tox',
]

setup(
    name='wmf_mjolnir',
    version='0.0.2',
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
        "Programming Language :: Python :: 3",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
