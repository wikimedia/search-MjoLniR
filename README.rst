MjoLniR - Machine Learned Ranking for Wikimedia
===============================================

MjoLniR is a library for handling the backend data processing for Machine
Learned Ranking at Wikimedia. It is specialized to how click logs are stored at
Wikimedia and provides functionality to transform the source click logs into ML
models for ranking in elasticsearch.

Requirements
============

Targets pyspark from cdh5.10.0. This is mostly pyspark 1.6.0, but has various
backports integrated. Requires python 2.7, as some dependencies (clickmodels)
do not support python 3 yet.

Running tests
=============

Tests can be run from within the provided Vagrant configuration. Use the
following from the root of this repository to build a vagrant box, ssh into it,
and run the tests::

    vagrant up
    vagrant ssh
    cd /vagrant
    venv/bin/tox

The test suite includes both flake8 (linter) and pytest (unit) tests. These
can be run independently with the -e option for tox::

    venv/bin/tox -e flake8

Individual pytest tests can be run by specifying the path on the command line::

    venv/bin/tox -e pytest mjolnir/test/test_sampling.py

Other
=====

Documentation follows the numpy documentation guidelines:
    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
