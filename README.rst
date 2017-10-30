MjoLniR - Machine Learned Ranking for Wikimedia
===============================================

MjoLniR is a library for handling the backend data processing for Machine
Learned Ranking at Wikimedia. It is specialized to how click logs are stored at
Wikimedia and provides functionality to transform the source click logs into ML
models for ranking in elasticsearch.

Requirements
============

Targets pyspark 2.1.0 and xgboost 0.7.  Requires python 2.7, as some
dependencies (clickmodels) do not support python 3 yet.

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

XGBoost version used is v0.7 (unreleased master), with a few additional patches. This
is tracked in the wmf repository cloned from https://gerrit.wikimedia.org/r/search/MjoLniR.
Versions are tagged with wmf, such as 0.7-wmf-1.

XGBoost needs to be built on a debian jessie host to match the analytics cluster. Building
on, for example, an ubuntu xenial install will generate the following error when submitted
to the WMF hadoop cluster:

  /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.23' not found

Somewhere before:
  java.lang.NoClassDefFoundError: Could not initialize class ml.dmlc.xgboost4j.java.XGBoostJNI
