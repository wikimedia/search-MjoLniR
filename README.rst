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

XGBoost version used is v0.7, with a few additional patches cherry picked in:
* https://github.com/dmlc/xgboost/pull/2234 - support json dumps in xgboost4j
* https://github.com/dmlc/xgboost/pull/2241 - store metrics with serialized learner
* https://github.com/dmlc/xgboost/pull/2247 - expose json dumps to scala
* https://github.com/dmlc/xgboost/pull/2244 - accept groupData in spark model eval
