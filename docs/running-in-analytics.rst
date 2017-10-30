Terms
=====

A few terms used in this document:

* yarn - This is the hadoop cluster manager. Anything that wants to request
  resources from the hadoop cluster must request it from yarn. Yarn has a webui
  at https://yarn.wikimedia.org which requires LDAP credentials to view.

* stage - At a high level spark jobs are described as a DAG of stages that need
  to be executed on data. Each stage represents one more more operations to
  perform on some data.

* task - Spark stages are broken up into many tasks. Each task represents the work
  to do for a single partition of a single stage of computation.

* driver - This is the JVM that is in control of the spark job and orchestrates
  everything. This instance does not generally get assigned any tasks, instead only
  being responsible for orchestrating the delivery of tasks to executors. For
  the instructions in this document this is always on stat1005, but it is
  possible for the driver to be created inside the hadoop cluster instead.

* executor - Also known as workers. These instances do the heavy lifting of executing
  spark tasks. When asked for resources yarn will spin up containers on nodes
  inside the hadoop cluster and startup JVM's running spark. Those JVM's will
  contact the driver and be assigned tasks to perform.

* RDD - This is the basic building block of spark parallelism. It stands for
  resilient distributed dataset. Underneath every parallel operation in spark
  there is eventually an rdd.

Caveats
=======

As MjoLniR is still in early days there is not a defined process to deploy to the analytics cluster.
This will be worked out, but for now it is all manual. Some caveats:
>
* MjoLniR must be run from a debian jessie or greater based host. The is
  because the hadoop workers are also running debian jessie and the python
  binary on ubuntu will not run properly on them.

* The best host to run everything from is stat1005, a debian stretch based machine.

* MjoLniR requires spark 2.1.0, but the default installed version in the hadoop
  cluster is 1.6.0.  You will need to fetch spark from apache and uncompress
  the archive in your home directory on stat1005.

* MjoLniR requires a python virtualenv containing all the appropriate library
  dependencies. Pip shiped with debian stretch is able to utilize wheel packages
  which makes installing all our dependencies a breeze.

* MjoLniR requires a jar built in the /jvm directory of this repository, along
  with a few others. These are all deployed to archiva and referenced from the
  configuration for the `spark` utility.

* Some conflict with dependencies installed in the analytics cluster may cause kafka streaming
  to fail with `kafka.cluster.BrokerEndPoint cannot be cast to kafka.cluster.broker`.
  This is probably due to `kafka 0.9` client jar being imported with flume. Workround
  is to copy wmf spark conf:
  `cp -r /etc/spark/conf.analytics-hadoop my_spark_conf`
  then comment the flume dependency in spark-env.sh:
  `#SPARK_DIST_CLASSPATH="$SPARK_DIST_CLASSPATH:/usr/lib/flume-ng/lib/*"`
  and run spark with SPARK_CONF_DIR set to your custom folder.

Spark gotchas
=============

* Spark will only write out new datasets to non-existent directories. It is possible if some command
  (i.e. data_pipeline.py) that was supposed to write out to directory fails mid run that the directory
  will be created but unpopulated. It needs to be removed with `hdfs dfs -rm -r -f hdfs://analytics-hadoop/...`
  before re-running the command. You can check if the directory exists in `/mnt/hdfs`.

Setting everything up
=====================

Taking the above into account, the process to get started is basically (all run from stat1005.eqiad.wmnet):

Clone mjolnir, build a virtualenv, and zip it up for deployment to spark executors::

	cd ~
	git clone https://gerrit.wikimedia.org/r/search/MjoLniR mjolnir
	cd mjolnir
	virtualenv venv
	venv/bin/pip install .
	cd venv
	zip -qr ../mjolnir_venv.zip .
	cd ..

Pull down spark 2.1 and decompress that into your home directory::

	cd ~
	https_proxy=http://webproxy.eqiad.wmnet:8080 wget https://archive.apache.org/dist/spark/spark-2.1.0/spark-2.1.0-bin-hadoop2.6.tgz
	tar -zxf spark-2.1.0-bin-hadoop2.6.tgz

Upgrading mjolnir
=================

Upgrading mjolnir in your analytics checkout is fairly painless. Run the
following commands. If you need dependencies as well leave off the --no-deps
argument, and be sure to set an appropriate https_proxy environment variable::

	cd ~/mjolnir
	git pull --ff-only
	venv/bin/pip install --upgrade --no-deps .
	cd venv
	zip -qr ../mjolnir_venv.zip .
	cd ..

The configuration file
======================

The configuration file, located at `example_train.yaml`, helps automate the
relatively tedious task of running spark command lines. Both spark and mjolnir
take an incredible amount of arguments that all have to be configured just-so
for training to work out. The high level design of this file is to have global
and per-profile configuration, and then to have defined commands. See the doc
comments in mjolnir/utilities/spark.py for more information. Many things in
this file are templated where they might not really need to be to allow
overriding them from the command line.

An explanation of some of the configuration used:

* PYSPARK_PYTHON - Tells spark where to find the python executable. This path
  must be a relative path to work both locally and on the worker nodes where
  mjolnir_venv.zip is decompressed.

* SPARK_CONF_DIR - Tells spark where to find it's configuration. This is
  required because we are using spark 2.1.0, but spark 1.6.0 is installed on
  the machines

* spark-2.1.0-bin-hadoop2.6/bin/pyspark - The executable that stands up the
  jvm, talks to yarn, etc. The pyspark executable specifically stands up an
  interactive python REPL.

* --repositories ... - Tells spark where to source jvm dependencies from

* --packages ... - Tells spark what our jvm dependencies are

* --master yarn - Tells spark we will be distributing the work across a cluster.
  Without this option all spark workers will be local within the same JVM

* --files ... - Additional files spark should ship to the executors. For some
  reason libhdfs isn't always found so this ensures it is available.

* --archives ... - Files that spark should decompress into the working
  directory. The part before # is the path to the file locally, and the part
  after the # is the directory to decompress to.

data_pipeline.py arguments:

* -i The input directory containing the query click data. It is unlikely you
  will ever need to use a different value than shown here.

* -o The output directory. This is where the training data will be stored. This
  must be on HDFS. This may vary as you generate different sizes of training data

* -c The search cluster to use. It is very important that this is pointed at
  the *hot*spare* search cluster.  Pointing this at the currently active cluster
  could cause increased latency for our users.

training_pipeline.py takes a few more arguments, mostly related to having
an appropriate amount of resources available for training:

* --conf spark.dynamicAllocation.maxExecutors=105 - The training process can
  use an incredible amount of resources on the cluster if allowed to. Generally
  we want to prevent mjolnir from taking up more than half the cluster for short
  runs, and probably less than 1/3 of the cluster for jobs that will run for many
  hours. Further below is some discussion on spark resource usage.

* --conf spark.sql.autoBroadcastJoinThreshold=-1 - Spark can do a join using an
  expensive distributed algorithm, or it can broadcast a small table to all
  executors and let them do a cheaper join directly against that broadcasted
  table. This configuration isn't strictly required, but if spark executors start
  getting killed for running over their memory limits on small to mid sized
  datasets this can help.

* --conf spark.task.cpus=4 - This sets the number of cpus in an executor to assign
  to an individual task. The default value of 1 means that if we spin up executors
  with 4 cores, 4 tasks will be assigned. When training with xgboost we want a single
  task to have access to all the cores, so we set this to the same value as the
  number of cores assigned to each executor.

* --conf spark.yarn.executor.memoryOverhead=1536 - This sets the amount of memory
  that will be requested from yarn (the cluster manager) but not provided to the
  JVM heap. When training with XGBoost all the training data is held off-heap in
  C++ so this needs to be large enough for general overhead and the off-heap
  training data.

* --executor-memory 2G - This is, approximately, the size of the java heap. Roughly
  60% of this will be reserved for spark block storage (local copies of dataframes
  held in memory, such as the cross-validation folds). The other 40% is available
  for execution overhead. A reasonably large amount of memory is needed for loading
  the training data and shipping it over to xgboost via JNI. See spark docs at
  https://spark.apache.org/docs/2.1.0/tuning.html#memory-management-overview

* --executor-cores 4 - This is the number of cores that will be requested from yarn
  for each executor. With the current cluster configuration 4 is the maximum that
  can be requested. Must be the same as spark.task.cpus above when training

* -i ... - Tells the training pipeline where to find the training data. This must be
  on HDFS and should be the output of the `data_pipeline.py` script.

* -o ... - Tells the training pipeline where to write out various information about
  the results of training. This must be a local path.

* -w 1 - Tells the training pipeline how many executors should be used to train a single
 model. When doing feature engineering with small-ish (~1M sample) training sets the most
 efficient use of resources is to train many models in parallel with a single worker per
 model.

* -c 100 - This is the number of models to train in parallel. The total number of executors
 required is this times the number of workers per model. In this example that is 100 * 1.

* -f 5 - The number of folds to use for cross-validation. This can be a bit of a complicated
  decision, but generally 5 is an acceptable, it not amazing, tradeoff of training time
  vs. training accuracy. Basically for every set of training parameters attempted this many
  models will be trained and the results averaged between them. If training is showing high
  variance increasing this to 11 will make the training take longer but might have more accurate
  statistics.

Running an interactive shell
============================

Finally we should be ready to run things. Lets start first with the pyspark
REPL to see things are working::

	ssh stat1005.eqiad.wmnet
	cd mjolnir/
	venv/bin/mjolnir-utilities.py --config example_config.yaml shell

After a bunch of output, some warnings, perhaps a few exceptions printed out
(normal, they are usually related to trying to find a port to run the web ui
on), you will be greated with a prompt. It should look something like::

	Welcome to
	      ____              __
	     / __/__  ___ _____/ /__
	    _\ \/ _ \/ _ `/ __/  '_/
	   /__ / .__/\_,_/_/ /_/\_\   version 2.1.0
	      /_/

	Using Python version 2.7.9 (default, Jun 29 2016 13:08:31)
	SparkSession available as 'spark'.
	>>>

From here you can do anything you could do when programming mjolnir. This can be quite
useful for one-off tasks such as evaluating a previously trained model against a new
dataset, or splitting up an existing dataset into smaller pieces.

Running data_pipeline.py
========================

The commandline for kicking off the data pipeline looks like::

	cd ~/mjolnir
	venv/bin/mjolnir-utilities.py spark --config example_config.yaml collect enwiki

Providing enwiki at the very end limits data collection to a single wiki. Leave
this parameter off to collect data for all wikis configured in
example_config.yaml.

With the default configuration this will store the data in hdfs at::

    hdfs://analytics-hadoop/user/<username>/mjolnir/<Ymd>

Running training_pipeline.py
============================

The commandline for kicking off training looks like::

	venv/bin/mjolnir-utilities.py spark --config example_config.yaml train enwiki

Similar to the collection phase, providing enwiki at the very end limits
training to a single wiki.  Leave this parameter off to train for all
configured wikis (that exist in the data).

By default this will look for data in hdfs at the same location that the
`collect` script stores data. Because this uses the current date as the name it
will not work correctly the next day. With the `example_config.yaml` file you
can override this location to point at a previous day like so::

	venv/bin/mjolnir-utilities.py spark \
		--config example_config.yaml \
		--template-var training_data_path=user/ebernhardson/mjolnir/20171023 \
		train enwiki

Running both together
=====================

The commandline for kicking off running a full data collect and training in one
go looks like::

	venv/bin/mjolnir-utilities.py spark --config example_config.yaml collect_and_train enwiki

Same as before the final argument is the wiki to limit data collection and
training to.

Resource usage in the hadoop cluster when training
==================================================

If the training data all fits on a single executor, that is the most efficient
use of cluster resources. This may not be the fastest way to train individual
models, but if we are doing hyperparameter tuning we are generally training
many models in parallel, and the lowest total cpu time used per model comes
from using a single executor.

Training speed vs core count looks to stay relatively flat up to about 6 cores.
Less parallelism is again more efficient in terms of total efficiency of the
cluster, but up to 6 cores has a very minimal decrease. After 6 cores the
efficiency loss starts to increase at a greater rate.

Overall suggestions:

* Train models with 4 or 6 cores per executor
* Aim for a single executor if reasonable.
* Limitation: Cluster has ~2GB of memory per core, so training data (with
  duplicates, due to spark storage, task data, and xgboost DMatrix copy in CPP)
  needs to fit in 4*2 or 6*2 GB of memory. This is actually quite reasonable with
  our current feature size, but may need to be revisited is we dramatically
  increase the number of features used.

Other:

* Minimum amounts of memory that work fine for training a single model will
  overrun their memory allocation regularly when used to train in mjolnir with
  hyperparameter optimization. We need to over provision memory vs what it takes
  to spin up a spark instance and train a single model. Perhaps this is some sort
  of leak, or late de-allocation, in xgboost? unsure.

Help! There are exceptions eveywhere!
=====================================

Unfortunately spark is pretty spammy around worker shutdown. Spark executors
will, by default, shut down after being idle for 60 seconds unless they contain
cached RDD's. Often enough the nodes shutdown before the driver has completely
cleared it's internal state about the node and you get exceptions about a socket
timing out, or a broadcast variable not being able to be cleaned up.  These
exceptions are basically OK and don't indicate anything wrong. There are on the
other hand exceptions that do need to be paid attention to. Task failures are
always important. Executors killed by yarn for overrunning their memory limits
are also worth paying attention to, although if the rate is very low it is
sometimes acceptable.

An example of when to expect node shutdowns is during model training when a
mjolnir.training.hyperopt.minimize run is completing. We may spin up 100 or so
containers to run the minimization, but at the end we are waiting for a few
stragglers to finish up. The first executors to finish may side idle for more than
60 seconds waiting for the last executors to finish and shut themselves down.


