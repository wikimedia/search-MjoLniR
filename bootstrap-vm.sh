#!/bin/sh

set -e

apt install -q -y software-properties-common
# Confluent is needed for installing kafka, which requires zookeeper and some other stuff
# This intentionally uses an older version to get kafka 0.9, to match production
sudo add-apt-repository "deb [arch=amd64] http://packages.confluent.io/deb/2.0 stable main"
wget -qO - https://packages.confluent.io/deb/2.0/archive.key | sudo apt-key add -

# Need backports for openjdk-8
echo "deb http://ftp.debian.org/debian jessie-backports main contrib" > /etc/apt/sources.list.d/backports.list

apt-get update
apt-get install -q -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
    openjdk-8-jre-headless \
    openjdk-8-jdk \
    confluent-kafka-2.11.7 \
    ca-certificates-java='20161107~bpo8+1' \
    python-virtualenv \
    git-core \
    build-essential \
    maven \
    liblapack-dev \
    python-dev \
    gfortran \
    zip

# xgboost master requires cmake > 3.2, so we need it from jessie-backports
apt-get -t jessie-backports install cmake cmake-data


# While we only asked for java 8, 7 was installed as well. switch over the
# alternative. TODO: Do we need anything else?
update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

# Setup kafka and zookeeper to start on boot. For whatever reason
# confluent doesn't set this up for us.
echo > /lib/systemd/system/zookeeper.service <<EOD
[Unit]
Description="zookeeper"

[Service]
Environment="KAFKA_HEAP_OPTS=-Xmx256m -Xms256m"
ExecStart=/usr/bin/zookeeper-server-start /etc/kafka/zookeeper.properties
ExecStop=/usr/bin/zookeeper-server-stop

[Install]
WantedBy=multi-user.target
EOD
echo > /lib/systemd/system/kafka.service <<EOD
[Unit]
Description="kafka"
After=zookeeper.service
Wants=zookeeper.service

[Service]
Environment="KAFKA_HEAP_OPTS=-Xmx256m -Xms256m"
ExecStart=/usr/bin/kafka-server-start /etc/kafka/server.properties
ExecStop=/usr/bin/kafka-server-stop

[Install]
WantedBy=multi-user.target
EOD

systemctl daemon-reload
systemctl enable zookeeper.service
systemctl enable kafka.service
systemctl start zookeeper.service
systemctl start kafka.service

# Grab spark 2.1.0 and put it in /opt
cd /opt
if [ ! -f /usr/local/bin/pyspark ]; then
    wget -qO - https://archive.apache.org/dist/spark/spark-2.1.0/spark-2.1.0-bin-hadoop2.6.tgz | tar -zxvf -
    ln -s /opt/spark-2.1.0-bin-hadoop2.6/bin/pyspark /usr/local/bin
fi
# findspark needs a SPARK_HOME to setup pyspark
cat >/etc/profile.d/spark.sh <<EOD
SPARK_HOME=/opt/spark-2.1.0-bin-hadoop2.6
export SPARK_HOME
EOD

# pyspark wants to put a metastore_db directory in your cwd, put it somewhere
# else
cat >/opt/spark-2.1.0-bin-hadoop2.6/conf/hive-site.xml <<EOD
<configuration>
   <property>
      <name>hive.metastore.warehouse.dir</name>
      <value>/tmp/</value>
      <description>location of default database for the warehouse</description>
   </property>
   <property>
      <name>javax.jdo.option.ConnectionURL</name>
      <value>jdbc:derby:;databaseName=/tmp/metastore_db;create=true</value>
   </property>
</configuration>
EOD

# pyspark wants to put a derby.log in cwd as well, put it elsewhere
cat >> /opt/spark-2.1.0-bin-hadoop2.6/conf/spark-defaults.conf <<EOD
spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp/derby.log
EOD

if [ ! -d /vagrant/venv ]; then
    cd /vagrant
    virtualenv -p /usr/bin/python2.7 venv
    venv/bin/pip install tox
fi

# Grab and compile xgboost. install it into local maven repository
# as they don't publish to maven central yet.
if [ ! -d /srv/xgboost ]; then
    git clone https://github.com/dmlc/xgboost.git /srv/xgboost
fi
cd /srv/xgboost
# We need d3b866e, da58f34, ccccf8a0 and 197a9eac from master which don't cherry-pick
# cleanly back to the last released tag (v0.60), so use a hardcoded version of
# master branch that we think works.
if [ ! -f /srv/xgboost/jvm-packages/xgboost4j-spark/target/xgboost4j-spark-0.7.jar ]; then
    git checkout 197a9eac
    git submodule update --init --recursive
    # TODO: We also need the patch that fixes parallel-CV. XGBoost has a race condition
    # triggered when running many models in parallel, but the current fix is more of
    # a hack and wont be merged upstream.
    git remote add ebernhardson https://github.com/ebernhardson/xgboost
    git fetch ebernhardson
    git cherry-pick ebernhardson/booster_to_bytes
    cd jvm-packages
    # The test suite requires 4 cores or it gets stuck. Not ideal but skip them for
    # now. It also needs a bit more than the default heap allocation.
    MAVEN_OPTS="-Xmx768M" mvn -DskipTests -Dmaven.test.skip=true install
fi

# Build the mjolnir jar which depends on xgboost4j-spark
cd /vagrant/jvm
mvn package

