#!/bin/sh

set -e

# Need backports for openjdk-8
echo "deb http://ftp.debian.org/debian jessie-backports main contrib" > /etc/apt/sources.list.d/backports.list

apt-get update
apt-get install -q -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
    openjdk-8-jre-headless \
    openjdk-8-jdk \
    ca-certificates-java='20161107~bpo8+1' \
    python-virtualenv \
    git-core \
    build-essential \
    maven \
    liblapack-dev \
    python-dev \
    gfortran \
    zip


# While we only asked for java 8, 7 was installed as well. switch over the
# alternative. TODO: Do we need anything else?
update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

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
    cd jvm-packages
    # The test suite requires 4 cores or it gets stuck. Not ideal but skip them for
    # now. It also needs a bit more than the default heap allocation.
    MAVEN_OPTS="-Xmx768M" mvn -DskipTests -Dmaven.test.skip=true install
fi

# Build the mjolnir jar which depends on xgboost4j-spark
cd /vagrant/jvm
mvn package

