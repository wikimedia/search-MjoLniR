#!/bin/sh

set -e

apt-get update
apt-get install -q -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
    openjdk-7-jre-headless \
    python-virtualenv

cd /opt
wget -qO - http://d4kbcqa49mib13.cloudfront.net/spark-2.1.0-bin-hadoop2.6.tgz | tar -zxvf
ln -s /opt/spark-2.1.0-bin-hadoop2.6/bin/pyspark /usr/local/bin
# findspark needs a SPARK_HOME to setup pyspark
cat >/etc/profile.d/spark.sh <<EOD
SPARK_HOME=/opt/spark-2.1.0-bin-hadoop2.6
export SPARK_HOME
EOD

# pyspark wants to put a metastore_db directory in /vagrant, put it somewhere else
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

# pyspark wants to put a derby.log in /vagrant as well, put it elsewhere
cat >> /opt/spark-2.1.0-bin-hadoop2.6/conf/spark-defaults.conf <<EOD
spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp/derby.log
EOD

if [ ! -d /vagrant/venv ]; then
    cd /vagrant
    mkdir venv
    virtualenv -p /usr/bin/python2.7 venv
    venv/bin/pip install tox
fi
