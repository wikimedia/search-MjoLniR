#!/bin/sh

set -e

cat >/etc/apt/sources.list.d/cloudera.list <<EOD
# Packages for Cloudera's Distribution for Hadoop, Version 5.10.0, on Ubuntu 14.04 amd64       
deb [arch=amd64] http://archive.cloudera.com/cdh5/ubuntu/trusty/amd64/cdh trusty-cdh5.10.0 contrib
deb-src http://archive.cloudera.com/cdh5/ubuntu/trusty/amd64/cdh trusty-cdh5.10.0 contrib
EOD

cat >/etc/apt/preferences.d/cloudera.pref <<EOD
Package: *
Pin: release o=Cloudera, l=Cloudera
Pin-Priority: 501
EOD

wget -q https://archive.cloudera.com/cdh5/ubuntu/trusty/amd64/cdh/archive.key -O /root/cloudera-archive.key
apt-key add /root/cloudera-archive.key

apt-get update
apt-get install -q -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
    spark-python \
    openjdk-7-jre-headless \
    python-virtualenv

cat >/etc/profile.d/spark.sh <<EOD
SPARK_HOME=/usr/lib/spark
export SPARK_HOME
EOD

cd /vagrant
virtualenv .
bin/pip install -r requirements.txt
