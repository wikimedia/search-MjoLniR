# Example build: docker build -t mjolnir:latest
FROM docker-registry.wikimedia.org/releng/ci-stretch

ENV VIRTUAL_ENV=/opt/venv

# TODO: build python-snappy wheel on a separate container instead of installing build-essential
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3.5-dev python3-virtualenv libsnappy1v5 libsnappy-dev build-essential && \
    python3 -m virtualenv --python /usr/bin/python3 $VIRTUAL_ENV && \
    $VIRTUAL_ENV/bin/pip install python-snappy && \
    apt-get remove -y build-essential python3.5-dev libsnappy-dev && \
    apt-get autoremove -y

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY setup.py README.rst mjolnir/
COPY mjolnir/ mjolnir/mjolnir/

RUN pip install mjolnir/
