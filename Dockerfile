# Example build: docker build -t mjolnir:latest
FROM docker-registry.wikimedia.org/releng/ci-stretch

ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-virtualenv && \
    python3 -m virtualenv --python /usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY setup.py README.rst mjolnir/
COPY mjolnir/ mjolnir/mjolnir/

RUN pip install mjolnir/
