#!/usr/bin/env bash
set -ex
cd /src
export PYTHONPATH=$PYTHONPATH:/src/python
python2 -m pytest -s -l -v --benchmark-disable --cov=aisdk.common --cov-report html --cov-report term --pyargs aisdk.common
python3 -m pytest -s -l -v --benchmark-disable --cov=aisdk.common --cov-report html --cov-report term --pyargs aisdk.common
