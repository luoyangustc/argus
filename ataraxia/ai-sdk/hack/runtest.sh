#!/usr/bin/env bash
set -ex
cd /src
export PYTHONPATH=$PYTHONPATH:/src/res/python-test-dep
python2 -m pytest --cov=aisdk.app.$APP --cov-report html --cov-report term -s -l -v --pyargs aisdk.app.$APP
# python2 -m pytest -s -v --pyargs aisdk.app.$APP
