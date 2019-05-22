#!/usr/bin/env bash
set -ex
cd /src
export PYTHONPATH=$PYTHONPATH:/src/python
cd /src/python
python2 -m pylint aisdk --rcfile=../pylintrc
python2 -m yapf -d -r .
