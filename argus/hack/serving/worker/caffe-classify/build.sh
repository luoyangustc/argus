#!/usr/bin/env bash
set -ex

rm -rf platform
cp -r $QBOXROOT/ava/platform platform

rm -rf evals
mkdir evals
cp -r $QBOXROOT/ava/docker/scripts/evals/utils evals/
cp -r $QBOXROOT/ava/docker/scripts/evals/caffe_base evals/
cp $QBOXROOT/ava/docker/scripts/evals/__init__.py evals/
cp $QBOXROOT/ava/docker/scripts/evals/caffe_classify.py evals/eval.py

echo $TAR
cp $TAR model.tar
echo $FILES
rm -rf example.files
cp -rf $FILES example.files

docker build -t ava-caffe-classify:local -f Dockerfile .
rm -rf platform
rm -rf evals
rm model.tar
rm -rf example.files

docker run --rm ava-caffe-classify:local "bash example.sh"
docker run -dt ava-caffe-classify:local
