#!/usr/bin/env bash
set -ex

echo $TAR
cp $TAR model.tar
echo $FILES
rm -rf example.files
cp -rf $FILES example.files

rm -rf evals
cp -r ../../../../docker/scripts/evals .

docker build -t ava-caffe-classify:local -f Dockerfile .
rm -rf evals
rm model.tar
rm -rf example.files

docker run --rm ava-caffe-classify:local "bash example.sh"
docker run -dt ava-caffe-classify:local
