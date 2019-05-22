#!/bin/bash

set -x

if [ $# -ne 1 ] ; then
    echo "usage: ./build.sh version"
    exit -1
fi

cd ..
export GOPATH=$GOPATH:$(pwd)

echo "GOAPATH: $GOPATH"

imageName=index.qiniu.com/ataraxia/ufop-proxy
backendFile=main
echo "build main..."
echo "======================="
GOOS=linux GOARCH=amd64 go build -o $backendFile src/main.go

cd -
rm -rf files
mkdir -p files
mv ../$backendFile files/
cp ../config/config.json files/

echo "begin build images"

imageTag=$1
echo "delete container"
containers=`docker ps -a | grep $imageName":"$imageTag`
if [ -n "$containers" ]; then
    for c in `docker ps -a | grep $imageName":"$imageTag | awk '{print $1}'`;
    do
        echo "remove container $c"
        docker rm $c
    done
fi

echo "delete container end, check result:"
docker ps -a | grep $imageName":"$imageTag

echo "=============="
echo "delete image"
for i in `docker images | grep $imageName":"$imageTag | awk '{print $3}'`;
do
    echo "delete image $i"
    docker rmi $i
done

echo "delete image end, check result:"
docker images | grep $imageName":"$imageTag

echo "=================="
echo "build image start"

docker build --no-cache -t="$imageName:$imageTag" .

echo "build end , result:"
docker images | grep $imageName
