#!/bin/bash

if [ $# -ne 4 ]; then
    echo "usage:$0 image version ufop-account expect-count"
    exit 1
fi


#==============================================================================
# build content
#==============================================================================
cd ..
echo `pwd`
export GOPATH=$GOPATH:$(pwd)

echo "GOAPATH: $GOPATH"

backendFile=main
echo "build main..."
echo "======================="
go build -o $backendFile src/main.go

cd -
echo `pwd`
rm -rf files
mkdir -p files
mv ../config/config.json.template ../config/config.json
sed -in -e "s/{ak}/$PULP_AK/" \
    -e "s/{sk}/$PULP_SK/" \
    -e "s/{name}/$BUCKET_NAME/" \
    -e "s/{domain}/$BUCKET_DOMAIN/" \
    ../config/config.json

cat ../config/config.json

mv ../$backendFile files/
cp ../config/config.json files/

image=$1
version=$2
ufopName=$3
expectCount=$4

echo "=================="
echo "build image start"
docker build --no-cache -t="$image:$version" .
rm -rf files

#############################
echo "=================="
echo " ufop oerations"
wget http://q.hi-hi.cn/qdoractl_linux_0.3.6
mv qdoractl_linux_0.3.6 qdoractl
chmod u+x qdoractl

# qdoractl: login
./qdoractl login --ak $PULP_AK --sk $PULP_SK
if [ $? -ne 0 ]; then
    echo "$LINENO error: $CMD login failed"
    exit -1
fi
if [ -z '$(./qdoractl list | grep $ufopName)' ]; then
	./qdoractl register $ufopName
    if [ $? -ne 0 ];then
        echo "$LINENO error: reigster $ufopName to ufop failed"
        exit 1
	fi
fi

./qdoractl push $image:$version
if [ $? -ne 0 ];then
        echo "$LINENO error: push image $image:$version to ufop failed"
        exit 1
fi

./qdoractl release --mkconfig .
if [ $? -ne 0 ];then
        echo "$LINENO error: create configure failed"
        exit 1
fi

sed -in -e "s/you-ufop-name/$ufopName/" \
	-e "s/v1/\"$version\"/" \
    -e "s|you-app:1.0|$image:$version|" \
    dora.yaml
cat dora.yaml

./qdoractl release --config .
if [ $? -ne 0 ];then
        echo "$LINENO error: configure failed"
        exit 1
fi

./qdoractl deploy $ufopName $version --region z0 --expect $expectCount
if [ $? -ne 0 ];then
        echo "$LINENO error: deploy failed"
        exit 1
fi

# see the result
./qdoractl release $ufopName $version -d
