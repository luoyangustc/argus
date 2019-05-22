#!/usr/bin/env bash
set -e

unset TEST_ENV
unset TEST_STORE

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILE_SERVER_PORT=18888
FILE_SERVER_NAME="private-test-file-server"

while getopts 'nv:e' opt; 
do	
    case ${opt} in
        n)
            UNNGINX="true";;
        v)
            VIDEO="${OPTARG}";;
        e)
            export TEST_ENV="private";;
        ?)
            echo "Usage: `basename $0` [-v] terror|pulp|politician "
	        exit 1
    esac
    #echo "opt is "${opt}", arg is "${OPTARG}", after index is "${OPTIND}  
done  


export AVA_TEST_ROOT=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

export TESTROOT=${AVA_TEST_ROOT}



export OS=`uname -s | tr 'A-Z' 'a-z'`

# start file server
if [ "$UNNGINX" != "true" ];then
echo "start file server"
docker rm --force ${FILE_SERVER_NAME} > /dev/null 2>&1 || true
docker run --name ${FILE_SERVER_NAME} -d \
    -p ${FILE_SERVER_PORT}:80 \
    -v ${CUR_DIR}/nginx.conf:/etc/nginx/nginx.conf:ro \
    -v ${CUR_DIR}/testdata:/test:ro \
    reg.qiniu.com/library/nginx:latest nginx-debug -g 'daemon off;' > /dev/null 2>&1

# update conf
sed -i'' 's#source=.*#source="http://localhost:'${FILE_SERVER_PORT}'/test/argusvideo/"#' locale.z0.conf
fi
cat locale.z0.conf

mkdir -p reporters

if [ "x$VIDEO" == "x" ]; then
    DEBUG=true TEST_ZONE=z0 ./bin/testbin-$OS -ginkgo.trace -ginkgo.v
else
    DEBUG=true TEST_ZONE=z0 ./bin/testbin-$OS -ginkgo.trace -ginkgo.v -ginkgo.focus ${OP}"]"
fi

echo "stop file server"
docker rm --force ${FILE_SERVER_NAME} > /dev/null 2>&1

echo "test case 执行完成!"
echo "SUCCESS"