#!/usr/bin/env bash
set -e

while getopts 's:p' opt; 
do	
    case ${opt} in
        s)
            SCENE="${OPTARG}";;
	      p) 
	          PACK="true";;
        ?)
            echo "Usage: `basename $0` [-s] scene_name [-v] select video case [-p] pack only not run case and download source"
	    exit 1
    esac
    #echo "opt is "${opt}", arg is "${OPTARG}", after index is "${OPTIND}  
done  

cd ..
source env.sh

make dependency

export AVA_TEST_ROOT=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

#生成scen test
echo "开始编译"
mkdir -p $AVA_TEST_ROOT/scene/
cd $AVA_TEST_ROOT/scene/
pwd
export GOOS=darwin && go test -c *_suite_test.go -o ./bin/testbin-$GOOS
export GOOS=linux && go test -c *_suite_test.go -o ./bin/testbin-$GOOS
echo "编译完成"

if [ "$PACK" != "true" ];then
#执行测试case及下载依赖资源
export OS=`uname -s | tr 'A-Z' 'a-z'`
#DEBUG=true TEST_STORE=true TEST_ENV=product TEST_ZONE=z0 ./bin/testbin-$OS -ginkgo.trace
DEBUG=true TEST_STORE=true TEST_ENV=private TEST_ZONE=z0 ./bin/testbin-$OS -ginkgo.dryRun
else
echo "跳过下载文件"
fi



#开始打包
echo "开始打包"
cd $AVA_TEST_ROOT
rm -rf test_out
mkdir test_out
cp script/run.sh test_out/
#cp configs/locale.z0.conf test_out/locale.z0.conf
cp configs/nginx.conf test_out/nginx.conf
cp $AVA_TEST_ROOT/scene/locale.z0.conf test_out/locale.z0.conf
if [ -d $AVA_TEST_ROOT/scene/testdata ];then
cp -rf $AVA_TEST_ROOT/scene/testdata test_out/
fi
cp -rf $AVA_TEST_ROOT/scene/bin  test_out/
if [ "$VIDEO" == "true" ];then
cp -rf $AVA_TEST_ROOT/src/qtest.com/argusvideo/bin/* test_out/bin/
fi

NAME="test"
cd $AVA_TEST_ROOT
DT=`date "+%Y%m%d%H%M%S"`

if [ -z "${TEST_NAME}" ]; then
    TEST_NAME="app.$DT"
fi

cp -Rf test_out $NAME && tar -zcvf $TEST_NAME $NAME && rm -rf $NAME && mv $TEST_NAME test_out/

echo "打包完成"
