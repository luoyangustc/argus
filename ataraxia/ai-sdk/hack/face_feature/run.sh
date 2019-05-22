#!/usr/bin/env bash
set -ex

SDK_ROOT="$(cd "$(dirname "$0")/../../cpp" && pwd)"
SDK_BUILD_ROOT=$SDK_ROOT/build

if ! [ -d $SDK_BUILD_ROOT ]; then
  mkdir $SDK_BUILD_ROOT
fi
cd $SDK_BUILD_ROOT
cmake .. -DCMAKE_INSTALL_PREFIX=$SDK_BUILD_ROOT \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_SHARED_LIBS=ON
if [ "P"`uname` = "PDarwin" ];
then
  make
else
  make -j"$(nproc)"
fi

# sh /src/app_cpp/face_detect/scripts/build_shell.sh 
/src/cpp/build/bin/test_ff
