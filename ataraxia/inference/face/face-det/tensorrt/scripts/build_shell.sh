#!/bin/bash
##############################################################################
# Example command to build the whole project with shell.
##############################################################################
# 
# This script shows how one can build tron with shell.

TRON_ROOT="$(cd "$(dirname "$0")/../" && pwd)"
TRON_BUILD_ROOT=$TRON_ROOT/build

mkdir -p $TRON_ROOT/third_party
wget http://pbv7wun2s.bkt.clouddn.com/tron_fd_quality_ubuntu16.04_cuda9.0_cudnn7.1.3_lib_v0.0.2.1.tar  -O $TRON_ROOT/third_party/mix_lib.tar
tar -xf $TRON_ROOT/third_party/mix_lib.tar -C $TRON_ROOT/third_party
rm -rf $TRON_ROOT/third_party/mix_lib.tar

######################### Building Tron #########################
echo "============================================================"
echo "Building Tron ... "
echo "Building directory $TRON_BUILD_ROOT"
echo "============================================================"
if ! [ -d $TRON_BUILD_ROOT ]; then
  mkdir $TRON_BUILD_ROOT
fi
cd $TRON_BUILD_ROOT
cmake .. -DCMAKE_INSTALL_PREFIX=$TRON_BUILD_ROOT \
         -DCMAKE_BUILD_TYPE=Release \
         -DUSE_CUDA=ON \
         -DUSE_CUDNN=ON \
         -DUSE_BLAS=OFF \
         -DUSE_OpenCV=ON \
         -DBUILD_SHARED_LIBS=ON
make -j"$(nproc)" install
