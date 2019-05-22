#!/bin/bash
##############################################################################
# Example command to build the whole project with shell.
##############################################################################
# 
# This script shows how one can build tron with shell.

TRON_ROOT="$(cd "$(dirname "$0")/../" && pwd)"
TRON_BUILD_ROOT=$TRON_ROOT/build

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
         -DUSE_CUDA=OFF   \
         -DUSE_CUDNN=OFF \
         -DUSE_BLAS=ON  \
         -DUSE_OpenCV=ON \
         -DBUILD_SHARED_LIBS=ON
make -j"$(nproc)" install
