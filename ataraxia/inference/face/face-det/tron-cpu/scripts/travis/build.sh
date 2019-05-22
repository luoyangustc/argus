#!/bin/bash

set -e
set -x

TRON_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
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
TRON_CMAKE_ARGS=('-DCMAKE_INSTALL_PREFIX=.')
TRON_CMAKE_ARGS+=('-DCMAKE_BUILD_TYPE=Release')
if [ "$BUILD_CUDA" = 'true' ]; then
    TRON_CMAKE_ARGS+=('-DUSE_CUDA=ON')
else
    TRON_CMAKE_ARGS+=('-DUSE_CUDA=OFF')
fi
if [ "$BUILD_CUDNN" = 'true' ]; then
    TRON_CMAKE_ARGS+=('-DUSE_CUDNN=ON')
else
    TRON_CMAKE_ARGS+=('-DUSE_CUDNN=OFF')
fi
if [ "$BUILD_OpenCV" = 'true' ]; then
    TRON_CMAKE_ARGS+=('-DUSE_OpenCV=ON')
else
    TRON_CMAKE_ARGS+=('-DUSE_OpenCV=OFF')
fi
if [ "$BUILD_SHARED_LIBS" = 'true' ]; then
    TRON_CMAKE_ARGS+=('-DBUILD_SHARED_LIBS=ON')
else
    TRON_CMAKE_ARGS+=('-DBUILD_SHARED_LIBS=OFF')
fi
cmake .. ${TRON_CMAKE_ARGS[*]}
if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
    make "-j$(nproc)" install
elif [ "$TRAVIS_OS_NAME" = 'osx' ]; then
    make "-j$(sysctl -n hw.ncpu)" install
fi
