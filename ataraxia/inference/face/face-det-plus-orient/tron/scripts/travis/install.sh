#!/bin/bash

set -e
set -x

ROOT_DIR="$(cd "$(dirname "$0")/../../" && pwd)"
cd $ROOT_DIR

APT_INSTALL_CMD='sudo apt-get install -y --no-install-recommends --allow-unauthenticated'

if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
    sudo apt-get update
    $APT_INSTALL_CMD \
        build-essential \
        ca-certificates \
        apt-transport-https \
        gnupg-curl \
        cmake \
        libopencv-dev \
        libprotobuf-dev \
        protobuf-compiler

    sudo rm -rf /var/lib/apt/lists/*
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub 
    sudo apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | sudo tail -n +2 > cudasign.pub 
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sudo sha256sum -c --strict - && sudo rm cudasign.pub 
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /" > local.txt
    sudo cp local.txt /etc/apt/sources.list.d/cuda.list


    CUDA_VERSION=8.0.61
    CUDA_PKG_VERSION=8-0=$CUDA_VERSION-1

    sudo apt-get update
    $APT_INSTALL_CMD \
        cuda-nvrtc-$CUDA_PKG_VERSION \
        cuda-nvgraph-$CUDA_PKG_VERSION \
        cuda-cusolver-$CUDA_PKG_VERSION \
        cuda-cublas-8-0=8.0.61.2-1 \
        cuda-cufft-$CUDA_PKG_VERSION \
        cuda-curand-$CUDA_PKG_VERSION \
        cuda-cusparse-$CUDA_PKG_VERSION \
        cuda-npp-$CUDA_PKG_VERSION \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-8.0 /usr/local/cuda && \
    sudo rm -rf /var/lib/apt/lists/*


    if [ "$BUILD_CUDNN" = 'true' ]; then
        CUDNN_REPO_PKG='nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb'
        CUDNN_PKG_VERSION='7.0.5.15-1+cuda8.0'
        wget "https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/${CUDNN_REPO_PKG}"
        sudo dpkg -i "$CUDNN_REPO_PKG"
        rm -f "$CUDNN_REPO_PKG"
        sudo apt-get update
        $APT_INSTALL_CMD \
            "libcudnn7=${CUDNN_PKG_VERSION}" \
            "libcudnn7-dev=${CUDNN_PKG_VERSION}"
    fi
elif [ "$TRAVIS_OS_NAME" = 'osx' ]; then
    brew update
    pip uninstall -y numpy  # use brew version (opencv dependency)
    brew install opencv protobuf
else
    echo "OS \"$TRAVIS_OS_NAME\" is unknown"
    exit 1
fi
