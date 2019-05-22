FROM reg.qiniu.com/ava-public/ava-cuda8-cudnn6:1.0
LABEL maintainer "Qiniu ATLab <ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
# 这两个 NVIDIA source list 更新存在问题
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# apt-get && python && pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl vim git unzip cmake \
    libprotobuf-dev libleveldb-dev liblmdb-dev libsnappy-dev \
    libopencv-dev libhdf5-serial-dev libatlas-base-dev \
    libgflags-dev libgoogle-glog-dev libboost-all-dev protobuf-compiler \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev \
    python-dev python-numpy python-pip python-tk && \
    pip install -U pip setuptools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# opencv 3
RUN export OPENCV_CONTRIB_ROOT=/tmp/opencv-contrib OPENCV_ROOT=/tmp/opencv OPENCV_VER=3.2.0 && \
    git clone -b ${OPENCV_VER} --depth 1 https://github.com/opencv/opencv.git ${OPENCV_ROOT} && \
    git clone -b ${OPENCV_VER} --depth 1 https://github.com/opencv/opencv_contrib.git ${OPENCV_CONTRIB_ROOT} && \
    mkdir -p ${OPENCV_ROOT}/build && cd ${OPENCV_ROOT}/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_ICV_URL="http://devtools.dl.atlab.ai/docker/" \
    -DOPENCV_PROTOBUF_URL="http://devtools.dl.atlab.ai/docker/" \
    -DOPENCV_CONTRIB_BOOSTDESC_URL="http://devtools.dl.atlab.ai/docker/" \
    -DOPENCV_CONTRIB_VGG_URL="http://devtools.dl.atlab.ai/docker/" \
    -DINSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_ROOT}/modules \
    -DWITH_CUDA=ON -DBUILD_opencv_python2=ON -DBUILD_EXAMPLES=OFF .. && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/*

# framework
ENV CAFFE_ROOT=/opt/caffe
ENV CAFFE_CLONE_VER=action_recog
RUN cd /opt && git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j"$(nproc)" install && cd .. && rm -rf nccl && \
    mkdir -p ${CAFFE_ROOT} && cd ${CAFFE_ROOT} && git clone -b ${CAFFE_CLONE_VER} --depth 1 https://github.com/ataraxialab/caffe.git . && \
    cd python && for req in $(cat requirements.txt) pydot; do pip --no-cache-dir install $req --index-url https://pypi.tuna.tsinghua.edu.cn/simple; done && cd .. && \
    mkdir build && cd build && \
    cmake -DUSE_MPI=ON -DUSE_CUDNN=1 -DUSE_NCCL=1 .. && \
    make -j"$(nproc)" && make install

# ENV PYTHONPATH $PYFASTERRCNN_ROOT:...
ENV PYTHONPATH $CAFFE_ROOT/python:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$CAFFE_ROOT/python:/usr/local/cuda/bin:$PATH

RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple ava-sdk

# 将时区改成 GMT+8
RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8
LABEL com.qiniu.atlab.os = "ubuntu-16.04"
LABEL com.qiniu.atlab.type = "caffe.acttion"
