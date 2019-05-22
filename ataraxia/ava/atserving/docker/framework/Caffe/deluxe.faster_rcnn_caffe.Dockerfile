FROM nvidia/cuda:8.0-cudnn5-devel
LABEL maintainer "Qiniu ATLab <ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
# 这两个 NVIDIA source list 更新存在问题
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# apt-get && python && pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget vim lrzsz curl git unzip build-essential cmake \
    python-dev python-pip python-opencv \
    libatlas-base-dev libopencv-dev libcurl4-openssl-dev \
    libboost-all-dev libgflags-dev libgoogle-glog-dev \
    libhdf5-serial-dev libleveldb-dev liblmdb-dev \
    libprotobuf-dev libsnappy-dev libopenblas-dev protobuf-compiler && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# framework
ENV CAFFE_ROOT=/opt/caffe 

WORKDIR $CAFFE_ROOT
RUN mkdir ${CAFFE_ROOT}_gpu && \
    ln -s ${CAFFE_ROOT}_gpu ${CAFFE_ROOT}

ADD models/caffe/faster_rcnn  ${CAFFE_ROOT}_gpu

RUN cd ${CAFFE_ROOT}_gpu && \
    pip --no-cache-dir install -U pip setuptools && pip install easydict && cd caffe/python && \
    for req in $(cat requirements.txt) pydot; do pip --no-cache-dir install $req --index-url https://mirrors.ustc.edu.cn/pypi/web/simple; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j"$(nproc)" install && cd .. && rm -rf nccl && \
    make -j"$(nproc)"  && \
    make py && \
    cd ../lib && make

ENV PYCAFFE_ROOT ${CAFFE_ROOT}/caffe/python
ENV PYTHONPATH $PYCAFFE_ROOT:${CAFFE_ROOT}/lib:$PYTHONPATH
# 默认使用 CPU
ENV PATH $CAFFE_ROOT/caffe/build/tools:$PYCAFFE_ROOT:$PATH
ENV USE_DEVICE=CPU
RUN echo "$CAFFE_ROOT/caffe/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
# 增加 dumb-init 和 entrypoint.sh 脚本
# file stored in qiniu://avatest@qiniu.com@z0/devtools/docker/dumb-init_1.2.0_amd64
RUN wget -O /usr/local/bin/dumb-init http://devtools.dl.atlab.ai/docker/dumb-init_1.2.0_amd64 && \
    chmod +x /usr/local/bin/dumb-init
COPY docker/framework/Caffe/deluxe-entrypoint.sh /workspace/deluxe-entrypoint.sh
ENTRYPOINT ["/workspace/deluxe-entrypoint.sh"]

# 将时区改成 GMT+8
RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8
LABEL com.qiniu.atlab.os = "ubuntu-16.04"
LABEL com.qiniu.atlab.type = "caffe"