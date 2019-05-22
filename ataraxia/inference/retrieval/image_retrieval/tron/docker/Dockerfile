FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
LABEL maintainer = "Qiniu ATLab<ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    curl \
    vim \
    git \
    ca-certificates \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

########## INSTALLATION STEPS ###################
ENV TRON_ROOT=/opt/tron
COPY . $TRON_ROOT
RUN sh $TRON_ROOT/scripts/build_shell.sh && \
    cp $TRON_ROOT/build/lib/linux/x86_64/libinference.so /usr/lib && \
    cp $TRON_ROOT/build/lib/linux/x86_64/libshadow.so /usr/lib && \
    mkdir -p /workspace/serving && \
    ln -s /usr/lib/libinference.so /workspace/serving/inference.so && \
    rm -rf $TRON_ROOT

RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8
LABEL com.qiniu.atlab.os = "ubuntu-16.04"
LABEL com.qiniu.atlab.type = "tron"
