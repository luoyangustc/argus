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
RUN mkdir -p /workspace/CPIO && wget -O /workspace/CPIO/cpio-2.12.tar.gz http://p3gvzhggv.bkt.clouddn.com/cpio-2.12.tar.gz && cd /workspace/CPIO && tar xvf cpio-2.12.tar.gz && cd cpio-2.12 && ./configure && make && make install
RUN mkdir -p /workspace/MKL && curl -o /workspace/MKL/lntel_mkl_2018.2.199.tar -H'host:p3gvzhggv.bkt.clouddn.com' http://iovip.qbox.me/lntel_mkl_2018.2.199.tar &&  cd /workspace/MKL && tar xvf lntel_mkl_2018.2.199.tar && rm lntel_mkl_2018.2.199.tar
RUN cd /workspace/MKL/l_mkl_2018.2.199 && mv silent.cfg silent.cfg.example && sed '9c ACCEPT_EULA=accept' silent.cfg.example > silent1.cfg && sed '36c ARCH_SELECTED=INTEL64' silent1.cfg > silent.cfg && ./install.sh -s silent.cfg && rm -r /opt/intel/mkl/lib/ia32* && mv /opt/intel/mkl/lib/intel64_lin /opt/intel/mkl/lib/intel64_lin.bak && mkdir -p /opt/intel/mkl/lib/intel64_lin && mv /opt/intel/mkl/lib/intel64_lin.bak/libmkl_sequential.so /opt/intel/mkl/lib/intel64_lin && mv /opt/intel/mkl/lib/intel64_lin.bak/libmkl_intel_lp64.so /opt/intel/mkl/lib/intel64_lin && mv /opt/intel/mkl/lib/intel64_lin.bak/libmkl_core.so /opt/intel/mkl/lib/intel64_lin && mv /opt/intel/mkl/lib/intel64_lin.bak/libmkl_avx2.so /opt/intel/mkl/lib/intel64_lin && rm -r /opt/intel/mkl/lib/intel64_lin.bak
RUN echo "/opt/intel/lib/intel64" >> /etc/ld.so.conf.d/intel_mkl.conf && \
    echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel_mkl.conf && \
    ldconfig

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
