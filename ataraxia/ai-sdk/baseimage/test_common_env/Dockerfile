FROM ubuntu:16.04
RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list 
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    python-pip \
    python-dev \
    python-setuptools \
    python3-pip \
    python3-dev \
    python3-setuptools
# opencv dep
RUN apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext-dev
RUN python2 -m pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip wheel
RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip wheel

RUN python2 -m pip install -i https://mirrors.aliyun.com/pypi/simple mxnet gevent
RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple mxnet gevent

ADD requirements.txt /dep/requirements.txt
RUN python2 -m pip install -i https://mirrors.aliyun.com/pypi/simple -r /dep/requirements.txt
RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple -r /dep/requirements.txt
