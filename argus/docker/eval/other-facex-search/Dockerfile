FROM nvidia/cuda:8.0-cudnn6-devel
LABEL maintainer "Qiniu ATLab <ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
# 这两个 NVIDIA source list 更新存在问题
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# apt-get && python && pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget vim lrzsz curl git unzip build-essential cmake \
    python-dev python-pip  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# pip
RUN pip --no-cache-dir install -U pip setuptools && \
    pip --no-cache-dir install hickle nose pylint pyyaml numpy nose-timer requests

RUN export CUDAMAT_VERSION=master CUDAMAT_ROOT=/opt/cudamat && \
    git clone -b ${CUDAMAT_VERSION} --depth 1 https://github.com/cudamat/cudamat.git ${CUDAMAT_ROOT} && \
    cd ${CUDAMAT_ROOT} && pip install . && \
    mv /usr/local/lib/python2.7/dist-packages/cudamat/*.so $CUDAMAT_ROOT/cudamat/
ENV PYTHONPATH /opt/cudamat:$PYTHONPATH

# 增加 dumb-init 和 entrypoint.sh 脚本
# file stored in qiniu://avatest@qiniu.com@z0/devtools/docker/dumb-init_1.2.0_amd64
RUN wget -O /usr/local/bin/dumb-init http://devtools.dl.atlab.ai/docker/dumb-init_1.2.0_amd64 && \
    mkdir /ava && wget -O /ava/entrypoint.sh http://devtools.dl.atlab.ai/docker/entrypoint.sh && \
    chmod +x /usr/local/bin/dumb-init /ava/entrypoint.sh
ENTRYPOINT ["/ava/entrypoint.sh"]

# 将时区改成 GMT+8
RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
ENV LC_ALL=C.UTF-8
LABEL com.qiniu.atlab.os = "ubuntu-16.04"

RUN mkdir -p /workspace/serving
RUN mkdir -p /workspace/serving/run/auditlog
RUN mkdir -p /workspace/serving/data
RUN mkdir -p /tmp/eval

RUN pip install prometheus_client

ADD serving-eval /workspace/serving/serving-eval
ADD serving-eval.conf /workspace/serving/serving-eval.conf

ADD evals /workspace/serving/python/evals

ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV PYTHONPATH=/workspace/serving/python:$PYTHONPATH

WORKDIR /workspace/serving
LABEL com.qiniu.atlab.biz = "ava-eval-facex-search"

CMD ["serving-eval","-f","serving-eval.conf"]
