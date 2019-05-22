FROM mxnet/python:1.3.0_gpu_cu80
RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list && sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN apt-get update && apt-get install  -y --no-install-recommends libprotobuf-dev libsnappy-dev libopenblas-dev protobuf-compiler libopencv-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get clean
RUN pip install -i https://mirrors.aliyun.com/pypi/simple protobuf pyzmq gevent opencv-python-headless==4.0.0.21 && python -c 'import cv2;assert cv2.__version__[0]=="4"'
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
LABEL com.qiniu.atlab.type = "mxnet"
