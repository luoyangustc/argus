FROM ubuntu:16.04
RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates telnet wget curl vim unzip && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget http://devtools.dl.atlab.ai/docker/PRC-tz -O /tmp/PRC-tz && mv /tmp/PRC-tz /etc/localtime

RUN mkdir -p /workspace/zhongxing

ADD ./zhongxing /workspace/zhongxing/zhongxing
ADD ./zhongxing.conf /workspace/zhongxing/zhongxing.conf

RUN mkdir -p /workspace/zhongxing/run/auditlog
ENV PATH=$PATH:/workspace/zhongxing

WORKDIR /workspace/zhongxing

CMD ./zhongxing -f ./zhongxing.conf
