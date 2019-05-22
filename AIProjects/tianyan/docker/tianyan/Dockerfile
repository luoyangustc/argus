FROM ubuntu:16.04
RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates telnet wget curl vim unzip && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget http://devtools.dl.atlab.ai/docker/PRC-tz -O /tmp/PRC-tz && mv /tmp/PRC-tz /etc/localtime

RUN mkdir -p /workspace/tianyan

ADD ./tianyan /workspace/tianyan/tianyan
ADD ./tianyan.conf /workspace/tianyan/tianyan.conf

RUN mkdir -p /workspace/tianyan/run/auditlog
ENV PATH=$PATH:/workspace/tianyan

WORKDIR /workspace/tianyan

CMD ./tianyan -f ./tianyan.conf
