FROM ubuntu:16.04 as builder
RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends make gcc libssl-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD onvif_porting /tmp

RUN cd /tmp && make

FROM ubuntu:16.04 

RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates telnet wget curl vim unzip && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget http://devtools.dl.atlab.ai/docker/PRC-tz -O /tmp/PRC-tz && mv /tmp/PRC-tz /etc/localtime

RUN mkdir -p /workspace/tianyan

COPY --from=builder /tmp/Onvif_Discover /workspace/tianyan/Onvif_Discover

ENV PATH=$PATH:/workspace/tianyan

WORKDIR /workspace/tianyan

CMD ./Onvif_Discover 0.0.0.0 4002
