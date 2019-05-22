FROM ubuntu:16.04
RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates telnet wget curl vim unzip libglu1 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget http://devtools.dl.atlab.ai/docker/PRC-tz -O /tmp/PRC-tz && mv /tmp/PRC-tz /etc/localtime

RUN mkdir -p /workspace/chengyun
RUN wget -q http://oq2qpeuia.bkt.clouddn.com/HK_libs.tar.gz && tar -xzvf HK_libs.tar.gz && cp ./HK_libs/* /usr/local/lib && ldconfig && rm -rf HK_libs.tar.gz HK_libs
RUN wget -q http://oq2qpeuia.bkt.clouddn.com/ffmpeg -O /workspace/chengyun/ffmpeg && chmod +x /workspace/chengyun/ffmpeg
RUN wget -q http://oq2qpeuia.bkt.clouddn.com/Extra_libs.tar.gz && tar -xzvf Extra_libs.tar.gz && cp ./Extra_libs/* /usr/local/lib && ldconfig && rm -rf Extra_libs.tar.gz Extra_libs
RUN wget -q http://oq2qpeuia.bkt.clouddn.com/DvrDL_20180626 -O /workspace/chengyun/DvrDL && chmod +x /workspace/chengyun/DvrDL && wget -q http://oq2qpeuia.bkt.clouddn.com/cfg.txt -O /workspace/chengyun/cfg.txt
RUN wget -q http://oq2qpeuia.bkt.clouddn.com/ExtractFrames_20180337_v0 -O /workspace/chengyun/ExtractFrames && chmod +x /workspace/chengyun/ExtractFrames
RUN wget -q http://oq2qpeuia.bkt.clouddn.com/ExtractVideo_20180328_v0 -O /workspace/chengyun/ExtractVideo && chmod +x /workspace/chengyun/ExtractVideo

ADD ./chengyun /workspace/chengyun/chengyun
ADD ./chengyun.conf /workspace/chengyun/chengyun.conf

RUN mkdir -p /workspace/chengyun/run/auditlog
ENV PATH=$PATH:/workspace/chengyun

WORKDIR /workspace/chengyun

CMD ./chengyun -f ./chengyun.conf
