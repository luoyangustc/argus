FROM ubuntu:16.04
RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates telnet wget curl vim unzip libcurl3 && apt-get clean libcurl4-openssl-dev && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget http://devtools.dl.atlab.ai/docker/PRC-tz -O /tmp/PRC-tz && mv /tmp/PRC-tz /etc/localtime

RUN mkdir -p /workspace/video-stream-backend

ADD ./video-stream-backend /workspace/video-stream-backend/video-stream-backend
ADD ./video-stream-backend.conf /workspace/video-stream-backend/video-stream-backend.conf

RUN mkdir -p /workspace/video-stream-backend/run/auditlog
ENV PATH=$PATH:/workspace/video-stream-backend

WORKDIR /workspace/video-stream-backend

RUN wget http://7xt47g.com1.z0.glb.clouddn.com/ffmpeg-video-stream-0622 -O /workspace/video-stream-backend/ffmpeg-ai
RUN chmod +x /workspace/video-stream-backend/ffmpeg-ai

CMD ./video-stream-backend -f ./video-stream-backend.conf
