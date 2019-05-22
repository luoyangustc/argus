FROM reg.qiniu.com/avaprd/argus-base-video-base:201808091633-v452-dev

RUN mkdir -p /workspace/argus && mkdir -p /workspace/argus/run/auditlog && mkdir -p /workspace/video && chmod 777 /workspace/video

ADD ./argus-video-private /workspace/argus/argus-video-private
ADD ./argus-video-private.conf /workspace/argus/argus-video-private.conf

ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ["argus-video-private", "-f","argus-video-private.conf"]
