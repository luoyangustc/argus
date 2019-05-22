FROM reg.qiniu.com/avaprd/argus-base-video-base:20180719-v112-dev

RUN mkdir -p /workspace/argus && mkdir -p /workspace/argus/run/auditlog

ADD ./argus-segment /workspace/argus/argus-segment
ADD ./argus-segment.conf /workspace/argus/argus-segment.conf

ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./argus-segment -f ./argus-segment.conf
