FROM reg.qiniu.com/avaprd/argus-base-video-base:20180719-v112-dev

RUN mkdir -p /workspace/argus/run/auditlog && mkdir -p /workspace/video && chmod 777 /workspace/video

ADD ./inference-video-worker /workspace/argus/inference-video-worker
ADD ./inference-video-worker.conf /workspace/argus/inference-video-worker.conf

ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./inference-video-worker -f ./inference-video-worker.conf
