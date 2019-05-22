FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./inference-image-worker /workspace/argus/inference-image-worker
ADD ./inference-image-worker.conf /workspace/argus/inference-image-worker.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./inference-image-worker -f ./inference-image-worker.conf
