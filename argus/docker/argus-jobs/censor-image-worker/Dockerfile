FROM reg.qiniu.com/avaprd/argus-base/serving-base:20180713-v63-dev

RUN mkdir -p /workspace/argus

ADD ./censor-image-worker /workspace/argus/censor-image-worker
ADD ./censor-image-worker.conf /workspace/argus/censor-image-worker.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./censor-image-worker -f ./censor-image-worker.conf
