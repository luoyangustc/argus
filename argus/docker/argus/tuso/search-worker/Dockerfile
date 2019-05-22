FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus
COPY /search-worker /workspace/argus/search-worker
ADD ./tuso-search-worker.conf /workspace/argus/tuso-search-worker.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./search-worker -f ./tuso-search-worker.conf
