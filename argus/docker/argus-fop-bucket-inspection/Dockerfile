FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./bucket-inspect /workspace/argus/bucket-inspect
ADD ./start.sh /workspace/argus/start.sh

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ["/workspace/argus/start.sh"]
