FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./argus-util /workspace/argus/argus-util
ADD ./argus-util.conf /workspace/argus/argus-util.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./argus-util -f ./argus-util.conf
