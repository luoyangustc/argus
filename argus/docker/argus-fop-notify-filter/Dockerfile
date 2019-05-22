FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./notify-filter /workspace/argus/notify-filter
ADD ./argus-fop.conf /workspace/argus/argus-fop.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./notify-filter -f ./argus-fop.conf
