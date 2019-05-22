FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./bucket-censor-master /workspace/argus/bucket-censor-master
ADD ./bucket-censor-master.conf /workspace/argus/bucket-censor-master.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./bucket-censor-master -f ./bucket-censor-master.conf
