FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./file-scan-master /workspace/argus/file-scan-master
ADD ./file-scan-master.conf /workspace/argus/file-scan-master.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./file-scan-master -f ./file-scan-master.conf
