FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./search-master /workspace/argus/search-master
ADD ./tuso-search-master.conf /workspace/argus/tuso-search-master.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./search-master -f ./tuso-search-master.conf
