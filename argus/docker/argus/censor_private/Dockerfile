FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./censor_private /workspace/argus/censor_private
ADD ./censor_private.conf /workspace/argus/censor_private.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./censor_private -f ./censor_private.conf
