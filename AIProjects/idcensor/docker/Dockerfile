FROM ava-serving-base:20170524-v70-ci5

RUN mkdir -p /workspace/argus

ADD ./idcensor /workspace/argus/idcensor
ADD ./idcensor.conf /workspace/argus/idcensor.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./idcensor -f ./idcensor.conf
