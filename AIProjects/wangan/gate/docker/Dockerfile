FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev
RUN mkdir -p /workspace/argus/run/auditlog
ADD ./wangan-gate /workspace/argus/wangan-gate
ADD ./wangan-gate.conf /workspace/argus/wangan-gate.conf
ENV PATH=$PATH:/workspace/argus
WORKDIR /workspace/argus
CMD ./wangan-gate -f ./wangan-gate.conf