FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./face_group_upgrade /workspace/argus/face_group_upgrade
ADD ./face_group_upgrade.conf /workspace/argus/face_group_upgrade.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./face_group_upgrade -f ./face_group_upgrade.conf