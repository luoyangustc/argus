FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

ADD ./dbstorage_service /workspace/argus/dbstorage_service
ADD ./dbstorage_service.conf /workspace/argus/dbstorage_service.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./dbstorage_service -f ./dbstorage_service.conf
