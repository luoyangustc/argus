#FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev
FROM golang:1.10
RUN mkdir -p /workspace/argus
ADD ./argus-service /workspace/argus/argus-service
ADD ./argus-service.empty.conf /workspace/argus/argus-service.empty.conf
ENV PATH=$PATH:/workspace/argus
WORKDIR /workspace/argus
CMD ./argus-service -f ./argus-service.conf
