FROM reg.qiniu.com/avaprd/argus-base-video-base:20181206-v1101-private
RUN mkdir -p /workspace/argus
ADD ./argus-service /workspace/argus/argus-service
ADD ./argus-service.empty.conf /workspace/argus/argus-service.empty.conf
ENV PATH=$PATH:/workspace/argus
WORKDIR /workspace/argus
CMD ./argus-service -f ./argus-service.conf
