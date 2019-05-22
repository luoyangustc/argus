FROM reg.qiniu.com/avaprd/argus_video_depend:201807101034-v10

RUN mkdir -p /workspace/argus
RUN mkdir -p /workspace/argus/run/auditlog && \
apt-get update && apt-get install -y --no-install-recommends ca-certificates wget curl python-dev && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN mkdir -p /workspace/video && chmod 777 /workspace/video

ADD ./censor-video-worker /workspace/argus/censor-video-worker
ADD ./censor-video-worker.conf /workspace/argus/censor-video-worker.conf

ENV PATH=$PATH:/workspace/argus:/root/argus_video_depend/_package/bin_for_fopd

WORKDIR /workspace/argus

CMD ./censor-video-worker -f ./censor-video-worker.conf
