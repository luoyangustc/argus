FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev as builder

RUN apt-get update && apt-get install -y --no-install-recommends wget gcc python-dev
ENV GOLANG_VERSION 1.10.3
RUN wget -O go.tgz --no-check-certificate "http://p7b5ku1yx.bkt.clouddn.com/go$GOLANG_VERSION.linux-amd64.tar.gz"; \
	tar -C /usr/local -xzf go.tgz; \
	rm -rf go.tgz; \
	\
	export PATH=$PATH:/usr/local/go/bin; \
	go version

WORKDIR /go/src
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
ADD argus /go/src/qiniu.com/argus

RUN go install qiniu.com/argus/cmd/argus-live-private/argus-live-private

FROM reg.qiniu.com/avaprd/argus-base-video-base:201808091633-v452-dev

RUN apt-get update && apt-get install -y --no-install-recommends libcurl3

RUN mkdir -p /workspace/argus/savespace && mkdir -p /workspace/argus/run/auditlog && mkdir -p /workspace/video && chmod 777 /workspace/video

COPY --from=builder /go/bin/argus-live-private /workspace/argus/argus-live-private
ADD ./argus-live-private.conf /workspace/argus/argus-live-private.conf

ENV PATH=$PATH:/workspace/argus

RUN wget http://7xt47g.com1.z0.glb.clouddn.com/ffmpeg-video-stream-0622 -O /workspace/argus/ffmpeg-ai
RUN chmod +x /workspace/argus/ffmpeg-ai
WORKDIR /workspace/argus

CMD ["argus-live-private", "-f", "argus-live-private.conf"]
