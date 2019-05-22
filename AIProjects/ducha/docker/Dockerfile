FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev as builder

RUN apt-get update && apt-get install -y --no-install-recommends wget gcc python-dev

ENV GOLANG_VERSION 1.10.3
RUN wget -O go.tgz --no-check-certificate "https://dl.google.com/go/go$GOLANG_VERSION.linux-amd64.tar.gz"; \
	tar -C /usr/local -xzf go.tgz; \
	rm -rf go.tgz; \
	\
	export PATH=$PATH:/usr/local/go/bin; \
	go version

WORKDIR /go
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
ADD src /go/src
RUN ls /go/src

RUN go install qiniu.com/argus/AIProjects/ducha/cmd/ducha-live
RUN go install qiniu.com/argus/cmd/serving-eval

FROM reg.qiniu.com/avaprd/argus-base-video-base:201808091633-v452-dev

RUN mkdir -p /workspace/argus/savespace && mkdir -p /workspace/argus/run/auditlog && mkdir -p /workspace/video && chmod 777 /workspace/video

COPY --from=builder /go/bin/ducha-live /workspace/argus/ducha-live
ADD ./ducha-live.conf /workspace/argus/ducha-live.conf

ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ["ducha-live", "-f", "ducha-live.conf"]