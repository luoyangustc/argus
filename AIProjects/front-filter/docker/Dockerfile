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

RUN go install qiniu.com/argus/AIProjects/mix/cmd/mix
RUN go install qiniu.com/argus/cmd/serving-eval

FROM reg.qiniu.com/inference/mix.tron:20180731-42-791c23d

RUN apt-get update -y 
RUN apt-get -y install python-setuptools libpython2.7-dev

RUN easy_install pip
RUN pip install prometheus_client

RUN mkdir -p /workspace/serving/run/auditlog 
RUN mkdir -p /tmp/eval

COPY --from=builder /go/bin/serving-eval /workspace/serving/serving-eval
COPY --from=builder /go/bin/mix /workspace/serving/mix

#Runtime set ENV OPENBLAS_NUM_THREADS=NUM_CPUS() to use multi cpu cores for caffe
ENV GODEBUG cgocheck=0
ENV PATH=$PATH:/workspace/serving
ENV INTEGRATE=lib

WORKDIR /workspace/serving

#CMD ["argus-live-private", "-f", "argus-live-private.conf"]
