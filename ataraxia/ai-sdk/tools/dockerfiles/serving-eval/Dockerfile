FROM golang:1.10.3

RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list 
RUN sed -i "s:security.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list
RUN sed -i "s:deb.debian.org:mirrors.aliyun.com:g" /etc/apt/sources.list
RUN sed -i "s:security.debian.org:mirrors.aliyun.com:g" /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends python-dev

WORKDIR /go/src
ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

ADD argus /go/src/qiniu.com/argus

RUN go install qiniu.com/argus/cmd/serving-eval