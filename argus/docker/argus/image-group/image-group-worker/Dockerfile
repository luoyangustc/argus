FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget vim lrzsz curl git unzip build-essential cmake clang-5.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN ln -s /usr/bin/clang-5.0 /usr/bin/clang

ENV GOLANG_VERSION 1.10.3
ENV GOLANG_DOWNLOAD_URL https://dl.google.com/go/go1.10.3.linux-amd64.tar.gz
ENV GOLANG_DOWNLOAD_SHA256 fa1b0e45d3b647c252f51f5e1204aba049cde4af177ef9f2181f43004f901035

RUN curl -fsSL "$GOLANG_DOWNLOAD_URL" -o golang.tar.gz \
        && echo "$GOLANG_DOWNLOAD_SHA256  golang.tar.gz" | sha256sum -c - \
        && tar -C /usr/local -xzf golang.tar.gz \
        && rm golang.tar.gz

WORKDIR /go
ENV GOPATH /go
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

ADD argus /go/src/qiniu.com/argus
ENV CC=clang
RUN go install -tags 'clangcgo' qiniu.com/argus/cmd/image_group/image_group_worker


FROM reg.qiniu.com/avaprd/argus-base-serving-base:20180719-v113-dev

RUN mkdir -p /workspace/argus

COPY --from=builder /go/bin/image_group_worker /workspace/argus/image_group_worker
ADD ./image_group_worker.conf /workspace/argus/image_group_worker.conf

RUN mkdir -p /workspace/argus/run/auditlog
ENV PATH=$PATH:/workspace/argus

WORKDIR /workspace/argus

CMD ./image_group_worker -f ./image_group_worker.conf
