FROM nvidia/cuda:8.0-devel as builder
LABEL maintainer "Qiniu ATLab <ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

# apt-get && python && pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget vim lrzsz curl git unzip build-essential cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV GOLANG_VERSION 1.8.7
ENV GOLANG_DOWNLOAD_URL https://dl.google.com/go/go1.8.7.linux-amd64.tar.gz
ENV GOLANG_DOWNLOAD_SHA256 de32e8db3dc030e1448a6ca52d87a1e04ad31c6b212007616cfcc87beb0e4d60

RUN curl -fsSL "$GOLANG_DOWNLOAD_URL" -o golang.tar.gz \
        && echo "$GOLANG_DOWNLOAD_SHA256  golang.tar.gz" | sha256sum -c - \
        && tar -C /usr/local -xzf golang.tar.gz \
        && rm golang.tar.gz

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH

ENV QBOXROOT /workspace
ENV CUDA_PATH "/usr/local/cuda"
ENV CPATH "$CUDA_PATH/include/"
ENV CGO_LDFLAGS "$CUDA_PATH/lib64/libcublas.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libcurand.so"
ENV GOPATH $QBOXROOT/ava
ENV PATH $QBOXROOT/ava/bin:/usr/local/go/bin:$PATH
ADD argus /workspace/ava/src/qiniu.com/argus
RUN go install -a -tags 'cublas cuda' qiniu.com/argus/cmd/feature-search


FROM nvidia/cuda:8.0-runtime
LABEL maintainer "Qiniu ATLab <ai@qiniu.com>"
WORKDIR /workspace
RUN mkdir -p /workspace/run/auditlog
COPY feature-search.conf /workspace/feature-search.conf
COPY --from=builder  /workspace/ava/bin/feature-search /workspace/feature-search
CMD [ "/workspace/feature-search", "-f", "/workspace/feature-search.conf" ]