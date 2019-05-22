FROM nvidia/cuda:8.0-devel as builder
LABEL maintainer "Qiniu ATLab <ai@qiniu.com>"

RUN sed -i s/archive.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.163.com/g /etc/apt/sources.list

# apt-get && python && pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget vim lrzsz curl git unzip build-essential cmake clang-5.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN ln -s /usr/bin/clang-5.0 /usr/bin/clang

ENV GOLANG_VERSION 1.10.3
ENV GOLANG_DOWNLOAD_URL http://p7b5ku1yx.bkt.clouddn.com/go1.10.3.linux-amd64.tar.gz
ENV GOLANG_DOWNLOAD_SHA256 fa1b0e45d3b647c252f51f5e1204aba049cde4af177ef9f2181f43004f901035

RUN curl -fsSL "$GOLANG_DOWNLOAD_URL" -o golang.tar.gz \
        && echo "$GOLANG_DOWNLOAD_SHA256  golang.tar.gz" | sha256sum -c - \
        && tar -C /usr/local -xzf golang.tar.gz \
        && rm golang.tar.gz
ENV QBOXROOT /workspace
ENV GOPATH $QBOXROOT/ava
ENV PATH $QBOXROOT/ava/bin:/usr/local/go/bin:$PATH

WORKDIR /workspace
ADD argus $QBOXROOT/ava/src/qiniu.com/argus
ENV CC=clang
RUN go install -a -tags 'clangcgo' qiniu.com/argus/cmd/feature_group_private && cp $QBOXROOT/ava/bin/feature_group_private $QBOXROOT/ava/bin/feature_group_private_cpu

ENV CUDA_PATH=/usr/local/cuda
ENV CPATH="$CUDA_PATH/include/"
ENV CGO_LDFLAGS="$CUDA_PATH/lib64/libcublas.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/stubs/libcuda.so $CUDA_PATH/lib64/libcudart.so $CUDA_PATH/lib64/libcurand.so"
RUN go install -a -tags 'cublas cuda' qiniu.com/argus/cmd/feature_group_private && cp $QBOXROOT/ava/bin/feature_group_private $QBOXROOT/ava/bin/feature_group_private_gpu


FROM nvidia/cuda:8.0-runtime
LABEL maintainer "Qiniu ATLab <ai@qiniu.com>"

ENV QBOXROOT /workspace
WORKDIR $QBOXROOT
RUN mkdir -p $QBOXROOT/run/auditlog
COPY --from=builder  $QBOXROOT/ava/bin/feature_group_private_cpu /workspace/feature_group_private_cpu
COPY --from=builder $QBOXROOT/ava/bin/feature_group_private_gpu /workspace/feature_group_private_gpu
COPY feature_group_private.conf /workspace/feature_group_private.conf
CMD [ "/workspace/feature_group_private_cpu", "-f", "/workspace/feature_group_private.conf" ]
