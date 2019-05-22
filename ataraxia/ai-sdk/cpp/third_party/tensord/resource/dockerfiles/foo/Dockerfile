ARG UBUNTU_VERSION=16.04
FROM ubuntu:${UBUNTU_VERSION}

RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    sed -i "s:security.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    apt-get update

#TEMPLATE tensord_base
