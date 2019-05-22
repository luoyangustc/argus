FROM golang:1.11.0-stretch
RUN sed -i "s:deb.debian.org:mirrors.aliyun.com:g" /etc/apt/sources.list && apt-get update
RUN apt-get install -y --no-install-recommends libzmq5-dev libsodium-dev libpgm-dev
WORKDIR /src

