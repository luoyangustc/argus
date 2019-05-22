# base image conains the normal softwares
FROM ubuntu:16.04
MAINTAINER zhangkai <zhangkai.zju@gmail.com>

RUN mkdir proxy
COPY files/* proxy/

EXPOSE 9100

WORKDIR proxy

ENTRYPOINT ["./main","-f", "config.json", ">log 2>&1"]
