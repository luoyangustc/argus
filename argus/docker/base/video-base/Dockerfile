FROM ubuntu:14.04 as builder
RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list
RUN apt-get update && apt-get install -y curl telnet vim build-essential gcc g++ libexpat-dev \
	libxml2-dev pkg-config make cmake tar autoconf autopoint libtool nasm gettext python-setuptools libjpeg-dev \
	libpng12-dev libtiff4-dev libgif-dev enca libexpat1-dev
RUN apt-get install -y libfreetype6-dev
ADD https://launchpad.net/ubuntu/+archive/primary/+files/libgif-dev_5.1.4-0.3_amd64.deb /tmp
ADD https://launchpad.net/ubuntu/+archive/primary/+files/libgif7_5.1.4-0.3_amd64.deb /tmp

RUN dpkg -i /tmp/libgif7_5.1.4-0.3_amd64.deb
RUN dpkg -i /tmp/libgif-dev_5.1.4-0.3_amd64.deb

ENV CMAKE_HOME /opt/cmake-3.5.2-Linux-x86_64
RUN curl -fsSL http://7xt77y.com1.z0.glb.clouddn.com/cmake-3.5.2-Linux-x86_64.tar.gz -o cmake.tar.gz \
    && tar -C /opt -xzf cmake.tar.gz \
    && rm cmake.tar.gz \
    && ln -sf /opt/cmake-3.5.2-Linux-x86_64/bin/cmake /usr/bin/cmake

RUN ln -sf /usr/bin/automake /usr/bin/automake-1.15 \
    && ln -sf /usr/bin/aclocal /usr/bin/aclocal-1.15

WORKDIR /workspace
ENV WORKSPACE=/workspace
ADD build.sh /workspace/
ADD ffmpeg /workspace/ffmpeg
RUN /workspace/build.sh
RUN find /workspace/_package

FROM ubuntu:16.04
RUN sed -i "s:archive.ubuntu.com:mirrors.aliyun.com:g" /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates wget curl vim git unzip python-dev iputils-ping net-tools netcat telnet htop lsof libcurl3 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN wget http://devtools.dl.atlab.ai/docker/PRC-tz -O /tmp/PRC-tz && mv /tmp/PRC-tz /etc/localtime
RUN wget -O /usr/local/bin/dumb-init http://devtools.dl.atlab.ai/docker/dumb-init_1.2.0_amd64 && \
    chmod +x /usr/local/bin/dumb-init
ENTRYPOINT ["/usr/local/bin/dumb-init", "--"]
COPY --from=builder /workspace/_package /root/argus_video_depend/_package
ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/argus_video_depend/_package/bin_for_fopd
ENV LD_LIBRARY_PATH /root/argus_video_depend/_package/lib_for_fopd
RUN ln -s /root/argus_video_depend/_package/bin_for_fopd/ffmpeg-3.3.2 /root/argus_video_depend/_package/bin_for_fopd/ffmpeg
