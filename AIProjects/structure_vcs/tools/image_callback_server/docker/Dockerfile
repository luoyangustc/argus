FROM openresty/openresty:1.13.6.2-0-xenial

COPY nginx.conf /usr/local/openresty/nginx/conf/nginx.conf
COPY lua_scripts /lua_scripts

RUN mkdir -p /data/images
RUN sed -i 's/archive.ubuntu.com/mirrors.163.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.163.com/g' /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y \
    libmagickwand-dev \
    libgraphicsmagick1-dev \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/libvips/libvips/releases/download/v8.7.4/vips-8.7.4.tar.gz && \
    tar -zxvf vips-8.7.4.tar.gz && cd vips-8.7.4 && \
    ./configure && make && make install && ldconfig && \
    luarocks install lua-vips
