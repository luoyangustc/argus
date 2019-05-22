#TEMPLATE mxnet as base

FROM base

ADD tensord /opt/tensord
RUN mkdir /opt/tensord/build && \
    cd /opt/tensord/build && \
    cmake -DPLATFORM_MXNET=ON .. && \
    make -j"$(nproc)" && \
    mkdir /opt/bin && mv bin/tensord /opt/bin/tensord && \
    rm -rf /opt/tensord
