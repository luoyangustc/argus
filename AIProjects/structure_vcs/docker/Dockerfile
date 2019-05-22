FROM reg-xs.qiniu.io/atlab/vss:v0.1.3

COPY . /src/structure_vcs
WORKDIR /src/structure_vcs

RUN mkdir -p data/models data/videos && \
    cat config/models.list | xargs -i wget -q --directory-prefix=/data/models {} && \
    wget -q --directory-prefix=/data/videos http://oquqvdmso.bkt.clouddn.com/videos/211nanquanxixia1700.mp4  && \
    cp -r config /
RUN find . -type f | grep -e 'pb.h' -e 'pb.cc' | xargs -i rm {} && \
    USE_MOCK_VSS=false ./scripts/build.sh && \
    ldconfig && \
    rm -rf /src/structure_vcs /opt/Alg-VideoAlgorithm/

WORKDIR /