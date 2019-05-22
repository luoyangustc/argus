FOP_DEPEND=${WORKSPACE}/_package
FOP_BIN=${FOP_DEPEND}/bin_for_fopd
FOP_LIB=${FOP_DEPEND}/lib_for_fopd

rm -rf ${FOP_DEPEND}

mkdir -p ${FOP_BIN}
mkdir -p ${FOP_LIB}
cd ${WORKSPACE}/ffmpeg/build-static

./build.sh
./build_ffmpeg_ai.sh

cp ${WORKSPACE}/ffmpeg/build-static/target/bin/ffmpeg-3.3.2 $FOP_BIN/ffmpeg-3.3.2
cp ${WORKSPACE}/ffmpeg/build-static/target/bin/gen_pic $FOP_BIN/gen_pic
cp ${WORKSPACE}/ffmpeg/build-static/target/bin/ffmpeg_ai $FOP_BIN/ffmpeg_ai

cp /usr/lib/x86_64-linux-gnu/libgif.so.7 ${FOP_LIB}
cp /usr/bin/enca ${FOP_BIN}
cp /usr/bin/enca ${FOP_BIN}/enconv

cd ${WORKSPACE}

curl -o ${FOP_LIB}/librecode.so.0 http://ogtoywd4d.bkt.clouddn.com/librecode.so.0
curl -o ${FOP_LIB}/libenca.so.0 http://ogtoywd4d.bkt.clouddn.com/libenca.so.0
curl -o ${FOP_LIB}/libfreetype.so.6 http://ogtoywd4d.bkt.clouddn.com/libfreetype.so.6

find ${FOP_DEPEND}
