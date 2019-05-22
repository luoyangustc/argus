rm -rf ../src/ffmpeg_static
mkdir ../src/ffmpeg_static
cd ffmpeg-4.0.2
./configure --prefix="../src/ffmpeg_static" --disable-shared --enable-static --enable-gpl --disable-lzma && make -j 4 && make install
cd ..
cd ./src/base_lib/build
make -f Makefile.linux

cd ../../qiniu_dev_net_lib/build
make -f Makefile.linux

cd ../../device_porting/build
make



