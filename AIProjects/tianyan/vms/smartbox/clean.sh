rm -rf ../src/ffmpeg_static
mkdir ../src/ffmpeg_static
cd ffmpeg-4.0.2
make distclean
cd ..
cd ./src/base_lib/build
make -f Makefile.linux clean

cd ../../qiniu_dev_net_lib/build
make -f Makefile.linux clean

cd ../../device_porting/build
make clean



