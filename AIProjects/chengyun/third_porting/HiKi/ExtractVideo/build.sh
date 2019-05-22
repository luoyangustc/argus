wget http://oq2qpeuia.bkt.clouddn.com/Hiki/PlayCtrl_Linux64_V7.3.3.30_Build20171214.zip -O PlayCtrl_Linux64_V7.3.3.30_Build20171214.zip
unzip PlayCtrl_Linux64_V7.3.3.30_Build20171214.zip
cd PlayCtrl_Linux64_V7.3.3.30_Build20171214
cd PlayCtrl_Linux64_V7.3.3.30_Build20171214
cp ./*.h /usr/local/include/
cp ./*.so /usr/local/lib/
cd ..
cd ..

wget http://oq2qpeuia.bkt.clouddn.com/Hiki/Build/ExtractVideo/libx264.tar.gz -O libx264.tar.gz
tar -xvf libx264.tar.gz
cd libx264
cp *.h /usr/local/include/
cp libx264.so /usr/local/lib/
cp libx264.so.155 /usr/local/lib/
cd ..

ldconfig
make clean
make
rm -rf PlayCtrl_Linux64_V7.3.3.30_Build20171214
rm -rf libx264
rm *.zip
rm *.gz
