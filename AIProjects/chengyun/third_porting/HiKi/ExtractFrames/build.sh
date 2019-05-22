wget http://oq2qpeuia.bkt.clouddn.com/Hiki/PlayCtrl_Linux64_V7.3.3.30_Build20171214.zip -O PlayCtrl_Linux64_V7.3.3.30_Build20171214.zip
unzip PlayCtrl_Linux64_V7.3.3.30_Build20171214.zip
cd PlayCtrl_Linux64_V7.3.3.30_Build20171214
cd PlayCtrl_Linux64_V7.3.3.30_Build20171214
cp ./*.h /usr/local/include/
cp ./*.so /usr/local/lib/
cd ..
cd ..

ldconfig
make clean
make
rm -rf PlayCtrl_Linux64_V7.3.3.30_Build20171214
rm *.zip
