wget http://oq2qpeuia.bkt.clouddn.com/Hiki/CH_HCNetSDK_V5.2.7.4_build20170606_Linux64.zip -O CH_HCNetSDK_V5.2.7.4_build20170606_Linux64.zip
unzip CH_HCNetSDK_V5.2.7.4_build20170606_Linux64.zip
cd CH_HCNetSDK_V5.2.7.4_build20170606_Linux64
cp ./lib/*.so /usr/local/lib
cp ./lib/HCNetSDKCom/*.so /usr/local/lib
cp ./incCn/*.h /usr/local/include
cd ..

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
rm -rf CH_HCNetSDK_V5.2.7.4_build20170606_Linux64
rm *.zip
