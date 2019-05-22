FROM node:4

RUN mkdir -p /workspace/rtsp
WORKDIR /workspace/rtsp

# NOTE: use the lastest comit, @2018/04/10
RUN git clone -b master https://github.com/iizukanao/node-rtsp-rtmp-server . && git checkout 77e9f609
RUN npm i -d
RUN npm install -g coffee-script

# add a customize settings.coffie to override the default conifg.coffee settings
# more detailed config, plz refer https://github.com/iizukanao/node-rtsp-rtmp-server/blob/master/config.coffee
RUN sed -i "4iObject.assign config, require './settings'" server.coffee
ADD settings.coffee .
CMD coffee server.coffee
