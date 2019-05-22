#!/bin/bash
# requires https://github.com/Yelp/dumb-init, mirrored at
# http://devtools.dl.atlab.ai/docker/dumb-init_1.2.0_amd64
# Works with ubuntu/debian/..., does not work with alpine.

set -e

cmd="$@"

if [ $# = 0 ]; then
      cmd="sleep infinity"
fi

# put some daemon here, e.g. /usr/sbin/sshd

# default CPU, use env: USE_DEVICE=GPU to use GPU version
if [ "${USE_DEVICE}" == "GPU" ]; then
       echo "Use caffe built in GPU mode!"
       rm -rf /opt/caffe && ln -s /opt/caffe_gpu /opt/caffe
else
       rm -rf /opt/caffe && ln -s /opt/caffe_cpu /opt/caffe
fi

exec /usr/local/bin/dumb-init -- $cmd

