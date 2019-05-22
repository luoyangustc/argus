#!/bin/sh

if [ x"$BOOTS_REGION" = x"cs" ]; then
    wget http://10.200.20.25:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
elif [ x"$BOOTS_REGION" = x"z0" ]; then
    wget http://10.34.38.200:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
elif [ x"$BOOTS_REGION" = x"z1" ]; then
    wget http://10.30.21.110:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
elif [ x"$BOOTS_REGION" = x"z2" ]; then
    wget http://10.44.34.200:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
elif [ x"$BOOTS_REGION" = x"nb" ]; then
    wget http://192.168.34.200:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
elif [ x"$BOOTS_REGION" = x"na0" ]; then
    wget http://10.40.34.27:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
elif [ x"$BOOTS_REGION" = x"gz" ]; then
    wget http://10.42.34.200:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
elif [ x"$BOOTS_REGION" = x"tj" ]; then
    wget http://10.90.22.200:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
elif [ x"$BOOTS_REGION" = x"jjh" ]; then
    wget http://10.20.33.200:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
elif [ x"$BOOTS_REGION" = x"dg" ]; then
    wget http://10.46.21.200:5030/ava_conf/bucket-inspect.conf -O /workspace/argus/bucket-inspect.conf
fi

/workspace/argus/bucket-inspect -f /workspace/argus/bucket-inspect.conf