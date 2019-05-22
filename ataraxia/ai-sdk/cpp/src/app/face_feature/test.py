#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import json
import os
import sys
import struct
import requests
from deepdiff import DeepDiff

if __name__ == "__main__":

    with open(sys.argv[2]) as lst:
        for line in lst.readlines():
            name, pts, ret1 = line.strip().split('\t')
            pts = json.loads(pts)
            ret1 = json.loads(ret1)
            content = None
            with open(os.path.join(sys.argv[3], name)) as f:
                content = f.read()

            uri = "data:application/octet-stream;base64," + \
                base64.b64encode(content)
            r = requests.post("http://127.0.0.1:" + sys.argv[1] + "/v1/eval",
                              headers={
                                  "Content-Type": "application/json",
                                  "Authorization": "QiniuStub uid=1&ut=2"},
                              data=json.dumps({
                                  "data": {
                                      "uri": uri,
                                      "attribute": {"pts": pts},
                                  },
                              }))

            assert r.status_code == 200
            ret2 = list(struct.unpack('>512f', r.content))

            if os.environ.get("NOCHECK", '0') == '1':
                print '''{}\t{}\t{}'''.format(
                    name, json.dumps(pts), json.dumps(ret2))
                continue

            print name, \
                sum([x * y for x, y in zip(ret1, ret2)]), \
                DeepDiff(ret1, ret2, significant_digits=3)
            assert sum([x * y for x, y in zip(ret1, ret2)]) > 0.9
            # assert DeepDiff(ret1, ret2, significant_digits=3) == {}
