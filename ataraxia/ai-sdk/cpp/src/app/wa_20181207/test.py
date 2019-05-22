#!/bin/python

import base64
import json
import os
import requests
import sys

if __name__ == "__main__":

    with open(sys.argv[2]) as lst:
        for line in lst.readlines():
            name, ret1 = line.strip().split('\t')
            ret1 = json.loads(ret1)
            content = None
            with open(os.path.join(sys.argv[3], name)) as f:
                content = f.read()

            r = requests.post("http://127.0.0.1:" + sys.argv[1] + "/v1/eval",
                              headers={
                                  "Content-Type": "application/json",
                                  "Authorization": "QiniuStub uid=1&ut=2"},
                              data=json.dumps({
                                  "data": {
                                      "uri": "data:application/octet-stream;base64," + base64.b64encode(content),
                                  }}))

            ret2 = r.json()
            if len(ret1['detection']) != len(ret2['result']['detection']) or \
                    len(ret1['classify']['confidences']) != len(ret2['result']['classify']['confidences']):
                print name, ret1, ret2
            for i in range(len(ret1['detection'])):
                d1, d2 = ret1['detection'][i], ret2['result']['detection'][i]
                if d1['index'] != d2['index'] or \
                        d1['class'] != d2['class'] or \
                        abs(d1['pts'][0][0] - d2['pts'][0][0]) > 2 or \
                        abs(d1['pts'][0][1] - d2['pts'][0][1]) > 2 or \
                        abs(d1['pts'][2][0] - d2['pts'][2][0]) > 2 or \
                        abs(d1['pts'][2][1] - d2['pts'][2][1]) > 2 or \
                        abs(d1['score'] - d2['score']) > 0.01:
                    print name, ret1, ret2
            c1, c2 = ret1['classify']['confidences'][0], ret2['result']['classify']['confidences'][0]
            if c1['index'] != c2['index'] or \
                    c1['class'] != c2['class'] or \
                    abs(c1['score'] - c2['score']) > 0.01:
                print name, c1, c2
