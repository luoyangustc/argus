#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import json
import os
import sys
import requests
from deepdiff import DeepDiff

if __name__ == "__main__":

    with open(sys.argv[2]) as lst:
        for line in lst.readlines():
            name, ret1 = line.strip().split('\t')
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
                                  "data": {"uri": uri},
                                  "params": {"use_quality": 1},
                              }))

            assert r.status_code == 200
            ret2 = r.json()

            if os.environ.get("NOCHECK", '0') == '1':
                print '''{}\t{}'''.format(name,
                                          json.dumps([x for x in ret2['result']['detections']]))
                continue

            ret1.sort(cmp=lambda a, b:
                      (a['pts'][0][0] - b['pts'][0][0]) and (a['pts'][0][1] - b['pts'][0][1]))
            ret2['result']['detections'].sort(cmp=lambda a, b:
                                              (a['pts'][0][0] - b['pts'][0][0]) and (a['pts'][0][1] - b['pts'][0][1]))

            print name, len(ret1), len(ret2['result']['detections'])
            if len(ret1) != len(ret2['result']['detections']):
                continue
            assert DeepDiff(len(ret1), len(ret2['result']['detections'])) == {}

            for i in range(0, len(ret1)):
                pts1, pts2 = ret1[i]['pts'], ret2['result']['detections'][i]['pts']
                print pts1, pts2
                #assert abs(pts1[0][0] - pts2[0][0]) <= 5
                #assert abs(pts1[0][1] - pts2[0][1]) <= 5
                #assert abs(pts1[2][0] - pts2[2][0]) <= 5
                #assert abs(pts1[2][1] - pts2[2][1]) <= 5

                ret1[i].pop('pts', None)
                ret2['result']['detections'][i].pop('pts', None)

                if ret1[i].has_key('q_score'):
                    ret1[i]['quality'] = u'small'
                    score = 0.0
                    for k, v in ret1[i].get('q_score', {}).items():
                        if v > score:
                            score = v
                            ret1[i]['quality'] = k

                    if ret1[i]['quality'] == u"clear" and score < 0.6:
                        ret1[i]['quality'] = u"blur"

            print name, DeepDiff(
                ret1, ret2["result"]["detections"], significant_digits=2)
            # assert DeepDiff(
            #     ret1, ret2["result"]["detections"], significant_digits=2) == {}
