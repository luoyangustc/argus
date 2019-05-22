#!/bin/python

import base64
import json
import os
import requests
import sys


def post(num, port, lines, return_dict):

    sums1 = {}
    sums2 = {}
    i = 0
    for line in lines[:]:
        ret1 = json.loads(line.strip())
        label1 = ret1['label'][0]['data'][0]['class']
        # rets.append(pool.apply_async(post, args=(sys.argv[1], label1, ret1['url'],)))

        r = requests.post("http://127.0.0.1:" + port + "/v1/eval",
                          headers={
                              "Content-Type": "application/json",
                              "Authorization": "QiniuStub uid=1&ut=2"},
                          data=json.dumps({
                              "data": {
                                  "uri": ret1['url'],
                              }}))
        ret2 = r.json()
        label1 = ret1['label'][0]['data'][0]['class']
        label2 = ret2.get('result', {}).get('classify', {}).get('confidences', [{}])[
            0].get('class', "").split("_")[0]
        print i, label1, label2
        sums1[label1] = list(map(
            lambda x: x[0]+x[1], zip(sums1.get(label1, [0, 0]), [1, 1 if label2 == label1 else 0])))
        sums2[label2] = list(map(
            lambda x: x[0]+x[1], zip(sums2.get(label2, [0, 0]), [1, 1 if label2 == label1 else 0])))
        i += 1

    return_dict[num] = {"r": sums1, "p": sums2}


if __name__ == "__main__":

    from multiprocessing import Process, Manager

    # 0	bloodiness
    # 1	bomb
    # 2	beheaded
    # 3	march
    # 4	fight
    # 5	normal
    labels = {
        "bloodiness": 0,
        "bomb": 1,
        "beheaded": 2,
        "march": 3,
        "fight": 4,
        "normal": 5,
    }

    lines = []
    with open(sys.argv[2]) as lst:
        lines = lst.readlines()

    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    # lines = lines[:100]
    n = 10
    m = len(lines) / n
    for i in range(n):
        lines0 = lines[i*m: len(lines) if i == n - 1 else (i + 1) * m]
        p = Process(target=post, args=(i, sys.argv[1], lines0, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    sums1 = {}
    sums2 = {}
    for ret in return_dict.values():
        for k in ret['r'].keys():
            sums1[k] = list(
                map(lambda x: x[0]+x[1], zip(sums1.get(k, [0, 0]), ret['r'][k])))
        for k in ret['p'].keys():
            sums2[k] = list(
                map(lambda x: x[0]+x[1], zip(sums2.get(k, [0, 0]), ret['p'][k])))

    print 'recall:'
    for item in list(map(lambda x: (x[0], x[1][0], x[1][1], x[1][1] * 1.0 / x[1][0], (x[1][0] - x[1][1]) * 1.0 / x[1][0]),  sums1.items())):
        print "\t", item[0], "\t", item[1], "\t", item[2], "\t", item[3]
    print 'precision:'
    for item in list(map(lambda x: (x[0], x[1][0], x[1][1], x[1][1] * 1.0 / x[1][0], (x[1][0] - x[1][1]) * 1.0 / x[1][0]),  sums2.items())):
        print "\t", item[0], "\t", item[1], "\t", item[2], "\t", item[3]
    print 'acc:'
    right, sum = (0, 0)
    for _, item in sums1.items():
        right += item[1]
        sum += item[0]
    print "\t", right * 1.0 / sum
