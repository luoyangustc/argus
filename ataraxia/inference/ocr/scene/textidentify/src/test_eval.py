# coding:utf8
from eval import *
import unittest
import json
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')
import argparse
from config import Config
from timer import Timer
import json
import sys
import traceback
import time
type_lst = ["ads", "pulp", "terror", "politician", "other"]
import random

# test.txt url:http://p3h77c140.bkt.clouddn.com/test.txt


class TestCase(unittest.TestCase):
    def setUp(self):
        configs = {}
        configs["batch_size"] = 1000
        self.model, _, _ = create_net(configs)

    def testInfer(self):
        reqs = []
        texts = ["圣塔菲联", "河床VS", "独赢&进球大/小", "投注项:", "2.5", "圣塔菲联&大2.508.4", "投注单号:", "58998501061034",
                 "时间:2019/01/2408:57", "[足球]阿根廷超级联赛", "投注额:", "290.10RMB", "可赢额:", "2,146.74RMB", "注单状态:", "投注成功"]
        body["body"] = json.dumps(texts, ensure_ascii=False)
        req["data"] = body
        req["params"] = {"type": ["ads"]}
        reqs.append(req)

        with open('test.txt', 'r') as f:
            for line in f:
                req = {}
                body = {}
                texts = line.strip().split("，")

                body["body"] = json.dumps(texts, ensure_ascii=False)
                req["data"] = body
                count = random.randint(0, 5)
                if count != 0:
                    rand_lst = random.sample(type_lst, count)
                    req["params"] = {"type": ["ads"]}
                reqs.append(req)

        rets, _, _ = net_inference(self.model, reqs)
        with open("test_text_identify.tsv", "w") as w:
            for req, ret in zip(reqs, rets):
                text = {}
                text["text"] = json.loads(req["data"]["body"])
                req_json = json.dumps(text, ensure_ascii=False)
                resp_json = json.dumps(ret["result"], ensure_ascii=False)
                w.write(req_json + "\t" + resp_json + "\n")


if __name__ == "__main__":
    unittest.main()
