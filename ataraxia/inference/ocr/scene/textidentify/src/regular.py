#-*-coding:utf-8-*-
import re
import sys
from config import LabelType
if sys.version_info.major == 2:
    reload(sys)
    sys.setdefaultencoding('utf-8')
elif sys.version_info.major == 3:
    pass


class Regular:

    def __init__(self, re_path="data/rules.txt"):

        self.res_lst = []
        self.score_lst = []
        with open(re_path, "r") as f:
            for line in f:
                items = line.encode(
                    "utf-8").decode("utf-8").strip().split("->")
                self.res_lst.append(re.compile(items[0]))
                self.score_lst.append(float(items[1]))

    def check(self, msg):
        keys = []
        score = 0.0

        for res, sc in zip(self.res_lst, self.score_lst):
            data = res.findall(msg)
            if data:
                score += sc
                for d in data:
                    if isinstance(d, tuple):
                        keys.append("".join(list(d)))
                    else:
                        keys.append(d)

            keys = list(set(keys))

        if score > 1:
            score = 1.0
        if keys and score > 0.0:
            return keys, LabelType.Ads, score
        else:
            return [], LabelType.Normal, 1.0


if __name__ == "__main__":
    hi = Regular()
    # msg = "qq:234332445   电话0536-6666888  123455234,18888886688,3445858@gmail.com,13524123160购买，联系人:,电话,Q币邮箱:23434@qq.com QQ:2323435 www.taobao.com 微信：abdcd02dfg3434545 地址: 联系人:刘先生 包治百病 仅仅20元 买二送一满20减30元"
    # print(hi.check(msg))
    msg = "73784366,加我微信"
    print(hi.check(msg))
    # msg = "我是好人"
    # print(hi.check(msg))
