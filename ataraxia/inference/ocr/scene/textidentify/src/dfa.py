# -*- coding: utf-8 -*-
import sys

import datetime
from config import Config, LabelType

py2 = False
if sys.version_info.major == 2:
    reload(sys)
    sys.setdefaultencoding('utf-8')
    py2 = True


class GFW(object):
    def __init__(self):
        self.d = {}

    # give a list of "ming gan ci"
    def set(self, keywords):
        p = self.d
        q = {}
        k = ''
        for word in keywords:
            word += chr(11)
            p = self.d
            for char in word:
                char = char.lower()
                if p == '':
                    q[k] = {}
                    p = q[k]
                if not (char in p):
                    p[char] = ''
                    q = p
                    k = char
                p = p[char]

    def replace(self, text, mask):
        """
        >>> gfw = GFW()
        >>> gfw.set(["sexy","girl","love","shit"])
        >>> s = gfw.replace("Shit!,Cherry is a sexy girl. She loves python.","*")
        >>> print s
        *!,Cherry is a * *. She *s python.
        """
        p = self.d
        i = 0
        j = 0
        z = 0
        result = []
        ln = len(text)
        while i + j < ln:
            # print i,j
            t = text[i + j].lower()
            # print hex(ord(t))
            if not (t in p):
                j = 0
                i += 1
                p = self.d
                continue
            p = p[t]
            j += 1
            if chr(11) in p:
                p = self.d
                result.append(text[z:i])
                result.append(mask)
                i = i + j
                z = i
                j = 0
        result.append(text[z:i + j])
        return "".join(result)

    def check(self, text):
        """
        >>> gfw = GFW()
        >>> gfw.set(["abd","defz","bcz"])
        >>> print gfw.check("xabdabczabdxaadefz")
        ['abd', 'bcz', 'abd', 'defz']
        """
        p = self.d
        i = 0
        j = 0
        result = []
        ln = len(text)
        while i + j < ln:
            t = text[i + j].lower()
            # print i,j,hex(ord(t))
            if not (t in p):
                j = 0
                i += 1
                p = self.d
                continue
            p = p[t]
            j += 1
            # print p,i,j
            if chr(11) in p:
                p = self.d
                result.append(text[i:i + j])
                i = i + j
                j = 0
        return result


class SensitiveDictLoader(object):
    """
    Define a default Dictionary Loader for Sensitive Dictionary
    """

    def __init__(self, key_type):

        print("dfa type:", key_type)

        self.key_type = key_type
        self.config = Config()
        self.sensitive_map = dict()
        self.sensitive_type = dict()
        self.sensitive_score = dict()
        self._is_ok = False
        self.type_map = {
            u"低俗": LabelType.Pulp,
            u"涉黄": LabelType.Ads,
            u"涉敏感": LabelType.Politician,
            u"其他违规": LabelType.Other,
            u"其他违禁": LabelType.Other,
            u"色情": LabelType.Pulp,
            u"涉暴恐": LabelType.Terror,
            u"广告": LabelType.Ads,
            u"违规": LabelType.Other,
            u"敏感": LabelType.Politician,
            u"暴恐": LabelType.Terror,
            u"涉政": LabelType.Politician
        }
        self.gfw = GFW()
        self.filter_keys = []
        self.set()

    def set(self):
        start_time = datetime.datetime.now()
        data_count = 0
        with open(self.config.SENSITIVE_KEYS_PATH, "r") as fin:
            for line in fin:
                if py2:
                    line = line.encode('utf-8').decode('utf-8')
                line = line.strip().lower()
                items = line.split(",")
                # print(line)
                # print(line)
                assert(len(items) == 3,line)
                # print(items)
                # print("checks,",self.key_type,self.type_map[items[1]])
                if self.key_type is not self.type_map[items[1]]:
                    continue
                data_count += 1
                self.sensitive_score[items[0]] = float(items[2])
                self.filter_keys.append(items[0])
        self.gfw.set(self.filter_keys)
        end_time = datetime.datetime.now()
        print("load {} keys with time:{}".format(
            data_count, end_time - start_time))

    def query(self, text):
        if not text:
            return [], LabelType.Normal, 1.0
        keys = self.gfw.check(text)

        if keys:
            score = 0.0
            for key in keys:
                
                if score < self.sensitive_score[key]:
                    score = self.sensitive_score[key]
            score += float(len(keys) / 100.0)
            if score > 1:
                score = 1.0
            return keys, self.key_type, score

        return [], LabelType.Normal, 1.0


if __name__ == "__main__":

    import doctest
    import sys
    doctest.testmod(sys.modules[__name__])

    dfa = []
    for t in LabelType:
        if t == LabelType.Normal:
            continue

        dfa.append(SensitiveDictLoader(key_type=t))

    # no sensitive words
    ads = SensitiveDictLoader(key_type=LabelType.Ads)
    #print(ads.query(u"唯品会"))
    for d in dfa:
        print(d.query(u"微信"))
