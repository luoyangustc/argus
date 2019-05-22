#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import unittest
from easydict import EasyDict as edict


def merge_dict(a, b):
    '''deep merge EasyDict'''

    if type(a) is not edict or type(b) is not edict:
        return

    for key, value in a.iteritems():
        if not b.has_key(key):
            continue
        if type(value) != type(b.get(key)):
            raise Exception, "diff value type. key[%s] %s %s" % (key, type(value), type(b.get(key)))
        assert type(value) == type(b.get(key))
        if type(value) is edict:
            merge_dict(value, b.get(key))
        else:
            a[key] = b.get(key)

    for key, value in b.iteritems():
        if not a.has_key(key):
            a[key] = value


class TestMergeDict(unittest.TestCase):
    def test_merge_dict(self):
        a = edict({"a": 1, "b": [1, 2], "c": {"a": 1.0}})
        merge_dict(a, edict({"d": 1.0, "c": {"a": 2.0}}))
        self.assertDictEqual(a, {"a": 1, "b": [1, 2], "c": {"a": 2.0}, "d": 1.0})


if __name__ == "__main__":
    unittest.main()
