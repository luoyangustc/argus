# coding: utf-8
from __future__ import absolute_import, division, generators, nested_scopes, print_function, unicode_literals, with_statement

import pytest
from aisdk.common.download_model import read_env_file, trim_suffix, read_tsv, parse_tsv, read_test_image, read_test_image_with_cache


def test_read_env_file():
    cfg = read_env_file()
    assert cfg['AVA_PUBLIC_USERNAME'] == 'ava-public@qiniu.com'


def test_trim_suffix():
    assert trim_suffix('abc_def', 'def') == 'abc_'
    with pytest.raises(AssertionError):
        trim_suffix('abc_def', 'defxx')


def test_read_tsv():
    s = read_tsv('serving/pulp/20180606/set1/20180606-2.tsv')
    cases = parse_tsv(s)
    assert cases[0][0] == 'Image-tupu-2016-09-01-15-30-3025.jpg'
    assert cases[0][
        1] == '[{"index": 0, "score": 0.999981, "class": "pulp"}, {"index": 1, "score": 1.51818e-05, "class": "sexy"}, {"index": 2, "score": 4.2335e-06, "class": "normal"}]'
    assert len(cases[-1]) == 2


def test_read_test_image():
    s = read_test_image(
        'serving/pulp/set1/Image-tupu-2016-09-01-00-00-327.jpg')
    assert len(s) == 33847


def test_read_test_image_with_cache():
    s = read_test_image_with_cache(
        'serving/pulp/set1/Image-tupu-2016-09-01-00-00-327.jpg')
    assert len(s) == 33847
