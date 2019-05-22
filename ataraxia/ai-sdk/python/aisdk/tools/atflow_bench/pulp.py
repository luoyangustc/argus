#coding=utf8
from __future__ import unicode_literals
import requests
import json
from deepdiff import DeepDiff
import schema

# 使用atflow导出的json来验证APP
# API: https://cf.qiniu.io/pages/viewpage.action?pageId=83301165
# 导出脚本： @dikeke

# {
#   "url": "qiniu:///ava-test/atflow-log-proxy/images/pulp-2018-09-27T11-51-49-UiDEX3-JbTPVnqnrrFNw_Q==",
#   "type": "image",
#   "ops": "logexporter",
#   "label": [
#     {
#       "name": "pulp",
#       "type": "classification",
#       "version": "1",
#       "data": [
#         { "class": "normal", "score": 0.9999032020568848, "index": 0 },
#         { "class": "sexy", "score": 0.00008413815521635115, "index": 0 },
#         { "class": "pulp", "score": 0.000012663709640037268, "index": 0 }
#       ]
#     }
#   ],
#   "uid": 1380919687,
#   "create_time": "2018-09-27T11:51:49.591889697+08:00",
#   "original_url": "sts://10.24.34.24:5557/v1/fetch?uri=qiniu%3A%2F%2F1380919687%40z0%2Fchouti%2FCHOUTI_BB1F50C8BB024DE2B1BED6F18B893B6B_W550H360.jpg"
# }

line_schema = schema.Schema({
    'url':
    basestring,
    'label': [{
        'data':
        schema.Or([{
            'class': basestring,
            'score': float,
            'index': int,
        }], None),
        basestring:
        object
    }],
    basestring:
    object
})


def pytest_generate_tests(metafunc):
    # TODO: use params
    lines = [
        json.loads(i) for i in open(
            'res/atflow/pulp_20180928101634_85ac78e8-c2c4-11e8-a40f-8c859011d6f5.json'
        ).read().split('\n') if len(i) > 1
    ]
    metafunc.parametrize('line', lines)


def test_compute(line):
    line_schema.validate(line)
    print('process {}'.format(line['url']))
    is_bad_request = False
    if not line['label'][0]['data']:
        is_bad_request = True
    # like qiniu:///ava-test/atflow-log-proxy/images/pulp-2018-09-27T11-51-49-UiDEX3-JbTPVnqnrrFNw_Q==
    url = line['url'].replace('qiniu:///ava-test',
                              'http://ava-test.hi-hi.cn')  # type: str
    api_response = requests.post(
        'http://127.0.0.1:9100/v1/eval',
        json={
            'data': {
                'uri': url
            },
            'params': {
                'limit': 3
            }
        }).json()
    if is_bad_request:
        assert api_response.get('error') != ''
        return
    api_result = api_response['result']['confidences']

    for i in api_result:
        i['index'] = 0
    # atflow bug: pulp index都是0
    atflow_result = line['label'][0]['data']

    api_result.sort(key=lambda x: x['score'])
    api_result_class = api_result[-1]['class']

    atflow_result.sort(key=lambda x: x['score'])
    atflow_result_class = atflow_result[-1]['class']
    # print(api_result, atflow_result, api_result_class, atflow_result_class)
    assert DeepDiff(
        atflow_result_class, api_result_class, significant_digits=2) == {}
    assert DeepDiff(atflow_result, api_result, significant_digits=2) == {}
