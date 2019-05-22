# -*- coding: utf-8 -*-
# linyining@mail.qiniu.com


import requests
import sys
from qiniu import Auth, put_file, etag, urlsafe_base64_encode, BucketManager
import logging

logging.basicConfig(filename='eval.log', level=logging.DEBUG)


class QiniuStorage:
    def __init__(self, ak, sk):
        self.q_access = Auth(ak, sk)
        self.bucket_manager = BucketManager(self.q_access)

    def create_bucket(self):
        print "not implemented"

    def list_bucket_single(self, bucket_name, prefix=None, marker=None, limit=1000, delimiter=None):
        ret, eof, info = self.bucket_manager.list(bucket_name, prefix, marker, limit, delimiter)
        if info.status_code == 200:
            if len(ret.get('items')) is None:
                return "fail", "no file matched"
            else:
                return "success", ret
        else:
            return "fail", "return_code is not 200"

    def list_bucket_loop(self, bucket_name, prefix=None, delimiter=None):
        ret, cont = self.list_bucket_single(bucket_name, prefix)
        content_all = []
        if not ret == "fail":
            content_all = cont.get('items')
        else:
            return ret, cont
        while not cont.get('marker') is None:
            marker = cont.get('marker')
            ret, cont = self.list_bucket_single(bucket_name, prefix, marker)
            if ret == "fail":
                return "fail", "unfinish list loop"
            else:
                content_all.extend(cont.get('items'))
        return "success", content_all

    def list_bucket(self, bucket_name, prefix=None, delimiter=None):
        return self.list_bucket_loop(bucket_name, prefix, delimiter)

    def download(self, bucket_domain, key, donwloaded_filename, expires=3600):
        prox = {"http": "http://iovip.qbox.me:80"}
        try:
            base_url = 'http://%s/%s' % (bucket_domain, key)
            # print prox, base_url, donwloaded_filename
            # private_url = self.q_access.private_download_url(private_url, expires)

            r = requests.get(base_url, stream=True, proxies=prox)

            # r = requests.get(base_url)
            # print base_url, "RRR status: ", r.status_code, "FILE - ", downloaded_filename

            if r.status_code == 200:
                # print "RRRRR.status_code == 200 --- downloaded", downloaded_filename

                self._write_image(donwloaded_filename, r)

                return "success"
            else:
                return "fail"
        except Exception as e:
            logging.ERROR(e)
            # print e
            return "fail"

    def _write_file(self, filename, resp):
        with open(donwloaded_filename, "wb") as fin:
            fin.write(r.content)

    def _write_image(self, filename, resp):
        print "WT IM:", filename
        with open(filename, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
            f.close()

    def upload(self, bucket_name, key, local_file, expires=3600):
        try:
            policy = {"scope": bucket_name + ":" + key}
            token = self.q_access.upload_token(bucket_name, key, expires, policy)
            ret, info = put_file(token, key, local_file)
            if ret['key'] == key and ret['hash'] == etag(local_file):
                return "success"
            else:
                return "fail"
        except Exception as e:
            logging.debug(str(e))
            return "fail"

    def fetch_url_to_bucket(self, url, bucket_name, key):
        bucket = BucketManager(self.q_access)
        try:
            ret, info = bucket.fetch(url, bucket_name, key)
            if ret['key'] == key:
                return "success"
            else:
                return "fail"
        except Exception as e:
            print e
            return "fail"


if __name__ == '__main__':
    bucket_name = sys.argv[1]
    prefix = sys.argv[2]
    AK = sys.argv[3]
    SK = sys.argv[4]
    qapi = QiniuStorage(AK, SK)
    url = "http://a.hiphotos.baidu.com/image/pic/item/e7cd7b899e510fb3a78c787fdd33c895d0430c44.jpg"
    ret, cont = qapi.list_bucket(bucket_name, prefix)
    print "GET FILES"
    # print cont
    fp = open("list.out", "w")
    for c in cont:
        print c
        fp.write(str(c) + "\n")
    fp.close()

    # '''
    bucket_domain = "o9d987omi.qnssl.com"
    key = "beau.jpg"
    new_key = "beauty.jpg"
    ret = qapi.fetch_url_to_bucket(url, bucket_name, key)
    print "fetch ", ret
    ret = qapi.download(bucket_domain, key, new_key)
    print "dowonload ", ret
    print "upload: ", bucket_name, new_key
    ret = qapi.upload(bucket_name, new_key, new_key)
    print "upload ", ret
