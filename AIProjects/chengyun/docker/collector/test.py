import db
import datetime
import json
import os
import logging
import requests


logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")


def test_mount():
    assert len(os.listdir("/mnt/samba/")) > 3


camera_configs = [
    {
        "camera_id": "NHPZXPZ001T1",
        "camera_ip": "pdcp04/192_168_72_155"
    },
    {
        "camera_id": "A20PVRTRAN030",
        "camera_ip": "pdcp03/192_168_22_104"
    }
]


def test_error():
    url = "http://localhost:7756/v1/fetch_imgs"
    r = requests.post(url, json={})
    assert r.status_code == 400
    assert 'error' in r.json()
    url = "http://localhost:7756/v1/fetch_imgsxxxx"
    r = requests.post(url, json={})
    assert r.status_code == 404
    assert 'error' in r.json()


def test_fetch_from_db():
    os.makedirs("res/testout", 0o777, True)
    for camera_config in camera_configs:
        for days in [0.9, 1, 4, 8, 20]:
            start_time = datetime.datetime.now() - datetime.timedelta(days=days)
            params = {
                "camera_id": camera_config["camera_id"],
                "camera_ip": camera_config["camera_ip"],
                "start_time": start_time.timestamp(),
                "duration": 7200
            }
            logging.info("param %s start_time %s", params, start_time)
            r = db.fetch_imgs_raw(params)
            assert len(r["imgs"]) > 50, "{} {}".format(params, start_time)

            file_name = "res/testout/"+camera_config["camera_id"] + \
                start_time.strftime("%Y%m%d-%H:%M:%S") + ".json"
            logging.info("write file %s", file_name)
            with open(file_name, 'w') as f:
                f.write(json.dumps(r, indent=2, ensure_ascii=False))
            base_url = r["base_url"]
            cnt200 = 0
            for i, img in enumerate(r["imgs"]):
                url = "http://localhost:7756" + base_url + img["file_name"]
                logging.info("url %s", url)
                r = requests.get(url)
                assert r.status_code == 200 or r.status_code == 404
                if r.status_code == 200:
                    cnt200 += 1
                if i > 10:
                    break
            assert cnt200 > 1
