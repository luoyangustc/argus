import json

req_url_fn = 'test_urls_2pts_float.json'

fp = open(req_url_fn, 'r')
reqs = json.load(fp)
fp.close()

for it in reqs:
    pts = it["data"]["attribute"]["pts"]
    for i in range(4):
        it["data"]["attribute"]["pts"][i] = int(round(pts[i]))
fp2 = open('./test_urls_2pts_int.json', 'w')
json.dump(reqs, fp2, indent=4)
fp2.close()

for it in reqs:
    pts = it["data"]["attribute"]["pts"]
    it["data"]["attribute"]["pts"] = [
            [pts[0], pts[1]],
            [pts[2], pts[1]],
            [pts[2], pts[3]],
            [pts[0], pts[3]]
    ]

fp2 = open('./test_urls_4pts_int.json', 'w')
json.dump(reqs, fp2, indent=4)
fp2.close()