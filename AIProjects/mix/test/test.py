import base64
import json
import fileinput
import multiprocessing
try:
    import requests
except ImportError as e:
    print('sudo python2 -m pip install -i https://mirrors.aliyun.com/pypi/simple  requests')
    raise e

image_txt = "15000_img.txt"
eval_url = "http://127.0.0.1:23400/v1/pic"
worker = 2

def b64_eval(img):
    b64 = base64.encodestring(open(img,"rb").read()).decode('ascii')
    req = {
        "uri": "data:application/octet-stream;base64," + b64,
        "meta":{
            "pic_name": img
        },
    }
    resp = requests.post(eval_url, data=json.dumps(req),headers={"Content-Type": "application/json"})
    print(resp.json())

def thread_func(images):
    for img in images:
        b64_eval(img)

def main():
    images = []
    for i in range(worker):
        images.append([])
    index = 0
    for line in fileinput.input(image_txt):
        images[i%worker].append(line[:-1])
        i = i+1
    for i in range(worker):
        thread = multiprocessing.Process(target=thread_func,args=(images[i],))
        thread.start()

if __name__ == '__main__':
    main()