from eval import *
import unittest
from PIL import Image, ImageDraw, ImageFont
model_path = "/workspace/serving/python/evals/crnn_0_8.pth"
import json
import codecs
import profile


class TestCase(unittest.TestCase):
    def setUp(self):
        configs = {}
        model_conf = {}
        model_conf['crnn_0_8.pth'] = model_path
        configs["use_device"] = "GPU"
        configs["batch_size"] = 10000
        configs['model_files'] = model_conf
        self.model, _, _ = create_net(configs)

    def testInfer(self):
        root_path = "images"
        reqs = []
        with open('test.tsv', 'r') as f:
            for line in f:
                req = {}
                body = {}
                words = line.strip().split("\t")
                img_name = words[0]
                box_json = words[-1]
                box_json = box_json.replace('\'', '\"')
                box_map = json.loads(box_json)
                new_box = {}
                pts_arr = []
                for pt in box_map["bboxes"]:
                    pts_arr.append(
                        {"pts": [[pt[0], pt[1]], [pt[2], pt[3]], [pt[4], pt[5]], [pt[6], pt[7]]]})
                new_box["detections"] = pts_arr
                body["uri"] = root_path + "/" + img_name
                body["body"] = None
                body["attribute"] = new_box
                req["data"] = body
                reqs.append(req)

        rets, _, _ = net_inference(self.model, reqs)
        w = open("test_recog_v3.tsv", "w")
        for req, ret in zip(reqs, rets):

            img = Image.open(req["data"]["uri"])
            draw = ImageDraw.Draw(img)
            req_str = json.dumps(req["data"]["attribute"], ensure_ascii=False)
            resp_str = json.dumps(ret["result"], ensure_ascii=False)
            w.write(req["data"]["uri"].split("/")[-1] +
                    "\t" + req_str + "\t" + resp_str + "\n")

            font = ImageFont.truetype("yahei.ttf", 19, encoding="utf-8")
            for text in ret["result"]["texts"]:
                if text["text"] == "":
                    continue
                pts = text["pts"]
                x0, y0 = text["pts"][0][0], text["pts"][0][1]
                if y0 - 20 < 0:
                    y0 = 0
                else:
                    y0 = y0 - 20
                print(x0, y0)
                # if True:
                try:
                    pt1 = pts[0]
                    pt2 = pts[1]
                    pt3 = pts[2]
                    pt4 = pts[3]
                    draw.text((x0, y0), text["text"],
                              (127, 152, 98), font=font)
                    draw.line(((pt1[0], pt1[1]),
                               (pt2[0], pt2[1])), fill=255)
                    draw.line(((pt1[0], pt1[1]),
                               (pt4[0], pt4[1])), fill=255)
                    draw.line(((pt2[0], pt2[1]),
                               (pt3[0], pt3[1])), fill=255)
                    draw.line(((pt4[0], pt4[1]),
                               (pt3[0], pt3[1])), fill=255)
                    pt1, pt2, pt3, pt4 = self.model["text_recognizer"].regular_pts(
                        pt1, pt2, pt3, pt4, img.size[0])

                    draw.line(((pt1[0], pt1[1]),
                               (pt2[0], pt2[1])), fill=(0, 0, 255))
                    draw.line(((pt2[0], pt2[1]),
                               (pt3[0], pt3[1])), fill=(0, 0, 255))
                    draw.line(((pt4[0], pt4[1]),
                               (pt3[0], pt3[1])), fill=(0, 0, 255))
                    draw.line(((pt1[0], pt1[1]),
                               (pt4[0], pt4[1])), fill=(0, 0, 255))
                    draw.text(
                        (max(pt1[0] - 10, 0), max(pt1[1] - 10, 0)), "1", (255, 0, 0), font=font)
                    draw.text((pt2[0], max(pt2[1] - 10, 0)),
                              "2", (255, 0, 0), font=font)
                    draw.text(
                        (pt3[0], min(pt3[1] + 10, img.size[1] - 1)), "3", (255, 0, 0), font=font)
                    draw.text(
                        (pt4[0], min(pt4[1] + 10, img.size[1] - 1)), "4", (255, 0, 0), font=font)
                except Exception as e:
                    print(e)
            img.save("image_output/" + req["data"]
                     ["uri"].split("/")[-1] + ".png")
        w.close()


if __name__ == "__main__":
    unittest.main()
