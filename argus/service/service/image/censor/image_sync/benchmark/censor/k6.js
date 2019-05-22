import http from "k6/http";
import { check } from "k6";
import encoding from "k6/encoding";

var url = "http://localhost:10000/v3/censor/image"; // 图片内容审核 API 接口，如果服务不在本机，需要将 localhost 换为服务器 IP

// 读取所有 benchmark 图片内容，存储至 file_list
let data_list_file = `${__ENV.BENCHMARK_DATA_LIST_FILE}`; //传入变量 传入list地址
let data_list = shuffle(open(data_list_file).split("\n"));
let file_list = [];
for (let i = 0; i < data_list.length; i++) {
  if (data_list[i] == "") {
    continue;
  }
  file_list.push(open(data_list[i], "b"));
}
let idx = 0;

let scenes = __ENV.SCENES.split("|");

export default function() {
  let params = {
    headers: {
      "Content-Type": "application/json",
      Authorization: "QiniuStub uid=1&ut=2"
    }
  };
  //console.log(idx);
  // 随机选取一张图片
  let file_content = file_list[idx];
  idx = (idx + 1) % file_list.length;
  //   console.log(idx + " " + data_list[idx] + "   " + data_list.length);

  let payload = JSON.stringify({
    data: {
      uri:
        "data:application/octet-stream;base64," +
        encoding.b64encode(file_content)
    },
    params: {
      scenes: scenes
    }
  });

  let res = http.post(url, payload, params);
  // console.log(res.body)
  check(res, {
    "is status 200": r => r.status === 200
  }) || console.log(res.body);
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
