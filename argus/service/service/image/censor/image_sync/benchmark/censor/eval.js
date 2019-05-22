import http from "k6/http";
import { check } from "k6";
import encoding from "k6/encoding";

var url = `http://localhost:9000/v1/eval/${__ENV.APP}`;

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

export default function() {
  let params = {
    headers: {
      "Content-Type": "application/json",
      Authorization: "QiniuStub uid=1&ut=2"
    }
  };
  // 随机选取一张图片
  let file_content = file_list[idx];
  idx = (idx + 1) % file_list.length;

  let payload = JSON.stringify({
    data: {
      uri:
        "data:application/octet-stream;base64," +
        encoding.b64encode(file_content)
    }
  });

  let res = http.post(url, payload, params);
  // console.log(res.body)
  check(res, {
    "ok": r => r.status === 200
  }) || console.log(res.body);
}

function shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
