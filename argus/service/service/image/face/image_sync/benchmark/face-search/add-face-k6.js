import http from "k6/http";
import { check } from "k6";
import encoding from "k6/encoding";

var url = "http://localhost:"+ __ENV.PORT +"/v1/face/groups/" + __ENV.GROUP +"/add"

// 读取所有 benchmark 图片内容，存储至 file_list
let data_list_file = `${__ENV.BENCHMARK_DATA_LIST_FILE}`; //传入变量 传入list地址
let data_list = shuffle(open(data_list_file).split("\n"));
let file_list = [];

let reject = true
if (__ENV.REJECT == "FALSE") {
    reject = false
}

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
      "Content-Type": "application/json"
    }
  };
  //console.log(idx);
  // 随机选取一张图片
  let file_content = file_list[idx];
  idx = (idx + 1) % file_list.length;
  //console.log(idx + " " + data_list[idx] + "   " + data_list.length);

  let payload = JSON.stringify({
    image: {
        uri: "data:application/octet-stream;base64," + encoding.b64encode(file_content)
    },
    params: {
        reject_bad_face: reject
    }
  });

  let res = http.post(url, payload, params);
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
