import http from "k6/http";
import { check } from "k6";

var url = "http://localhost:11000/v1/video/test_wangan_benchmark"; // 图片内容审核 API 接口，如果服务不在本机，需要将 localhost 换为服务器 IP

// 读取所有 benchmark 图片内容，存储至 file_list
let data_list_file = `${__ENV.BENCHMARK_DATA_LIST_FILE}`; //传入变量 传入list地址
let interval = parseFloat(`${__ENV.BENCHMARK_INTERVAL}`)
let data_list = shuffle(open(data_list_file).split("\n"));
let file_list = [];
for (let i = 0; i < data_list.length; i++) {
  if (data_list[i] == "") {
    continue;
  }
  file_list.push(data_list[i]);
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
    data: {
      uri: file_content
    },
    params: {
        vframe: {
            mode: 0,
            interval: interval
        }
    },
    ops: [{
        op: "wangan_mix"
    }]
  });

  console.log(payload)

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
