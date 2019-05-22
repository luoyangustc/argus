import http from "k6/http";
import { check, group } from "k6";
import encoding from "k6/encoding";

var filepath =
  "../../../../res/testdata/image/serving/pulp/set1/Image-tupu-2016-09-01-00-00-327.jpg";

export let options = {
  vus: 5, // 并发请求数
  duration: "30s" // 压测时间，例如 "5m" / "1h"
};

let file_content = open(filepath, "b");

export default function() {
  var params = {
    headers: {
      "Content-Type": "application/json",
      Authorization: "QiniuStub uid=1&ut=2"
    }
  };

  var payload = JSON.stringify({
    data: {
      uri:
        "data:application/octet-stream;base64," +
        encoding.b64encode(file_content)
    }
  });
  group("basic", () => {
    var url = "http://localhost:9100/v1/eval";
    let res = http.post(url, payload, params);
    check(res, {
      "status 200": r => r.status === 200
    });
  });
}
